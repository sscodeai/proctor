// Package attribution implements causal failure attribution for agent
// trajectories: locate the key failing steps, build an evidence chain, and
// classify the root cause. This is the differentiation core of trajectory-eval.
//
// Methodology: AgentRx's "constraint -> check -> locate" pipeline fused with
// TrajDebug's "expected vs actual backtracking". Deterministic-first, judge
// fallback — attribution works with zero LLM calls.
package attribution

import (
	"context"
	"fmt"
	"sort"

	"github.com/hermes/trajectory-eval/metrics/judge"
	"github.com/hermes/trajectory-eval/trajectory"
)

// FailureStep is the minimal attribution output: one key failing step.
type FailureStep struct {
	StepIndex  int                 `json:"step_index"`
	StepKind   trajectory.StepKind `json:"step_kind"`
	Summary    string              `json:"summary"`    // what went wrong at this step
	Constraint string              `json:"constraint"` // the violated constraint (AgentRx semantics)
	Evidence   []Evidence          `json:"evidence"`   // evidence chain
}

// Evidence is one piece of evidence: who asserts, what, and on what basis.
type Evidence struct {
	Source    string `json:"source"`    // deterministic | metric:<name> | judge
	Assertion string `json:"assertion"` // the evidence claim
	Detail    string `json:"detail"`    // supporting detail (missing/extra, similarity, ...)
}

// RootCauseCategory enumerates the root cause classes.
type RootCauseCategory string

const (
	CauseWrongTool        RootCauseCategory = "wrong_tool"        // selected the wrong tool
	CauseBadArguments     RootCauseCategory = "bad_arguments"     // argument error
	CausePlanDeviation    RootCauseCategory = "plan_deviation"    // deviated from plan
	CauseMissingStep      RootCauseCategory = "missing_step"      // skipped a required step
	CauseRedundantLoop    RootCauseCategory = "redundant_loop"    // loop / repetition
	CauseInsufficientInfo RootCauseCategory = "insufficient_info" // not enough information
	CauseUnknown          RootCauseCategory = "unknown"
)

// RootCause is the root cause classification result.
type RootCause struct {
	Category   RootCauseCategory `json:"category"`
	Detail     string            `json:"detail"`
	Confidence float64           `json:"confidence"` // 0..1
}

// Attribution is the complete output of one analysis.
type Attribution struct {
	KeyFailures []FailureStep `json:"key_failures"`
	RootCause   RootCause     `json:"root_cause"`
	CausalChain []int         `json:"causal_chain"` // ordered failing step indices
}

// Analyzer runs the attribution pipeline.
type Analyzer struct {
	Judge      judge.JudgeFunc // nil = pure deterministic attribution
	Thresholds Thresholds
}

// Thresholds tunes the deterministic detectors.
type Thresholds struct {
	SimilarityThreshold float64 // default 0.85
	RepetitionThreshold int     // default 3
}

// DefaultThresholds returns the defaults.
func DefaultThresholds() Thresholds {
	return Thresholds{SimilarityThreshold: 0.85, RepetitionThreshold: 3}
}

// Analyze performs the full pipeline on a sample and its metric results.
func (a Analyzer) Analyze(ctx context.Context, s trajectory.Sample, results []trajectory.Result) (Attribution, error) {
	// ① ② Constraint check: collect deterministic failure locations from
	// metric results that explicitly located a step (Located=true).
	var failures []FailureStep
	for _, r := range results {
		if r.Passed || !r.Located {
			continue
		}
		st := stepAt(s, r.StepIdx)
		failures = append(failures, FailureStep{
			StepIndex:  r.StepIdx,
			StepKind:   stepKindOf(st),
			Summary:    r.Reason,
			Constraint: constraintFor(r.Metric),
			Evidence: []Evidence{{
				Source:    "metric:" + r.Metric,
				Assertion: r.Reason,
				Detail:    fmt.Sprintf("score=%.2f step=%d", r.Score, r.StepIdx),
			}},
		})
	}

	// ③ Backtracking (TrajDebug): earliest deviation -> causal chain.
	earliest := earliestDeviation(failures)
	causalChain := []int{}
	if earliest >= 0 {
		causalChain = append(causalChain, earliest)
		// Append later failing steps after the earliest, in order.
		seen := map[int]bool{earliest: true}
		for _, f := range failures {
			if f.StepIndex > earliest && !seen[f.StepIndex] {
				causalChain = append(causalChain, f.StepIndex)
				seen[f.StepIndex] = true
			}
		}
	}

	// ④⑤ Root cause classification: rule mapping first, judge fallback.
	root := classify(failures)

	// Judge fallback: if no deterministic failure located but the run overall
	// failed (e.g. task_completion), ask the judge for the most suspicious step.
	if root.Category == CauseUnknown && a.Judge != nil && hasFailure(results) {
		if jr, err := a.judgeFallback(ctx, s); err == nil && jr.StepIndex >= 0 {
			failures = append(failures, FailureStep{
				StepIndex:  jr.StepIndex,
				StepKind:   stepKindOf(stepAt(s, jr.StepIndex)),
				Summary:    jr.Summary,
				Constraint: "judge-identified",
				Evidence: []Evidence{{
					Source:    "judge",
					Assertion: jr.Summary,
					Detail:    fmt.Sprintf("confidence=%.2f", jr.Confidence),
				}},
			})
			sort.Slice(failures, func(i, j int) bool { return failures[i].StepIndex < failures[j].StepIndex })
			root = RootCause{CauseUnknown, jr.Summary, jr.Confidence}
		}
	}

	// Sort failures by step index for a readable causal chain.
	sort.Slice(failures, func(i, j int) bool { return failures[i].StepIndex < failures[j].StepIndex })

	return Attribution{
		KeyFailures: failures,
		RootCause:   root,
		CausalChain: causalChain,
	}, nil
}

// judgeResult is the parsed judge fallback output.
type judgeResult struct {
	StepIndex  int
	Summary    string
	Confidence float64
}

func (a Analyzer) judgeFallback(ctx context.Context, s trajectory.Sample) (judgeResult, error) {
	prompt := fmt.Sprintf(
		"Given the following agent trajectory, identify the single most suspicious step "+
			"that caused the task to fail.\n\nTask: %s\n\nTrajectory steps:\n%s\n\n"+
			"Return JSON: {\"step_index\": <int>, \"summary\": \"<short reason>\", \"confidence\": <0..1>}",
		s.Input, formatSteps(s.Steps))
	resp, err := a.Judge(ctx, prompt)
	if err != nil {
		return judgeResult{}, err
	}
	idx, summary, conf, err := parseJudgeJSON(resp)
	if err != nil {
		return judgeResult{}, err
	}
	return judgeResult{StepIndex: idx, Summary: summary, Confidence: conf}, nil
}

// --- helpers ---

func stepAt(s trajectory.Sample, idx int) *trajectory.Step {
	for i := range s.Steps {
		if s.Steps[i].Index == idx {
			return &s.Steps[i]
		}
	}
	return nil
}

func stepKindOf(st *trajectory.Step) trajectory.StepKind {
	if st == nil {
		return trajectory.StepReasoning
	}
	return st.Kind
}

func earliestDeviation(failures []FailureStep) int {
	earliest := -1
	for _, f := range failures {
		if earliest == -1 || f.StepIndex < earliest {
			earliest = f.StepIndex
		}
	}
	return earliest
}

func hasFailure(results []trajectory.Result) bool {
	for _, r := range results {
		if !r.Passed {
			return true
		}
	}
	return false
}

func constraintFor(metric string) string {
	switch metric {
	case "tool_correctness":
		return "expected tool set must be called exactly"
	case "agent_loop_detection":
		return "no loops / repeated work"
	case "argument_correctness":
		return "tool arguments must be correct and sufficient"
	case "plan_adherence":
		return "execution must follow the plan"
	case "plan_quality":
		return "plan must be sound and complete"
	case "task_completion":
		return "task must be completed"
	case "step_efficiency":
		return "steps must be minimal"
	default:
		return "unknown constraint"
	}
}

func formatSteps(steps []trajectory.Step) string {
	var b []byte
	for _, st := range steps {
		line := fmt.Sprintf("%d. [%s] %s", st.Index, st.Kind, st.Text)
		if st.ToolCall != nil {
			line += fmt.Sprintf(" (tool: %s)", st.ToolCall.Name)
		}
		b = append(b, []byte(line+"\n")...)
	}
	return string(b)
}
