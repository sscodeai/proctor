package agent

import (
	"context"
	"fmt"

	"github.com/sscodeai/proctor/metrics/judge"
	"github.com/sscodeai/proctor/trajectory"
)

// PlanQuality asks the judge whether the agent's stated plan is sound and
// complete for the task.
type PlanQuality struct {
	Judge         judge.JudgeFunc
	PassThreshold float64
}

func (m PlanQuality) Name() string { return "plan_quality" }

func (m PlanQuality) Evaluate(ctx context.Context, s trajectory.Sample) (trajectory.Result, error) {
	if s.Plan == "" {
		return trajectory.Result{Metric: "plan_quality", Passed: true, Score: 1, Reason: "no plan (skipped)"}, nil
	}
	prompt := fmt.Sprintf(
		"Evaluate the quality of the agent's plan for the task: is it sound, complete, and well-ordered?\n\nTask: %s\n\nPlan: %s\n\n"+
			"Return JSON: {\"passed\": true|false, \"score\": <0..1>, \"reason\": \"<short>\"}",
		s.Input, s.Plan)
	v, err := judge.CallVerdict(ctx, m.Judge, prompt)
	if err != nil {
		return trajectory.Result{}, err
	}
	return trajectory.Result{
		Metric: "plan_quality",
		Passed: v.Passed && v.Score >= m.PassThreshold,
		Score:  v.Score,
		Reason: v.Reason,
	}, nil
}

// PlanAdherence checks whether execution followed the stated plan. Uses a
// deterministic weighted-LCS alignment first; a judge scores the overall
// adherence. On deviation, the first unaligned step is located for
// attribution.
type PlanAdherence struct {
	Judge         judge.JudgeFunc
	PassThreshold float64
}

func (m PlanAdherence) Name() string { return "plan_adherence" }

func (m PlanAdherence) Evaluate(ctx context.Context, s trajectory.Sample) (trajectory.Result, error) {
	// Deterministic: locate the first plan step that has no matching
	// trajectory step (weighted LCS alignment).
	planSteps := splitPlan(s.Plan)
	deviationStep := -1
	if len(planSteps) > 0 {
		deviationStep = firstPlanDeviation(planSteps, s.Steps)
	}

	prompt := fmt.Sprintf(
		"Evaluate whether the agent's execution adhered to its stated plan.\n\nTask: %s\n\nPlan: %s\n\nExecution steps:\n%s\n\n"+
			"Return JSON: {\"passed\": true|false, \"score\": <0..1>, \"reason\": \"<short>\"}",
		s.Input, s.Plan, formatStepsForPrompt(s.Steps))
	v, err := judge.CallVerdict(ctx, m.Judge, prompt)
	if err != nil {
		return trajectory.Result{}, err
	}

	// Deterministic deviation is strong evidence of non-adherence.
	if deviationStep >= 0 && v.Score > 0.5 {
		v.Score = 0.3
		v.Passed = false
		v.Reason = fmt.Sprintf("plan step not followed (first deviation ~step %d); %s", deviationStep, v.Reason)
	}

	return trajectory.Result{
		Metric:  "plan_adherence",
		Passed:  v.Passed && v.Score >= m.PassThreshold,
		Score:   v.Score,
		Reason:  v.Reason,
		StepIdx: deviationStep,
		Located: deviationStep >= 0,
	}, nil
}

// splitPlan splits a plan string into a step sequence (line or numbered item).
func splitPlan(plan string) []string {
	var out []string
	cur := ""
	flush := func() {
		if cur != "" {
			out = append(out, cur)
			cur = ""
		}
	}
	for _, r := range plan {
		if r == '\n' {
			flush()
			continue
		}
		cur += string(r)
	}
	flush()
	return out
}

// firstPlanDeviation aligns plan steps against trajectory text and returns the
// first plan index that has no matching execution step (weighted LCS-style
// greedy alignment). Returns -1 if all plan steps match.
func firstPlanDeviation(planSteps []string, steps []trajectory.Step) int {
	execText := make([]string, 0, len(steps))
	for _, st := range steps {
		t := st.Text
		if st.ToolCall != nil {
			t += " " + st.ToolCall.Name
		}
		execText = append(execText, t)
	}

	// Greedy forward scan: each plan step must appear (as a fuzzy substring)
	// in some execution step, in order.
	execIdx := 0
	for _, ps := range planSteps {
		matched := false
		for execIdx < len(execText) {
			if planStepMatches(ps, execText[execIdx]) {
				matched = true
				execIdx++
				break
			}
			execIdx++
		}
		if !matched {
			return execIdx // first unaligned execution position
		}
	}
	return -1
}

// planStepMatches does fuzzy containment: plan step tokens (minus stopwords)
// substantially appear in the execution text.
func planStepMatches(plan, exec string) bool {
	pt := tokens(plan)
	if len(pt) == 0 {
		return true
	}
	et := tokens(exec)
	if len(et) == 0 {
		return false
	}
	// Fraction of plan tokens appearing in exec.
	execSet := map[string]bool{}
	for _, t := range et {
		execSet[t] = true
	}
	hits := 0
	for _, t := range pt {
		if execSet[t] {
			hits++
		}
	}
	return float64(hits)/float64(len(pt)) >= 0.5
}
