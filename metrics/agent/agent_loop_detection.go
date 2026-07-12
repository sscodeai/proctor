package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/sscodeai/proctor/trajectory"
)

// AgentLoopDetection detects when an agent is stuck in a loop, porting
// DeepEval's AgentLoopDetectionMetric semantics. Three weighted sub-signals:
//
//   - tool repetition: same (tool name, canonical args) called >= threshold
//   - reasoning stall: adjacent reasoning steps too similar (bigram Jaccard /
//     SequenceMatcher ratio, stopwords removed)
//   - call-graph cycle: a (type:name:input_hash) label repeats on an
//     ancestor path in the span tree
//
// Weights: 0.40 / 0.35 / 0.25 (aligned with DeepEval). Deterministic — no LLM.
type AgentLoopDetection struct {
	RepetitionThreshold  int     // default 3
	StallThreshold       float64 // default 0.85
	ToolRepeatWeight     float64 // default 0.40
	ReasoningStallWeight float64 // default 0.35
	GraphCycleWeight     float64 // default 0.25
}

// NewAgentLoopDetection returns a metric with DeepEval-aligned defaults.
func NewAgentLoopDetection() AgentLoopDetection {
	return AgentLoopDetection{
		RepetitionThreshold:  3,
		StallThreshold:       0.85,
		ToolRepeatWeight:     0.40,
		ReasoningStallWeight: 0.35,
		GraphCycleWeight:     0.25,
	}
}

func (m AgentLoopDetection) Name() string { return "agent_loop_detection" }

func (m AgentLoopDetection) Evaluate(_ context.Context, s trajectory.Sample) (trajectory.Result, error) {
	rep, repStep := m.toolRepetition(s)
	stall, stallStep := m.reasoningStall(s)
	cycle, cycleStep := m.graphCycle(s)

	score := 0.0
	if rep {
		score += m.ToolRepeatWeight
	}
	if stall {
		score += m.ReasoningStallWeight
	}
	if cycle {
		score += m.GraphCycleWeight
	}

	passed := score == 0
	reasons := []string{}
	if rep {
		reasons = append(reasons, fmt.Sprintf("tool repeated >= %d times", m.RepetitionThreshold))
	}
	if stall {
		reasons = append(reasons, "adjacent reasoning steps too similar")
	}
	if cycle {
		reasons = append(reasons, "call-graph cycle detected")
	}

	// Locate the earliest offending step for attribution.
	stepIdx := -1
	for _, i := range []int{repStep, stallStep, cycleStep} {
		if i >= 0 && (stepIdx == -1 || i < stepIdx) {
			stepIdx = i
		}
	}

	return trajectory.Result{
		Metric:  "agent_loop_detection",
		Passed:  passed,
		Score:   score,
		Reason:  strings.Join(reasons, "; "),
		StepIdx: stepIdx,
		Located: stepIdx >= 0,
	}, nil
}

// --- sub-signal 1: tool repetition ---

// canonicalArgs renders args deterministically so that semantically identical
// calls compare equal.
func canonicalArgs(args map[string]any) string {
	if len(args) == 0 {
		return ""
	}
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		v, _ := json.Marshal(args[k])
		fmt.Fprintf(&b, "%s=%s;", k, v)
	}
	return b.String()
}

func (m AgentLoopDetection) toolRepetition(s trajectory.Sample) (bool, int) {
	counts := map[string]int{}
	firstIdx := map[string]int{}
	for _, st := range s.Steps {
		if st.Kind != trajectory.StepToolCall || st.ToolCall == nil {
			continue
		}
		key := st.ToolCall.Name + "(" + canonicalArgs(st.ToolCall.Args) + ")"
		counts[key]++
		if _, ok := firstIdx[key]; !ok {
			firstIdx[key] = st.Index
		}
	}
	rep := false
	step := -1
	for key, n := range counts {
		if n >= m.RepetitionThreshold {
			rep = true
			if step == -1 || firstIdx[key] < step {
				step = firstIdx[key]
			}
		}
	}
	return rep, step
}

// --- sub-signal 2: reasoning stall ---

var stopwords = map[string]struct{}{
	"the": {}, "a": {}, "an": {}, "and": {}, "or": {}, "of": {}, "to": {},
	"in": {}, "on": {}, "for": {}, "with": {}, "is": {}, "are": {}, "was": {},
	"were": {}, "be": {}, "been": {}, "i": {}, "we": {}, "you": {}, "it": {},
	"that": {}, "this": {}, "then": {}, "now": {}, "so": {}, "as": {},
}

func tokens(s string) []string {
	fields := strings.Fields(strings.ToLower(s))
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		f = strings.Trim(f, ".,;:!?()[]{}'\"")
		if f == "" {
			continue
		}
		if _, stop := stopwords[f]; stop {
			continue
		}
		out = append(out, f)
	}
	return out
}

func bigrams(ts []string) map[string]int {
	out := map[string]int{}
	for i := 0; i+1 < len(ts); i++ {
		out[ts[i]+" "+ts[i+1]]++
	}
	return out
}

func jaccard(a, b map[string]int) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	intersect := 0
	for k := range a {
		if _, ok := b[k]; ok {
			intersect++
		}
	}
	union := len(a) + len(b) - intersect
	if union == 0 {
		return 0
	}
	return float64(intersect) / float64(union)
}

func (m AgentLoopDetection) reasoningStall(s trajectory.Sample) (bool, int) {
	var reasoning []trajectory.Step
	for _, st := range s.Steps {
		if st.Kind == trajectory.StepReasoning {
			reasoning = append(reasoning, st)
		}
	}
	for i := 1; i < len(reasoning); i++ {
		prev := bigrams(tokens(reasoning[i-1].Text))
		cur := bigrams(tokens(reasoning[i].Text))
		if jaccard(prev, cur) >= m.StallThreshold {
			return true, reasoning[i].Index
		}
	}
	return false, -1
}

// --- sub-signal 3: call-graph cycle ---

func (m AgentLoopDetection) graphCycle(s trajectory.Sample) (bool, int) {
	if s.Spans == nil {
		return false, -1
	}
	seen := map[string]bool{}
	step := -1
	var walk func(sp trajectory.Span, depth int) bool
	walk = func(sp trajectory.Span, depth int) bool {
		if depth > 100 { // safety bound
			return false
		}
		inputJSON, _ := json.Marshal(sp.Input)
		label := sp.Type + ":" + sp.Name + ":" + string(inputJSON)
		if seen[label] {
			return true
		}
		seen[label] = true
		for _, c := range sp.Children {
			if walk(c, depth+1) {
				return true
			}
		}
		return false
	}
	if walk(*s.Spans, 0) {
		// Without explicit step mapping in spans, report the first tool step
		// as the cycle anchor (best effort).
		for _, st := range s.Steps {
			if st.Kind == trajectory.StepToolCall {
				step = st.Index
				break
			}
		}
		return true, step
	}
	return false, -1
}
