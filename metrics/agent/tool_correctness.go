// Package agent implements the agent trajectory metrics:
// deterministic (tool_correctness, agent_loop_detection) and judge-based
// (argument_correctness, task_completion, ...) — the latter added in later
// milestones. This package is stdlib-only for the deterministic metrics.
package agent

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/sscodeai/proctor/trajectory"
)

// ToolCorrectness checks the tools the agent actually invoked against the
// ground-truth ExpectedTools (order-independent set comparison). Score is the
// Jaccard overlap so missing AND extraneous tool calls both cost points; the
// metric passes only on an exact set match. Deterministic — no LLM.
//
// It also records the earliest step index of an unexpected tool call in
// Result.StepIdx, so the attribution layer can locate the failure.
type ToolCorrectness struct{}

func (ToolCorrectness) Name() string { return "tool_correctness" }

func (ToolCorrectness) Evaluate(_ context.Context, s trajectory.Sample) (trajectory.Result, error) {
	if len(s.ExpectedTools) == 0 {
		return trajectory.Result{
			Metric: "tool_correctness",
			Passed: true,
			Score:  1,
			Reason: "no expected tools (skipped)",
		}, nil
	}

	actual := toolNames(s.ToolCallsFromSteps())
	expected := make(map[string]struct{}, len(s.ExpectedTools))
	for _, t := range s.ExpectedTools {
		expected[t] = struct{}{}
	}
	actualSet := make(map[string]struct{}, len(actual))
	for _, t := range actual {
		actualSet[t] = struct{}{}
	}

	union := make(map[string]struct{}, len(expected)+len(actualSet))
	for t := range expected {
		union[t] = struct{}{}
	}
	for t := range actualSet {
		union[t] = struct{}{}
	}

	matched := 0
	for t := range expected {
		if _, ok := actualSet[t]; ok {
			matched++
		}
	}

	var missing, extra []string
	for t := range expected {
		if _, ok := actualSet[t]; !ok {
			missing = append(missing, t)
		}
	}
	for t := range actualSet {
		if _, ok := expected[t]; !ok {
			extra = append(extra, t)
		}
	}
	sort.Strings(missing)
	sort.Strings(extra)

	score := float64(matched) / float64(len(union))
	passed := len(missing) == 0 && len(extra) == 0

	reason := fmt.Sprintf("%d/%d expected tools called", matched, len(expected))
	if len(missing) > 0 {
		reason += "; missing: " + strings.Join(missing, ", ")
	}
	if len(extra) > 0 {
		reason += "; unexpected: " + strings.Join(extra, ", ")
	}

	// Locate the earliest unexpected tool_call step for attribution.
	stepIdx := -1
	if len(extra) > 0 {
		extraSet := make(map[string]struct{}, len(extra))
		for _, e := range extra {
			extraSet[e] = struct{}{}
		}
		for _, st := range s.Steps {
			if st.Kind == trajectory.StepToolCall && st.ToolCall != nil {
				if _, ok := extraSet[st.ToolCall.Name]; ok {
					stepIdx = st.Index
					break
				}
			}
		}
	}

	return trajectory.Result{
		Metric:  "tool_correctness",
		Passed:  passed,
		Score:   score,
		Reason:  reason,
		StepIdx: stepIdx,
		Located: stepIdx >= 0,
	}, nil
}

func toolNames(calls []trajectory.ToolCall) []string {
	names := make([]string, 0, len(calls))
	for _, c := range calls {
		names = append(names, c.Name)
	}
	return names
}
