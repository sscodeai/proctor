package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/sscodeai/proctor/metrics/judge"
	"github.com/sscodeai/proctor/trajectory"
)

// ToolUse evaluates multi-turn tool usage: whether the right tools were
// selected at the right times and with correct arguments across a
// conversation. This is the DeepEval ToolUseMetric equivalent.
type ToolUse struct {
	Judge         judge.JudgeFunc
	PassThreshold float64
}

func (m ToolUse) Name() string { return "tool_use" }

func (m ToolUse) Evaluate(ctx context.Context, s trajectory.Sample) (trajectory.Result, error) {
	if len(s.Turns) == 0 {
		// Fall back to single-turn trajectory if no turns provided.
		if len(s.ToolCallsFromSteps()) == 0 {
			return trajectory.Result{Metric: "tool_use", Passed: true, Score: 1, Reason: "no turns (skipped)"}, nil
		}
	}

	prompt := m.buildPrompt(s)
	var resp struct {
		ToolSelection struct {
			Passed bool    `json:"passed"`
			Score  float64 `json:"score"`
			Reason string  `json:"reason"`
		} `json:"tool_selection"`
		ArgumentCorrectness struct {
			Passed bool    `json:"passed"`
			Score  float64 `json:"score"`
			Reason string  `json:"reason"`
		} `json:"argument_correctness"`
	}
	if err := judge.CallJSON(ctx, m.Judge, prompt, &resp); err != nil {
		return trajectory.Result{}, err
	}

	// Combine both sub-scores (aligned with DeepEval: overall = tool selection
	// then argument correctness).
	score := (resp.ToolSelection.Score + resp.ArgumentCorrectness.Score) / 2
	passed := resp.ToolSelection.Passed && resp.ArgumentCorrectness.Passed && score >= m.PassThreshold
	reason := fmt.Sprintf("tool_selection: %s; argument_correctness: %s",
		resp.ToolSelection.Reason, resp.ArgumentCorrectness.Reason)

	return trajectory.Result{
		Metric: "tool_use",
		Passed: passed,
		Score:  score,
		Reason: reason,
	}, nil
}

func (m ToolUse) buildPrompt(s trajectory.Sample) string {
	var b []byte
	b = append(b, []byte("Evaluate the multi-turn tool usage of this conversation.\n\nTask: "+s.Input+"\n\nConversation:\n")...)
	for i, t := range s.Turns {
		line := fmt.Sprintf("[%d] %s: %s", i, t.Role, t.Content)
		if len(t.ToolCalls) > 0 {
			args, _ := json.Marshal(t.ToolCalls)
			line += fmt.Sprintf(" (tools: %s)", args)
		}
		b = append(b, []byte(line+"\n")...)
	}
	b = append(b, []byte("\nReturn JSON:\n{\n\"tool_selection\": {\"passed\": bool, \"score\": <0..1>, \"reason\": \"<short>\"},\n\"argument_correctness\": {\"passed\": bool, \"score\": <0..1>, \"reason\": \"<short>\"}\n}")...)
	return string(b)
}
