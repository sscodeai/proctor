package agent

import (
	"context"
	"strconv"
	"testing"

	"github.com/sscodeai/proctor/metrics/judge"
	"github.com/sscodeai/proctor/trajectory"
)

func fakeVerdict(passed bool, score float64, reason string) judge.JudgeFunc {
	return func(ctx context.Context, prompt string) (string, error) {
		return `{"passed": ` + strconv.FormatBool(passed) + `, "score": ` + strconv.FormatFloat(score, 'f', -1, 64) + `, "reason": "` + reason + `"}`, nil
	}
}

func TestArgumentCorrectness(t *testing.T) {
	j := func(ctx context.Context, prompt string) (string, error) {
		return `{"verdicts": [
			{"step_index": 1, "verdict": "yes", "reason": "ok"},
			{"step_index": 3, "verdict": "no", "reason": "missing order_id"}
		]}`, nil
	}
	m := ArgumentCorrectness{Judge: j, PassThreshold: 0.7}
	s := trajectory.Sample{
		Input: "find order",
		Steps: []trajectory.Step{
			{Index: 1, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "get_order", Args: map[string]any{"id": 5}}},
			{Index: 3, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "search_orders"}},
		},
	}
	r, err := m.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if r.Score != 0.5 {
		t.Errorf("expected score 0.5, got %.2f", r.Score)
	}
	if r.Passed {
		t.Error("expected fail (0.5 < 0.7)")
	}
	if r.StepIdx != 3 {
		t.Errorf("expected fail step 3, got %d", r.StepIdx)
	}
}

func TestArgumentCorrectnessAllPass(t *testing.T) {
	j := func(ctx context.Context, prompt string) (string, error) {
		return `{"verdicts": [{"step_index": 0, "verdict": "yes", "reason": "ok"}]}`, nil
	}
	m := ArgumentCorrectness{Judge: j, PassThreshold: 0.7}
	s := trajectory.Sample{
		Input: "x",
		Steps: []trajectory.Step{
			{Index: 0, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "t"}},
		},
	}
	r, err := m.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Passed || r.Score != 1.0 {
		t.Errorf("expected pass, got %+v", r)
	}
}

func TestTaskCompletion(t *testing.T) {
	m := TaskCompletion{Judge: fakeVerdict(true, 0.9, "done"), PassThreshold: 0.7}
	s := trajectory.Sample{Input: "do it", Steps: []trajectory.Step{{Index: 0, Kind: trajectory.StepReasoning, Text: "ok"}}}
	r, err := m.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Passed || r.Score != 0.9 {
		t.Errorf("expected pass 0.9, got %+v", r)
	}
}

func TestStepEfficiencyRedundantPrecheck(t *testing.T) {
	// Judge says pass, but deterministic pre-check finds repeated calls.
	m := StepEfficiency{Judge: fakeVerdict(true, 0.9, "efficient"), PassThreshold: 0.7}
	s := trajectory.Sample{
		Input: "search",
		Steps: []trajectory.Step{
			{Index: 0, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "s", Args: map[string]any{"q": "x"}}},
			{Index: 1, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "s", Args: map[string]any{"q": "x"}}},
		},
	}
	r, err := m.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if r.Passed {
		t.Errorf("expected fail due to deterministic redundancy, got %+v", r)
	}
}

func TestPlanAdherenceDeviationLocated(t *testing.T) {
	j := fakeVerdict(true, 0.9, "followed")
	m := PlanAdherence{Judge: j, PassThreshold: 0.7}
	s := trajectory.Sample{
		Input: "do task",
		Plan:  "1. search\n2. summarize",
		Steps: []trajectory.Step{
			{Index: 0, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "search"}},
			// plan step 2 "summarize" never executed
		},
	}
	r, err := m.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if r.Passed {
		t.Errorf("expected fail due to plan deviation, got %+v", r)
	}
	if r.StepIdx < 0 {
		t.Errorf("expected deviation step located, got %d", r.StepIdx)
	}
}

func TestToolUse(t *testing.T) {
	j := func(ctx context.Context, prompt string) (string, error) {
		return `{"tool_selection": {"passed": true, "score": 0.9, "reason": "right tools"},
		        "argument_correctness": {"passed": true, "score": 0.8, "reason": "good args"}}`, nil
	}
	m := ToolUse{Judge: j, PassThreshold: 0.7}
	s := trajectory.Sample{
		Input: "multi-turn",
		Turns: []trajectory.Turn{
			{Role: "user", Content: "find order"},
			{Role: "assistant", Content: "looking", ToolCalls: []trajectory.ToolCall{{Name: "search"}}},
		},
	}
	r, err := m.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Passed || r.Score < 0.84 || r.Score > 0.86 {
		t.Errorf("expected pass ~0.85, got %+v", r)
	}
}
