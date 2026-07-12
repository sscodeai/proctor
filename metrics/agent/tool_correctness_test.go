package agent

import (
	"context"
	"testing"

	"github.com/sscodeai/proctor/trajectory"
)

func mkStep(idx int, kind trajectory.StepKind, text string, tc *trajectory.ToolCall) trajectory.Step {
	return trajectory.Step{Index: idx, Kind: kind, Text: text, ToolCall: tc}
}

func toolCall(name string, args map[string]any) *trajectory.ToolCall {
	return &trajectory.ToolCall{Name: name, Args: args}
}

func TestToolCorrectnessExactMatch(t *testing.T) {
	s := trajectory.Sample{
		ExpectedTools: []string{"search_orders", "get_user"},
		Steps: []trajectory.Step{
			mkStep(0, trajectory.StepReasoning, "need to find user", nil),
			mkStep(1, trajectory.StepToolCall, "", toolCall("get_user", nil)),
			mkStep(2, trajectory.StepToolCall, "", toolCall("search_orders", nil)),
		},
	}
	r, err := ToolCorrectness{}.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Passed {
		t.Errorf("expected pass, got %+v", r)
	}
	if r.Score != 1.0 {
		t.Errorf("expected score 1.0, got %.2f", r.Score)
	}
	if r.StepIdx != -1 {
		t.Errorf("expected no step located, got %d", r.StepIdx)
	}
}

func TestToolCorrectnessMissingAndExtra(t *testing.T) {
	s := trajectory.Sample{
		ExpectedTools: []string{"search_orders", "get_user"},
		Steps: []trajectory.Step{
			mkStep(0, trajectory.StepReasoning, "start", nil),
			mkStep(1, trajectory.StepToolCall, "", toolCall("get_user", nil)),
			mkStep(2, trajectory.StepToolCall, "", toolCall("delete_order", nil)), // extra + wrong
		},
	}
	r, err := ToolCorrectness{}.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if r.Passed {
		t.Errorf("expected fail, got %+v", r)
	}
	// matched=1 (get_user), union={get_user, search_orders, delete_order}=3 => 1/3
	if r.Score != 1.0/3.0 {
		t.Errorf("expected score %.2f, got %.2f", 1.0/3.0, r.Score)
	}
	// The extra tool "delete_order" is at step 2.
	if r.StepIdx != 2 {
		t.Errorf("expected step 2 located, got %d", r.StepIdx)
	}
}

func TestToolCorrectnessNoExpected(t *testing.T) {
	s := trajectory.Sample{
		Steps: []trajectory.Step{mkStep(0, trajectory.StepReasoning, "hi", nil)},
	}
	r, err := ToolCorrectness{}.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Passed {
		t.Errorf("expected skip-pass, got %+v", r)
	}
}

func TestToolCorrectnessOrderIndependent(t *testing.T) {
	// Order of calls should not matter for the set-based metric.
	s := trajectory.Sample{
		ExpectedTools: []string{"a", "b"},
		Steps: []trajectory.Step{
			mkStep(0, trajectory.StepToolCall, "", toolCall("b", nil)),
			mkStep(1, trajectory.StepToolCall, "", toolCall("a", nil)),
		},
	}
	r, err := ToolCorrectness{}.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Passed || r.Score != 1.0 {
		t.Errorf("expected pass regardless of order, got %+v", r)
	}
}

func TestToolCorrectnessUsesStepsProjection(t *testing.T) {
	// ToolCallsFromSteps should pick up tool_call steps even if ToolCalls field
	// is empty (convenience view is derived).
	s := trajectory.Sample{
		ExpectedTools: []string{"x"},
		Steps: []trajectory.Step{
			mkStep(0, trajectory.StepToolCall, "", toolCall("x", nil)),
		},
		ToolCalls: nil,
	}
	r, err := ToolCorrectness{}.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Passed {
		t.Errorf("expected pass via steps projection, got %+v", r)
	}
}
