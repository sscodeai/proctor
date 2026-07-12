package agent

import (
	"context"
	"testing"

	"github.com/sscodeai/proctor/trajectory"
)

func TestLoopToolRepetition(t *testing.T) {
	m := NewAgentLoopDetection()
	// Same tool + same args called 3 times.
	steps := []trajectory.Step{}
	for i := 0; i < 3; i++ {
		steps = append(steps, mkStep(i, trajectory.StepToolCall, "", toolCall("search", map[string]any{"q": "same"})))
	}
	s := trajectory.Sample{Steps: steps}
	r, err := m.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if r.Passed {
		t.Errorf("expected loop detected, got %+v", r)
	}
	if r.Score != m.ToolRepeatWeight {
		t.Errorf("expected score %.2f (tool repeat only), got %.2f", m.ToolRepeatWeight, r.Score)
	}
	if r.StepIdx != 0 {
		t.Errorf("expected step 0 located, got %d", r.StepIdx)
	}
}

func TestLoopNoRepetition(t *testing.T) {
	m := NewAgentLoopDetection()
	steps := []trajectory.Step{}
	for i := 0; i < 3; i++ {
		steps = append(steps, mkStep(i, trajectory.StepToolCall, "", toolCall("search", map[string]any{"q": "query" + string(rune('a'+i))})))
	}
	s := trajectory.Sample{Steps: steps}
	r, err := m.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Passed {
		t.Errorf("expected no loop (different args), got %+v", r)
	}
}

func TestLoopReasoningStall(t *testing.T) {
	m := NewAgentLoopDetection()
	// Two nearly identical reasoning steps.
	s := trajectory.Sample{Steps: []trajectory.Step{
		mkStep(0, trajectory.StepReasoning, "I need to search for the order details first", nil),
		mkStep(1, trajectory.StepReasoning, "I need to search for the order details first", nil),
	}}
	r, err := m.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if r.Passed {
		t.Errorf("expected reasoning stall detected, got %+v", r)
	}
	if r.Score != m.ReasoningStallWeight {
		t.Errorf("expected score %.2f, got %.2f", m.ReasoningStallWeight, r.Score)
	}
	if r.StepIdx != 1 {
		t.Errorf("expected step 1 located, got %d", r.StepIdx)
	}
}

func TestLoopGraphCycle(t *testing.T) {
	m := NewAgentLoopDetection()
	// Span tree with a repeated (type:name:input) label on an ancestor path.
	child := trajectory.Span{Type: "tool", Name: "search", Input: "q=orders"}
	root := trajectory.Span{
		Type: "agent", Name: "main", Input: "task",
		Children: []trajectory.Span{
			{Type: "agent", Name: "sub", Input: "step",
				Children: []trajectory.Span{child}},
			child, // repeated label -> cycle
		},
	}
	s := trajectory.Sample{Spans: &root, Steps: []trajectory.Step{
		mkStep(0, trajectory.StepToolCall, "", toolCall("search", nil)),
	}}
	r, err := m.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if r.Passed {
		t.Errorf("expected graph cycle detected, got %+v", r)
	}
	if r.Score != m.GraphCycleWeight {
		t.Errorf("expected score %.2f, got %.2f", m.GraphCycleWeight, r.Score)
	}
}

func TestLoopNone(t *testing.T) {
	m := NewAgentLoopDetection()
	s := trajectory.Sample{Steps: []trajectory.Step{
		mkStep(0, trajectory.StepReasoning, "first", nil),
		mkStep(1, trajectory.StepToolCall, "", toolCall("a", map[string]any{"x": 1})),
		mkStep(2, trajectory.StepObservation, "done", nil),
	}}
	r, err := m.Evaluate(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Passed || r.Score != 0 {
		t.Errorf("expected clean pass, got %+v", r)
	}
	if r.StepIdx != -1 {
		t.Errorf("expected no step, got %d", r.StepIdx)
	}
}
