package attribution

import (
	"context"
	"testing"

	"github.com/hermes/trajectory-eval/metrics"
	"github.com/hermes/trajectory-eval/metrics/agent"
	"github.com/hermes/trajectory-eval/trajectory"
)

func sampleWithWrongTool() trajectory.Sample {
	return trajectory.Sample{
		Name:          "wrong_tool_golden",
		Input:         "Find orders for user 42",
		ExpectedTools: []string{"get_user", "search_orders"},
		Steps: []trajectory.Step{
			{Index: 0, Kind: trajectory.StepReasoning, Text: "I need user first"},
			{Index: 1, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "get_user"}},
			{Index: 2, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "delete_order"}}, // wrong tool
			{Index: 3, Kind: trajectory.StepObservation, Text: "order deleted"},
		},
	}
}

func TestAnalyzeLocatesWrongTool(t *testing.T) {
	s := sampleWithWrongTool()
	results, err := metrics.RunAll(context.Background(), s, []metrics.Metric{
		metrics.Named("tool_correctness", agent.ToolCorrectness{}),
		metrics.Named("agent_loop_detection", agent.NewAgentLoopDetection()),
	})
	if err != nil {
		t.Fatal(err)
	}

	a := Analyzer{Thresholds: DefaultThresholds()}
	attr, err := a.Analyze(context.Background(), s, results)
	if err != nil {
		t.Fatal(err)
	}

	if len(attr.KeyFailures) == 0 {
		t.Fatalf("expected key failures, got %+v", attr)
	}
	// The wrong tool (delete_order) is at step 2 -> earliest deviation = 2.
	if attr.CausalChain[0] != 2 {
		t.Errorf("expected causal chain to start at step 2, got %v", attr.CausalChain)
	}
	if attr.RootCause.Category != CauseWrongTool {
		t.Errorf("expected root cause wrong_tool, got %s", attr.RootCause.Category)
	}
	if attr.RootCause.Confidence != 1.0 {
		t.Errorf("expected confidence 1.0 for deterministic rule, got %.2f", attr.RootCause.Confidence)
	}
	// Every failure step must carry evidence.
	for _, f := range attr.KeyFailures {
		if len(f.Evidence) == 0 {
			t.Errorf("failure at step %d has no evidence", f.StepIndex)
		}
	}
}

func TestAnalyzeCleanRunNoAttribution(t *testing.T) {
	s := trajectory.Sample{
		Name:          "clean",
		Input:         "hi",
		ExpectedTools: []string{"greet"},
		Steps: []trajectory.Step{
			{Index: 0, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "greet"}},
		},
	}
	results, err := metrics.RunAll(context.Background(), s, []metrics.Metric{
		metrics.Named("tool_correctness", agent.ToolCorrectness{}),
		metrics.Named("agent_loop_detection", agent.NewAgentLoopDetection()),
	})
	if err != nil {
		t.Fatal(err)
	}
	a := Analyzer{Thresholds: DefaultThresholds()}
	attr, err := a.Analyze(context.Background(), s, results)
	if err != nil {
		t.Fatal(err)
	}
	if len(attr.KeyFailures) != 0 {
		t.Errorf("expected no failures on clean run, got %+v", attr.KeyFailures)
	}
	if attr.RootCause.Category != CauseUnknown {
		t.Errorf("expected unknown root cause on clean run, got %s", attr.RootCause.Category)
	}
}

func TestAnalyzeLoopRootCause(t *testing.T) {
	s := trajectory.Sample{
		Name:          "loop",
		Input:         "search repeatedly",
		ExpectedTools: []string{"search"},
		Steps: []trajectory.Step{
			{Index: 0, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "search", Args: map[string]any{"q": "x"}}},
			{Index: 1, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "search", Args: map[string]any{"q": "x"}}},
			{Index: 2, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "search", Args: map[string]any{"q": "x"}}},
		},
	}
	results, err := metrics.RunAll(context.Background(), s, []metrics.Metric{
		metrics.Named("tool_correctness", agent.ToolCorrectness{}),
		metrics.Named("agent_loop_detection", agent.NewAgentLoopDetection()),
	})
	if err != nil {
		t.Fatal(err)
	}
	a := Analyzer{Thresholds: DefaultThresholds()}
	attr, err := a.Analyze(context.Background(), s, results)
	if err != nil {
		t.Fatal(err)
	}
	if attr.RootCause.Category != CauseRedundantLoop {
		t.Errorf("expected redundant_loop root cause, got %s", attr.RootCause.Category)
	}
	if attr.CausalChain[0] != 0 {
		t.Errorf("expected causal chain to start at step 0, got %v", attr.CausalChain)
	}
}

func TestJudgeFallback(t *testing.T) {
	// task_completion fails but no deterministic metric locates a step:
	// judge fallback should kick in.
	s := trajectory.Sample{
		Name:          "judge_fallback",
		Input:         "do the thing",
		ExpectedTools: []string{"search"},
		Steps: []trajectory.Step{
			{Index: 0, Kind: trajectory.StepReasoning, Text: "thinking about it"},
			{Index: 1, Kind: trajectory.StepToolCall, ToolCall: &trajectory.ToolCall{Name: "search", Args: map[string]any{"q": "y"}}},
		},
	}
	results := []trajectory.Result{
		{Metric: "tool_correctness", Passed: true, Score: 1.0},
		{Metric: "agent_loop_detection", Passed: true, Score: 0},
		{Metric: "task_completion", Passed: false, Score: 0.4, Reason: "incomplete"},
	}

	a := Analyzer{
		Judge: func(ctx context.Context, prompt string) (string, error) {
			return `{"step_index": 1, "summary": "tool returned empty result", "confidence": 0.8}`, nil
		},
		Thresholds: DefaultThresholds(),
	}
	attr, err := a.Analyze(context.Background(), s, results)
	if err != nil {
		t.Fatal(err)
	}
	if len(attr.KeyFailures) == 0 {
		t.Fatalf("expected judge fallback failure, got %+v", attr)
	}
	if attr.KeyFailures[0].StepIndex != 1 {
		t.Errorf("expected judge to locate step 1, got %d", attr.KeyFailures[0].StepIndex)
	}
	if attr.RootCause.Confidence != 0.8 {
		t.Errorf("expected confidence 0.8 from judge, got %.2f", attr.RootCause.Confidence)
	}
}
