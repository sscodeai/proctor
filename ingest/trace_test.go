package ingest

import (
	"testing"

	"github.com/hermes/trajectory-eval/trajectory"
)

func TestParseOpenAITrace(t *testing.T) {
	data := []byte(`{
		"name": "order_run",
		"input": "find orders for user 42",
		"expected_tools": ["get_user", "search_orders"],
		"messages": [
			{"role": "user", "content": "find orders for user 42"},
			{"role": "assistant", "content": "I'll look that up",
			 "tool_calls": [{"id": "c1", "type": "function",
			   "function": {"name": "get_user", "arguments": "{\"user_id\": 42}"}}]},
			{"role": "tool", "tool_call_id": "c1", "content": "{\"id\": 42, \"name\": \"Alice\"}"},
			{"role": "assistant", "content": "Alice found"}
		]
	}`)

	samples, ok, err := ParseOpenAITrace(data)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || len(samples) != 1 {
		t.Fatalf("expected 1 sample, ok=%v got %d", ok, len(samples))
	}
	s := samples[0]
	if s.Input != "find orders for user 42" {
		t.Errorf("input: %s", s.Input)
	}
	// Steps: assistant tool_call (get_user) + observation + reasoning.
	if len(s.ToolCalls) != 1 || s.ToolCalls[0].Name != "get_user" {
		t.Errorf("tool calls: %+v", s.ToolCalls)
	}
	if s.ToolCalls[0].Args["user_id"] != float64(42) {
		t.Errorf("args parsed: %+v", s.ToolCalls[0].Args)
	}
	// Steps should contain tool_call, observation, reasoning.
	kinds := map[trajectory.StepKind]bool{}
	for _, st := range s.Steps {
		kinds[st.Kind] = true
	}
	if !kinds[trajectory.StepToolCall] || !kinds[trajectory.StepObservation] {
		t.Errorf("expected tool_call + observation steps, got kinds %v", kinds)
	}
	// Turns view built.
	if len(s.Turns) == 0 {
		t.Error("expected turns view")
	}
}

func TestParseOpenAITraceNotTrace(t *testing.T) {
	// A plain samples array should not be treated as an OpenAI trace.
	data := []byte(`{"samples": [{"name": "a", "input": "x"}]}`)
	_, ok, err := ParseOpenAITrace(data)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Error("expected ok=false for non-trace envelope")
	}
}

func TestParseLangGraphTrace(t *testing.T) {
	data := []byte(`{
		"name": "lg_run",
		"input": "find orders",
		"expected_tools": ["get_user"],
		"nodes": [
			{"id": "n0", "type": "llm", "name": "Chat", "output": "searching"},
			{"id": "n1", "type": "tool", "name": "get_user", "input": {"user_id": 42}, "output": "{\"id\": 42}"}
		],
		"edges": [["n0", "n1"]]
	}`)

	samples, ok, err := ParseLangGraphTrace(data)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || len(samples) != 1 {
		t.Fatalf("expected 1 sample, ok=%v got %d", ok, len(samples))
	}
	s := samples[0]
	if len(s.ToolCalls) != 1 || s.ToolCalls[0].Name != "get_user" {
		t.Errorf("tool calls: %+v", s.ToolCalls)
	}
	if s.ToolCalls[0].Args["user_id"] != float64(42) {
		t.Errorf("args: %+v", s.ToolCalls[0].Args)
	}
	// Has reasoning (llm node) + tool_call + observation.
	found := map[trajectory.StepKind]bool{}
	for _, st := range s.Steps {
		found[st.Kind] = true
	}
	if !found[trajectory.StepReasoning] || !found[trajectory.StepToolCall] || !found[trajectory.StepObservation] {
		t.Errorf("expected all step kinds, got %v", found)
	}
}

func TestParseLangGraphNotTrace(t *testing.T) {
	data := []byte(`[{"name": "a"}]`)
	_, ok, err := ParseLangGraphTrace(data)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Error("expected ok=false for array")
	}
}

func TestParseSamplesDetectsTraces(t *testing.T) {
	// OpenAI trace content with .json extension should route to the converter.
	openai := []byte(`{"input": "q", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "tool_calls": [{"function": {"name": "t", "arguments": "{}"}}]}]}`)
	samples, err := ParseSamples(openai, "run.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 1 || len(samples[0].ToolCalls) != 1 {
		t.Errorf("expected 1 sample with 1 tool call, got %+v", samples)
	}
}
