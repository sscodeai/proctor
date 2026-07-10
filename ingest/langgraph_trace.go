package ingest

import (
	"encoding/json"
	"fmt"

	"github.com/hermes/trajectory-eval/trajectory"
)

// LangGraph / LangSmith trace format (simplified): a graph run with nodes
// that may be tool calls or LLM steps, plus edges.
//
//	{
//	  "name": "agent_run_1",
//	  "input": "find orders for user 42",
//	  "expected_tools": ["get_user", "search_orders"],
//	  "nodes": [
//	    {"id": "node_0", "type": "llm", "name": "ChatOpenAI", "input": "...", "output": "I'll look that up"},
//	    {"id": "node_1", "type": "tool", "name": "get_user", "input": {"user_id": 42}, "output": "{\"id\": 42}"},
//	    {"id": "node_2", "type": "tool", "name": "search_orders", "input": {"user_id": 42}, "output": "[...]"}
//	  ],
//	  "edges": [["node_0", "node_1"], ["node_1", "node_2"]]
//	}

type langGraphNode struct {
	ID     string `json:"id"`
	Type   string `json:"type"` // llm | tool | agent
	Name   string `json:"name"`
	Input  any    `json:"input"`
	Output string `json:"output"`
}

type langGraphTrace struct {
	Name          string          `json:"name"`
	Input         string          `json:"input"`
	ExpectedTools []string        `json:"expected_tools"`
	Expected      string          `json:"expected"`
	Nodes         []langGraphNode `json:"nodes"`
	Edges         [][2]string     `json:"edges"`
}

// ParseLangGraphTrace converts a LangGraph-format trajectory export into
// Samples. Returns ok=false when the input is not a LangGraph envelope.
func ParseLangGraphTrace(data []byte) ([]trajectory.Sample, bool, error) {
	var tr langGraphTrace
	if err := json.Unmarshal(data, &tr); err != nil {
		return nil, false, nil
	}
	if len(tr.Nodes) == 0 {
		return nil, false, nil
	}
	s := langGraphTraceToSample(tr)
	return []trajectory.Sample{s}, true, nil
}

func langGraphTraceToSample(tr langGraphTrace) trajectory.Sample {
	s := trajectory.Sample{
		Name:          tr.Name,
		Input:         tr.Input,
		Expected:      tr.Expected,
		ExpectedTools: tr.ExpectedTools,
	}

	for i, n := range tr.Nodes {
		switch n.Type {
		case "tool":
			args := map[string]any{}
			if m, ok := n.Input.(map[string]any); ok {
				args = m
			}
			call := trajectory.ToolCall{Name: n.Name, Args: args, Output: n.Output}
			s.Steps = append(s.Steps, trajectory.Step{
				Index:    i,
				Kind:     trajectory.StepToolCall,
				ToolCall: &call,
			})
			s.ToolCalls = append(s.ToolCalls, call)
			// Tool output as observation.
			if n.Output != "" {
				s.Steps = append(s.Steps, trajectory.Step{
					Index:    i + 1,
					Kind:     trajectory.StepObservation,
					Text:     n.Output,
					SpanID:   n.ID,
				})
			}
		default: // llm / agent → reasoning step
			text := n.Output
			if text == "" {
				text = n.Name
			}
			s.Steps = append(s.Steps, trajectory.Step{
				Index:  i,
				Kind:   trajectory.StepReasoning,
				Text:   text,
				SpanID: n.ID,
			})
		}
	}

	if s.Name == "" {
		s.Name = "langgraph_trace"
	}
	return s
}

// traceToSampleError is a helper for consistent errors (kept for tests).
func traceToSampleError(format string) error {
	return fmt.Errorf("invalid %s trace", format)
}
