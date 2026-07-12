package ingest

import (
	"encoding/json"
	"fmt"

	"github.com/sscodeai/proctor/trajectory"
)

// OpenAI trace format (chat.completions message list, e.g. from a recorded
// OpenAI / compatible agent run):
//
//	{
//	  "input": "find orders for user 42",
//	  "messages": [
//	    {"role": "user", "content": "find orders for user 42"},
//	    {"role": "assistant", "content": "I'll look that up",
//	     "tool_calls": [{"id": "call_1", "type": "function",
//	       "function": {"name": "get_user", "arguments": "{\"user_id\": 42}"}}]},
//	    {"role": "tool", "tool_call_id": "call_1", "content": "{\"id\": 42, \"name\": \"Alice\"}"},
//	    {"role": "assistant", "content": "Alice found"}
//	  ],
//	  "expected_tools": ["get_user", "search_orders"],
//	  "expected": "the orders list"
//	}
//
// The converter flattens the message list into Steps (reasoning for assistant
// text, tool_call for tool invocations, observation for tool results) and
// builds the Turns view for multi-turn metrics.

type openAIMessage struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	ToolCalls []struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
	ToolCallID string `json:"tool_call_id"`
}

type openAITrace struct {
	Input         string          `json:"input"`
	Messages      []openAIMessage `json:"messages"`
	ExpectedTools []string        `json:"expected_tools"`
	Expected      string          `json:"expected"`
	Name          string          `json:"name"`
}

// ParseOpenAITrace converts an OpenAI-format trajectory export into Samples.
// Returns ok=false when the input is not an OpenAI trace envelope.
func ParseOpenAITrace(data []byte) ([]trajectory.Sample, bool, error) {
	var tr openAITrace
	if err := json.Unmarshal(data, &tr); err != nil {
		return nil, false, nil // not our shape; let other parsers try
	}
	// Envelope detection: OpenAI trace has "messages" with role/tool_calls.
	if len(tr.Messages) == 0 {
		return nil, false, nil
	}
	s, err := openAITraceToSample(tr)
	if err != nil {
		return nil, true, err
	}
	return []trajectory.Sample{s}, true, nil
}

func openAITraceToSample(tr openAITrace) (trajectory.Sample, error) {
	s := trajectory.Sample{
		Name:          tr.Name,
		Input:         tr.Input,
		Expected:      tr.Expected,
		ExpectedTools: tr.ExpectedTools,
	}

	stepIdx := 0
	for _, msg := range tr.Messages {
		switch msg.Role {
		case "user":
			if s.Input == "" {
				s.Input = msg.Content
			}
		case "assistant":
			// Tool calls in the assistant message become tool_call steps.
			for _, tc := range msg.ToolCalls {
				args := map[string]any{}
				if tc.Function.Arguments != "" {
					_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				}
				call := trajectory.ToolCall{
					Name: tc.Function.Name,
					Args: args,
					Type: tc.Type,
				}
				s.Steps = append(s.Steps, trajectory.Step{
					Index:    stepIdx,
					Kind:     trajectory.StepToolCall,
					ToolCall: &call,
				})
				s.ToolCalls = append(s.ToolCalls, call)
				stepIdx++
			}
			// Free text becomes a reasoning step (if any, after tool calls
			// so the ordering reads naturally).
			if msg.Content != "" {
				s.Steps = append(s.Steps, trajectory.Step{
					Index: stepIdx,
					Kind:  trajectory.StepReasoning,
					Text:  msg.Content,
				})
				stepIdx++
			}
		case "tool":
			// Tool result becomes an observation step.
			s.Steps = append(s.Steps, trajectory.Step{
				Index: stepIdx,
				Kind:  trajectory.StepObservation,
				Text:  msg.Content,
			})
			stepIdx++
		}
	}

	// Build Turns view.
	var curTurn *trajectory.Turn
	for _, msg := range tr.Messages {
		switch msg.Role {
		case "user":
			s.Turns = append(s.Turns, trajectory.Turn{Role: "user", Content: msg.Content})
			curTurn = nil
		case "assistant":
			t := trajectory.Turn{Role: "assistant", Content: msg.Content}
			for _, tc := range msg.ToolCalls {
				args := map[string]any{}
				if tc.Function.Arguments != "" {
					_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				}
				t.ToolCalls = append(t.ToolCalls, trajectory.ToolCall{Name: tc.Function.Name, Args: args, Type: tc.Type})
			}
			s.Turns = append(s.Turns, t)
			curTurn = &s.Turns[len(s.Turns)-1]
		case "tool":
			// Attach tool result to the previous assistant turn as observation.
			if curTurn != nil {
				curTurn.Content += " [tool: " + msg.Content + "]"
			}
		}
	}

	if s.Name == "" {
		s.Name = "openai_trace"
	}
	if len(s.Steps) == 0 {
		return s, fmt.Errorf("OpenAI trace has no tool calls or text to evaluate")
	}
	return s, nil
}
