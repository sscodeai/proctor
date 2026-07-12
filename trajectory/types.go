// Package trajectory defines the core data model for agent trajectory
// evaluation: the Sample under test, structured steps, tool calls, and the
// result of running metrics against a sample.
//
// The package is stdlib-only (no third-party dependencies), following the
// philosophy of the eval-go skeleton this project is built on.
package trajectory

// StepKind enumerates the kinds of steps that can appear in an agent
// trajectory.
type StepKind string

const (
	// StepReasoning is a reasoning / thought step (no external side effect).
	StepReasoning StepKind = "reasoning"
	// StepToolCall is a tool invocation step.
	StepToolCall StepKind = "tool_call"
	// StepObservation is a step recording the observation after a tool call
	// (tool output, retrieved context, etc.).
	StepObservation StepKind = "observation"
)

// ToolCall is a single tool invocation recorded from an agent run.
type ToolCall struct {
	Name   string         `json:"name"`             // tool / function name invoked
	Args   map[string]any `json:"args,omitempty"`   // arguments passed to the tool
	Output string         `json:"output,omitempty"` // tool return value (for judge / attribution)
	Type   string         `json:"type,omitempty"`   // tool type (function / mcp / ...)
}

// Step is one ordered step in a trajectory. Unlike eval-go's flat []string,
// Step is structured so the attribution layer can locate "which step" failed.
type Step struct {
	Index    int       `json:"index"`
	Kind     StepKind  `json:"kind"` // reasoning | tool_call | observation
	Text     string    `json:"text,omitempty"`
	ToolCall *ToolCall `json:"tool_call,omitempty"`
	SpanID   string    `json:"span_id,omitempty"` // passthrough from tracing systems
}

// Span is an optional tree model of a trajectory, used by agent loop
// detection for call-graph cycle detection. Aligns with DeepEval's trace
// dict children tree.
type Span struct {
	Type     string `json:"type"` // llm | tool | agent | ...
	Name     string `json:"name"`
	Input    any    `json:"input,omitempty"`
	Output   string `json:"output,omitempty"`
	Children []Span `json:"children,omitempty"`
}

// Turn is one message in a multi-turn conversation (for tool_use metric).
type Turn struct {
	Role      string     `json:"role"`                 // "user" | "assistant"
	Content   string     `json:"content"`              // message text
	ToolCalls []ToolCall `json:"tool_calls,omitempty"` // tool calls made in this turn
}

// Sample is one row of a golden dataset plus the system's actual trajectory.
// It is the record-then-evaluate model: any framework can emit these fields
// and be scored the same way.
type Sample struct {
	Name     string            `json:"name"`
	Input    string            `json:"input"`
	Output   string            `json:"output,omitempty"`
	Expected string            `json:"expected,omitempty"`
	Context  []string          `json:"context,omitempty"`
	Rubric   string            `json:"rubric,omitempty"`
	Meta     map[string]string `json:"meta,omitempty"`

	// --- trajectory (structured) ---
	Plan          string     `json:"plan,omitempty"`           // the agent's stated plan
	Steps         []Step     `json:"steps,omitempty"`          // ordered trajectory steps
	ToolCalls     []ToolCall `json:"tool_calls,omitempty"`     // convenience view (projection of tool_call steps)
	ExpectedTools []string   `json:"expected_tools,omitempty"` // ground-truth tool names for tool_correctness
	Spans         *Span      `json:"spans,omitempty"`          // tree for loop detection

	// --- multi-turn tool-use ---
	Turns   []Turn `json:"turns,omitempty"`
	Persona string `json:"persona,omitempty"`

	// Labels carries human gold judgments per metric name, normalized 0..1
	// (binary: 1=pass, 0=fail). Used by the align command to measure
	// judge-vs-human agreement.
	Labels map[string]float64 `json:"labels,omitempty"`
}

// ToolCallsFromSteps projects the tool_call steps of a trajectory into a
// flat []ToolCall in order. Used by metrics that only care about tool calls.
func (s Sample) ToolCallsFromSteps() []ToolCall {
	var out []ToolCall
	for _, st := range s.Steps {
		if st.Kind == StepToolCall && st.ToolCall != nil {
			out = append(out, *st.ToolCall)
		}
	}
	return out
}

// Result is the outcome of one metric against one sample.
type Result struct {
	Metric  string  `json:"metric"`
	Passed  bool    `json:"passed"`
	Score   float64 `json:"score"`              // normalized 0..1
	Reason  string  `json:"reason,omitempty"`   // human-readable explanation
	StepIdx int     `json:"step_idx,omitempty"` // optional: step located by a deterministic metric (for attribution)
	Located bool    `json:"-"`                  // true when StepIdx was explicitly set by a locating metric
}

// Report is the top-level outcome of evaluating a dataset with a set of
// metrics. Kept minimal here; the report package renders it.
type Report struct {
	Schema  string              `json:"schema"` // fixed "proctor-report/v1"
	Commit  string              `json:"commit,omitempty"`
	Samples []Sample            `json:"samples"`
	Results map[string][]Result `json:"results"` // sample name -> results
}
