// Package metrics defines the Metric interface and the deterministic agent
// metrics (tool_correctness, agent_loop_detection) plus the judge plumbing.
//
// Deterministic metrics run offline with zero LLM cost; judge-based metrics
// (added in later milestones) plug in via JudgeFunc.
package metrics

import (
	"context"
	"fmt"

	"github.com/sscodeai/proctor/trajectory"
)

// JudgeFunc runs an LLM judge prompt and returns the raw text response.
// Implementations live in the llmjudge package (OpenAI-compatible) or in
// tests as fakes.
type JudgeFunc func(ctx context.Context, prompt string) (string, error)

// Verdict is the parsed result of a judge call.
type Verdict struct {
	Passed  bool
	Score   float64 // normalized 0..1
	Reason  string
	StepIdx int // optional: step the verdict refers to (for attribution)
}

// Metric evaluates one Sample and returns a Result.
type Metric interface {
	Name() string
	Evaluate(ctx context.Context, s trajectory.Sample) (trajectory.Result, error)
}

// MetricFunc adapts a function to the Metric interface.
type MetricFunc struct {
	Fn func(ctx context.Context, s trajectory.Sample) (trajectory.Result, error)
}

func (m MetricFunc) Name() string { return "" }

func (m MetricFunc) Evaluate(ctx context.Context, s trajectory.Sample) (trajectory.Result, error) {
	return m.Fn(ctx, s)
}

// namedMetric wraps a MetricFunc with an explicit name.
type namedMetric struct {
	name string
	Metric
}

func (n namedMetric) Name() string { return n.name }

// Named wraps a MetricFunc with a name, since MetricFunc itself cannot carry one.
func Named(name string, m Metric) Metric {
	return namedMetric{name: name, Metric: m}
}

// RunAll evaluates a sample against every metric in order, stopping on the
// first error.
func RunAll(ctx context.Context, s trajectory.Sample, ms []Metric) ([]trajectory.Result, error) {
	results := make([]trajectory.Result, 0, len(ms))
	for _, m := range ms {
		r, err := m.Evaluate(ctx, s)
		if err != nil {
			return nil, fmt.Errorf("metric %q: %w", m.Name(), err)
		}
		r.Metric = m.Name()
		results = append(results, r)
	}
	return results, nil
}
