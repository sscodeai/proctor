package judge

import (
	"context"
	"sync"
	"time"
)

// Meter records judge usage: call count, latency, and (when the underlying
// adapter reports it) token usage. Thread-safe.
type Meter struct {
	mu        sync.Mutex
	Calls     int     `json:"calls"`
	TotalMS   int64   `json:"total_ms"`
	InputTok  int     `json:"input_tokens"`
	OutputTok int     `json:"output_tokens"`
	Cost      float64 `json:"cost"`

	// UsageFn, if set, is called after each judge call to capture token/cost.
	// The llmjudge adapter sets it to read per-call usage from its client.
	UsageFn func() Usage
}

// Wrap returns a JudgeFunc that records usage.
func (m *Meter) Wrap(j JudgeFunc) JudgeFunc {
	return func(ctx context.Context, prompt string) (string, error) {
		start := time.Now()
		resp, err := j(ctx, prompt)
		elapsed := time.Since(start)
		m.mu.Lock()
		m.Calls++
		m.TotalMS += elapsed.Milliseconds()
		if m.UsageFn != nil {
			u := m.UsageFn()
			m.InputTok += u.InputTokens
			m.OutputTok += u.OutputTokens
			m.Cost += u.Cost
		}
		m.mu.Unlock()
		return resp, err
	}
}

// Snapshot returns a copy of the accumulated usage.
func (m *Meter) Snapshot() (calls int, totalMS int64, inputTok, outputTok int, cost float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.Calls, m.TotalMS, m.InputTok, m.OutputTok, m.Cost
}

// Usage carries token/cost accounting for a single call.
type Usage struct {
	InputTokens  int
	OutputTokens int
	Cost         float64
}
