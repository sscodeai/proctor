// Package judge provides LLM-as-a-Judge plumbing: the JudgeFunc type,
// JSON-parsing helpers, rate limiting, result caching, and usage metering.
//
// JudgeFunc wraps any LLM endpoint; the llmjudge package adapts
// OpenAI-compatible APIs to it. Tests use fake JudgeFuncs (no real LLM).
package judge

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// JudgeFunc executes one LLM-as-a-Judge call and returns the raw text
// response, which should contain JSON (possibly wrapped in code fences).
// Implementations may wrap with RateLimit / Cache / Meter.
type JudgeFunc func(ctx context.Context, prompt string) (string, error)

// Verdict is the unified return shape of judge prompts (aligned with eval-go).
type Verdict struct {
	Passed bool    `json:"passed"`
	Score  float64 `json:"score"`
	Reason string  `json:"reason"`
}

// CallVerdict runs a judge prompt and parses the {passed,score,reason} verdict.
func CallVerdict(ctx context.Context, j JudgeFunc, prompt string) (Verdict, error) {
	return callVerdict(ctx, j, prompt)
}

// CallJSON runs a judge prompt and unmarshals the JSON response into out.
func CallJSON(ctx context.Context, j JudgeFunc, prompt string, out any) error {
	return callJSON(ctx, j, prompt, out)
}

// callVerdict runs a judge prompt and parses the {passed,score,reason} verdict.
func callVerdict(ctx context.Context, j JudgeFunc, prompt string) (Verdict, error) {
	raw, err := j(ctx, prompt)
	if err != nil {
		return Verdict{}, fmt.Errorf("judge call: %w", err)
	}
	jsonStr, err := extractJSON(raw)
	if err != nil {
		return Verdict{}, fmt.Errorf("extract verdict JSON: %w", err)
	}
	var v Verdict
	if err := json.Unmarshal([]byte(jsonStr), &v); err != nil {
		return Verdict{}, fmt.Errorf("parse verdict %q: %w", jsonStr, err)
	}
	if v.Score < 0 {
		v.Score = 0
	}
	if v.Score > 1 {
		v.Score = 1
	}
	return v, nil
}

// callJSON runs a judge prompt and unmarshals the JSON response into out.
func callJSON(ctx context.Context, j JudgeFunc, prompt string, out any) error {
	raw, err := j(ctx, prompt)
	if err != nil {
		return fmt.Errorf("judge call: %w", err)
	}
	jsonStr, err := extractJSON(raw)
	if err != nil {
		return fmt.Errorf("extract JSON: %w", err)
	}
	if err := json.Unmarshal([]byte(jsonStr), out); err != nil {
		return fmt.Errorf("parse JSON %q: %w", jsonStr, err)
	}
	return nil
}

// extractJSON tolerantly extracts a JSON object/array from a model response,
// stripping markdown code fences and surrounding prose.
func extractJSON(s string) (string, error) {
	s = strings.TrimSpace(s)
	// Strip ```json ... ``` fences.
	if strings.HasPrefix(s, "```") {
		lines := strings.Split(s, "\n")
		if len(lines) >= 3 {
			body := strings.Join(lines[1:], "\n")
			body = strings.TrimSuffix(body, "```")
			s = strings.TrimSpace(body)
		}
	}
	// Find first '{' and last '}' (or '[' and ']') as a loose extraction.
	start, end := -1, -1
	for i, c := range s {
		if c == '{' || c == '[' {
			start = i
			break
		}
	}
	if start == -1 {
		return "", fmt.Errorf("no JSON object/array found in response: %q", truncate(s, 120))
	}
	open, close := s[start], byte('}')
	if open == '[' {
		close = ']'
	}
	for i := len(s) - 1; i >= start; i-- {
		if s[i] == close {
			end = i
			break
		}
	}
	if end == -1 {
		return "", fmt.Errorf("unterminated JSON in response: %q", truncate(s, 120))
	}
	return s[start : end+1], nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
