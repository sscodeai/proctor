package judge

import (
	"context"
	"encoding/json"
	"testing"
)

func fakeJudge(resp string) JudgeFunc {
	return func(ctx context.Context, prompt string) (string, error) {
		return resp, nil
	}
}

func TestExtractJSONPlain(t *testing.T) {
	got, err := extractJSON(`{"passed": true, "score": 0.9}`)
	if err != nil {
		t.Fatal(err)
	}
	if got != `{"passed": true, "score": 0.9}` {
		t.Errorf("unexpected: %s", got)
	}
}

func TestExtractJSONFenced(t *testing.T) {
	got, err := extractJSON("Here is the result:\n```json\n{\"passed\": false, \"score\": 0.2}\n```\nHope that helps")
	if err != nil {
		t.Fatal(err)
	}
	var v Verdict
	if err := json.Unmarshal([]byte(got), &v); err != nil {
		t.Fatal(err)
	}
	if v.Passed || v.Score != 0.2 {
		t.Errorf("unexpected verdict: %+v", v)
	}
}

func TestExtractJSONProseAround(t *testing.T) {
	got, err := extractJSON(`The answer is {"passed": true, "score": 1.0} as you can see.`)
	if err != nil {
		t.Fatal(err)
	}
	if !contains(got, "passed") {
		t.Errorf("expected JSON extracted, got %s", got)
	}
}

func TestExtractJSONArray(t *testing.T) {
	got, err := extractJSON(`[{"step_index": 1, "verdict": "no"}]`)
	if err != nil {
		t.Fatal(err)
	}
	if got != `[{"step_index": 1, "verdict": "no"}]` {
		t.Errorf("unexpected: %s", got)
	}
}

func TestCallVerdict(t *testing.T) {
	v, err := CallVerdict(context.Background(), fakeJudge(`{"passed": false, "score": 0.3, "reason": "bad"}`), "p")
	if err != nil {
		t.Fatal(err)
	}
	if v.Passed || v.Score != 0.3 || v.Reason != "bad" {
		t.Errorf("unexpected: %+v", v)
	}
}

func TestCallJSON(t *testing.T) {
	var out struct {
		Verdicts []struct {
			StepIndex int    `json:"step_index"`
			Verdict   string `json:"verdict"`
		} `json:"verdicts"`
	}
	err := CallJSON(context.Background(), fakeJudge(`{"verdicts": [{"step_index": 2, "verdict": "no"}]}`), "p", &out)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Verdicts) != 1 || out.Verdicts[0].StepIndex != 2 {
		t.Errorf("unexpected: %+v", out)
	}
}

func TestCallVerdictBadJSON(t *testing.T) {
	_, err := CallVerdict(context.Background(), fakeJudge("not json at all"), "p")
	if err == nil {
		t.Fatal("expected error on non-JSON response")
	}
}

func TestCache(t *testing.T) {
	dir := t.TempDir()
	c, err := NewCache(dir)
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	inner := func(ctx context.Context, prompt string) (string, error) {
		calls++
		return `{"passed": true, "score": 1.0}`, nil
	}
	wrapped := c.Wrap(inner)

	// First call hits inner, second hits cache.
	if _, err := wrapped(context.Background(), "same prompt"); err != nil {
		t.Fatal(err)
	}
	if _, err := wrapped(context.Background(), "same prompt"); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Errorf("expected 1 inner call (cache hit), got %d", calls)
	}
	// Different prompt -> new call.
	if _, err := wrapped(context.Background(), "other prompt"); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Errorf("expected 2 inner calls, got %d", calls)
	}
}

func TestMeter(t *testing.T) {
	m := &Meter{}
	inner := fakeJudge(`{"passed": true, "score": 1.0}`)
	wrapped := m.Wrap(inner)
	if _, err := wrapped(context.Background(), "p"); err != nil {
		t.Fatal(err)
	}
	calls, _, _, _, _ := m.Snapshot()
	if calls != 1 {
		t.Errorf("expected 1 call, got %d", calls)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
