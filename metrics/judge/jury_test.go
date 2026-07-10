package judge

import (
	"context"
	"encoding/json"
	"testing"
)

func TestJuryMajorityVote(t *testing.T) {
	// 2 pass, 1 fail -> majority pass.
	j := NewJury(
		fakeJudge(`{"passed": true, "score": 0.9, "reason": "a"}`),
		fakeJudge(`{"passed": true, "score": 0.8, "reason": "b"}`),
		fakeJudge(`{"passed": false, "score": 0.3, "reason": "c"}`),
	)
	out, err := j.Ask(context.Background(), "p")
	if err != nil {
		t.Fatal(err)
	}
	var v Verdict
	if err := json.Unmarshal([]byte(out), &v); err != nil {
		t.Fatal(err)
	}
	if !v.Passed {
		t.Errorf("expected majority pass, got %+v", v)
	}
	if v.Score < 0.6 || v.Score > 0.7 {
		t.Errorf("expected mean ~0.67, got %.2f", v.Score)
	}
}

func TestJuryMinorityVote(t *testing.T) {
	// 1 pass, 2 fail -> majority fail.
	j := NewJury(
		fakeJudge(`{"passed": true, "score": 0.9, "reason": "a"}`),
		fakeJudge(`{"passed": false, "score": 0.4, "reason": "b"}`),
		fakeJudge(`{"passed": false, "score": 0.2, "reason": "c"}`),
	)
	out, err := j.Ask(context.Background(), "p")
	if err != nil {
		t.Fatal(err)
	}
	var v Verdict
	if err := json.Unmarshal([]byte(out), &v); err != nil {
		t.Fatal(err)
	}
	if v.Passed {
		t.Errorf("expected majority fail, got %+v", v)
	}
}

func TestJurySingleJudge(t *testing.T) {
	j := NewJury(fakeJudge(`{"passed": true, "score": 1.0, "reason": "solo"}`))
	out, err := j.Ask(context.Background(), "p")
	if err != nil {
		t.Fatal(err)
	}
	if out != `{"passed": true, "score": 1.0, "reason": "solo"}` {
		t.Errorf("single judge should pass through, got %s", out)
	}
}

func TestJuryOneFails(t *testing.T) {
	// One judge errors; the other two still form a verdict.
	bad := func(ctx context.Context, prompt string) (string, error) {
		return "", context.DeadlineExceeded
	}
	j := NewJury(
		bad,
		fakeJudge(`{"passed": true, "score": 0.9, "reason": "a"}`),
		fakeJudge(`{"passed": true, "score": 0.8, "reason": "b"}`),
	)
	out, err := j.Ask(context.Background(), "p")
	if err != nil {
		t.Fatal(err)
	}
	var v Verdict
	if err := json.Unmarshal([]byte(out), &v); err != nil {
		t.Fatal(err)
	}
	if !v.Passed || v.Score != 0.85 {
		t.Errorf("expected pass 0.85 from 2 valid judges, got %+v", v)
	}
}

func TestJuryAllFail(t *testing.T) {
	bad := func(ctx context.Context, prompt string) (string, error) {
		return "", context.DeadlineExceeded
	}
	j := NewJury(bad, bad)
	_, err := j.Ask(context.Background(), "p")
	if err == nil {
		t.Fatal("expected error when all judges fail")
	}
}

func TestSortedVerdicts(t *testing.T) {
	vs := []Verdict{
		{Passed: false, Score: 0.2},
		{Passed: true, Score: 0.9},
		{Passed: true, Score: 0.5},
	}
	sorted := SortedVerdicts(vs)
	if sorted[0].Score != 0.9 || sorted[2].Score != 0.2 {
		t.Errorf("expected descending sort, got %+v", sorted)
	}
}
