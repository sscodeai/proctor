package report

import (
	"bytes"
	"strings"
	"testing"

	"github.com/hermes/trajectory-eval/attribution"
	"github.com/hermes/trajectory-eval/trajectory"
)

func TestBuildAndWriteJSON(t *testing.T) {
	perSample := map[string][]trajectory.Result{
		"s1": {
			{Metric: "tool_correctness", Passed: true, Score: 1.0},
		},
		"s2": {
			{Metric: "tool_correctness", Passed: false, Score: 0.5, Reason: "missing: search_orders"},
		},
	}
	attr := &attribution.Attribution{
		KeyFailures: []attribution.FailureStep{
			{StepIndex: 2, Summary: "wrong tool", Evidence: []attribution.Evidence{{Source: "metric:tool_correctness", Assertion: "missing: search_orders"}}},
		},
		RootCause:   attribution.RootCause{Category: attribution.CauseWrongTool, Detail: "missing expected tool", Confidence: 1.0},
		CausalChain: []int{2},
	}
	r := Build("v1", perSample, map[string]*attribution.Attribution{"s2": attr}, nil)

	if r.Schema != Schema {
		t.Errorf("expected schema %s, got %s", Schema, r.Schema)
	}
	if !r.Failed {
		t.Error("expected report failed (s2 failed)")
	}
	if len(r.Summary) != 1 {
		t.Fatalf("expected 1 metric summary, got %d", len(r.Summary))
	}
	if r.Summary[0].PassRate != 0.5 {
		t.Errorf("expected pass rate 0.5, got %.2f", r.Summary[0].PassRate)
	}

	var buf bytes.Buffer
	if err := r.WriteJSON(&buf); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), `"schema": "traj-eval-report/v1"`) {
		t.Errorf("json missing schema: %s", buf.String())
	}
	if !strings.Contains(buf.String(), `"root_cause"`) {
		t.Errorf("json missing attribution: %s", buf.String())
	}
}

func TestWriteMarkdown(t *testing.T) {
	perSample := map[string][]trajectory.Result{
		"s1": {
			{Metric: "tool_correctness", Passed: true, Score: 1.0, Reason: "ok"},
		},
	}
	r := Build("v1", perSample, nil, nil)

	var buf bytes.Buffer
	if err := r.WriteMarkdown(&buf); err != nil {
		t.Fatal(err)
	}
	md := buf.String()
	if !strings.Contains(md, "Trajectory Evaluation Report") {
		t.Errorf("markdown missing title: %s", md)
	}
	if !strings.Contains(md, "PASS") {
		t.Errorf("markdown missing PASS status: %s", md)
	}
}
