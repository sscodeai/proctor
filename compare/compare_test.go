package compare

import (
	"strings"
	"testing"

	"github.com/hermes/trajectory-eval/attribution"
	"github.com/hermes/trajectory-eval/report"
	"github.com/hermes/trajectory-eval/trajectory"
)

func mkReport(commit string, sampleResults map[string][]trajectory.Result, attribs map[string]*attribution.Attribution) *report.Report {
	r := report.Build(commit, sampleResults, attribs, nil)
	return &r
}

func mkResult(metric string, passed bool, score float64) trajectory.Result {
	return trajectory.Result{Metric: metric, Passed: passed, Score: score}
}

func TestBuildMatrix(t *testing.T) {
	// Model A: 2/3 pass. Model B: 1/3 pass. Same dataset.
	makeEntries := func() []Entry {
		ra := mkReport("v1", map[string][]trajectory.Result{
			"s1": {mkResult("tool_correctness", true, 1.0)},
			"s2": {mkResult("tool_correctness", true, 1.0)},
			"s3": {mkResult("tool_correctness", false, 0.3)},
		}, map[string]*attribution.Attribution{
			"s3": {RootCause: attribution.RootCause{Category: attribution.CauseWrongTool, Confidence: 1.0}},
		})
		rb := mkReport("v1", map[string][]trajectory.Result{
			"s1": {mkResult("tool_correctness", true, 1.0)},
			"s2": {mkResult("tool_correctness", false, 0.2)},
			"s3": {mkResult("tool_correctness", false, 0.1)},
		}, map[string]*attribution.Attribution{
			"s2": {RootCause: attribution.RootCause{Category: attribution.CauseBadArguments, Confidence: 1.0}},
			"s3": {RootCause: attribution.RootCause{Category: attribution.CauseBadArguments, Confidence: 1.0}},
		})
		return []Entry{
			{Model: "model-a", Dataset: "ds1", Report: ra},
			{Model: "model-b", Dataset: "ds1", Report: rb},
		}
	}

	m := Build(makeEntries())
	if len(m.Models) != 2 || len(m.Datasets) != 1 {
		t.Fatalf("expected 2 models 1 dataset, got %v %v", m.Models, m.Datasets)
	}
	if len(m.Cells) != 2 {
		t.Fatalf("expected 2 cells, got %d", len(m.Cells))
	}

	// model-a cell: 2/3 pass.
	cellA := m.Cells[0]
	if cellA.Model != "model-a" || cellA.Passed != 2 || cellA.Samples != 3 {
		t.Errorf("model-a cell wrong: %+v", cellA)
	}
	if cellA.PassRate != 2.0/3.0 {
		t.Errorf("model-a pass rate %.2f", cellA.PassRate)
	}
	if cellA.TopRootCause != "wrong_tool" {
		t.Errorf("model-a top rc: %s", cellA.TopRootCause)
	}

	best, rate, _ := m.Best()
	if best != "model-a" {
		t.Errorf("expected model-a best, got %s", best)
	}
	if rate != 2.0/3.0 {
		t.Errorf("expected rate 0.667, got %.3f", rate)
	}
}

func TestMatrixMarkdown(t *testing.T) {
	ra := mkReport("v1", map[string][]trajectory.Result{
		"s1": {mkResult("tool_correctness", true, 1.0)},
	}, nil)
	m := Build([]Entry{{Model: "m1", Dataset: "d1", Report: ra}})
	md := m.Markdown()
	if !strings.Contains(md, "| m1 | d1 |") {
		t.Errorf("markdown missing cell: %s", md)
	}
	if !strings.Contains(md, "Pass Rate") {
		t.Errorf("markdown missing header: %s", md)
	}
}

func TestMultipleDatasets(t *testing.T) {
	ra1 := mkReport("v1", map[string][]trajectory.Result{
		"s1": {mkResult("tool_correctness", true, 1.0)},
	}, nil)
	ra2 := mkReport("v1", map[string][]trajectory.Result{
		"s1": {mkResult("tool_correctness", false, 0.0)},
	}, nil)
	m := Build([]Entry{
		{Model: "m", Dataset: "d1", Report: ra1},
		{Model: "m", Dataset: "d2", Report: ra2},
	})
	if len(m.Datasets) != 2 {
		t.Fatalf("expected 2 datasets, got %v", m.Datasets)
	}
	if len(m.Cells) != 2 {
		t.Fatalf("expected 2 cells, got %d", len(m.Cells))
	}
	best, rate, _ := m.Best()
	if best != "m" || rate != 0.5 {
		t.Errorf("expected m 0.5, got %s %.2f", best, rate)
	}
}
