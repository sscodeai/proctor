// Package compare builds a model × dataset comparison matrix from multiple
// evaluation reports. Given reports labeled by (model, dataset), it produces
// a grid where each cell shows the pass rate / mean score / root-cause mix
// for that combination — the "which model should I pick" answer at a glance.
package compare

import (
	"fmt"
	"sort"
	"strings"

	"github.com/sscodeai/proctor/attribution"
	"github.com/sscodeai/proctor/report"
)

// Entry associates a report with its (model, dataset) labels.
type Entry struct {
	Model   string // e.g. "deepseek-v4-flash"
	Dataset string // e.g. "customer_service"
	Report  *report.Report
}

// Cell is one (model, dataset) combination's aggregate.
type Cell struct {
	Model        string  `json:"model"`
	Dataset      string  `json:"dataset"`
	PassRate     float64 `json:"pass_rate"`
	MeanScore    float64 `json:"mean_score"`
	Samples      int     `json:"samples"`
	Passed       int     `json:"passed"`
	TopRootCause string  `json:"top_root_cause,omitempty"`
	RCFrac       float64 `json:"root_cause_fraction,omitempty"` // fraction of failed samples with the top root cause
}

// Matrix is the full comparison grid.
type Matrix struct {
	Schema   string   `json:"schema"` // "proctor-compare/v1"
	Models   []string `json:"models"`
	Datasets []string `json:"datasets"`
	Cells    []Cell   `json:"cells"`
}

// Build assembles a Matrix from entries.
func Build(entries []Entry) Matrix {
	m := Matrix{Schema: "proctor-compare/v1"}

	modelSet := map[string]bool{}
	dsSet := map[string]bool{}
	for _, e := range entries {
		modelSet[e.Model] = true
		dsSet[e.Dataset] = true
	}
	m.Models = sortedKeys(modelSet)
	m.Datasets = sortedKeys(dsSet)

	byKey := map[string]Entry{}
	for _, e := range entries {
		byKey[e.Model+"\x00"+e.Dataset] = e
	}

	// One cell per model × dataset that exists.
	for _, model := range m.Models {
		for _, ds := range m.Datasets {
			e, ok := byKey[model+"\x00"+ds]
			if !ok {
				continue
			}
			c := aggregate(e)
			m.Cells = append(m.Cells, c)
		}
	}
	sort.Slice(m.Cells, func(i, j int) bool {
		if m.Cells[i].Model != m.Cells[j].Model {
			return m.Cells[i].Model < m.Cells[j].Model
		}
		return m.Cells[i].Dataset < m.Cells[j].Dataset
	})
	return m
}

func aggregate(e Entry) Cell {
	r := e.Report
	c := Cell{Model: e.Model, Dataset: e.Dataset, Samples: len(r.Samples)}
	scoreSum := 0.0
	rcCounts := map[attribution.RootCauseCategory]int{}
	failed := 0
	for _, s := range r.Samples {
		if s.Passed {
			c.Passed++
		} else {
			failed++
			if s.Attribution != nil {
				rcCounts[s.Attribution.RootCause.Category]++
			}
		}
		for _, res := range s.Results {
			scoreSum += res.Score
		}
	}
	if c.Samples > 0 {
		c.PassRate = float64(c.Passed) / float64(c.Samples)
		// Mean across all metric results.
		totalResults := 0
		for _, s := range r.Samples {
			totalResults += len(s.Results)
		}
		if totalResults > 0 {
			c.MeanScore = scoreSum / float64(totalResults)
		}
	}
	// Top root cause among failed samples.
	if failed > 0 {
		top := attribution.CauseUnknown
		topN := 0
		for rc, n := range rcCounts {
			if n > topN {
				top = rc
				topN = n
			}
		}
		c.TopRootCause = string(top)
		c.RCFrac = float64(topN) / float64(failed)
	}
	return c
}

// Markdown renders the matrix as a human-readable table.
func (m Matrix) Markdown() string {
	var b strings.Builder
	b.WriteString("| Model | Dataset | Pass Rate | Mean Score | Passed/Total | Top Root Cause |\n")
	b.WriteString("|-------|---------|-----------|------------|--------------|----------------|\n")
	for _, c := range m.Cells {
		rc := c.TopRootCause
		if rc == "" {
			rc = "-"
		} else {
			rc = fmt.Sprintf("%s (%.0f%%)", rc, c.RCFrac*100)
		}
		fmt.Fprintf(&b, "| %s | %s | %.0f%% | %.2f | %d/%d | %s |\n",
			c.Model, c.Dataset, c.PassRate*100, c.MeanScore, c.Passed, c.Samples, rc)
	}
	return b.String()
}

// Best returns the model with the highest overall pass rate (weighted by
// datasets present). Ties broken by mean score.
func (m Matrix) Best() (model string, passRate float64, meanScore float64) {
	agg := map[string]struct {
		passed, total int
		scoreSum      float64
		scoreN        int
	}{}
	for _, c := range m.Cells {
		a := agg[c.Model]
		a.passed += c.Passed
		a.total += c.Samples
		a.scoreSum += c.MeanScore * float64(c.Samples)
		a.scoreN += c.Samples
		agg[c.Model] = a
	}
	best := ""
	bestRate := -1.0
	bestMean := 0.0
	for model, a := range agg {
		if a.total == 0 {
			continue
		}
		rate := float64(a.passed) / float64(a.total)
		mean := 0.0
		if a.scoreN > 0 {
			mean = a.scoreSum / float64(a.scoreN)
		}
		if rate > bestRate || (rate == bestRate && mean > bestMean) {
			best, bestRate, bestMean = model, rate, mean
		}
	}
	return best, bestRate, bestMean
}

func sortedKeys(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
