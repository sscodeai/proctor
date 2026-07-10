package report

import (
	"encoding/json"
	"fmt"
	"os"
)

// DiffItem is one sample's status change between V1 and V2.
type DiffItem struct {
	Sample     string `json:"sample"`
	From       string `json:"from"` // "PASS" | "FAIL"
	To         string `json:"to"`
	Change     string `json:"change"` // regressed | fixed | unchanged
	RootCauseChanged bool `json:"root_cause_changed,omitempty"`
}

// Diff compares two reports and classifies per-sample changes.
func Diff(v1, v2 *Report) []DiffItem {
	byName := map[string]SampleReport{}
	for _, s := range v2.Samples {
		byName[s.Sample] = s
	}
	fromName := map[string]SampleReport{}
	for _, s := range v1.Samples {
		fromName[s.Sample] = s
	}

	var out []DiffItem
	for _, s2 := range v2.Samples {
		s1, ok := fromName[s2.Sample]
		if !ok {
			out = append(out, DiffItem{Sample: s2.Sample, From: "-", To: status(s2.Passed), Change: "added"})
			continue
		}
		change := "unchanged"
		if s1.Passed != s2.Passed {
			if s2.Passed {
				change = "fixed"
			} else {
				change = "regressed"
			}
		}
		rcChanged := false
		if s1.Attribution != nil && s2.Attribution != nil &&
			s1.Attribution.RootCause.Category != s2.Attribution.RootCause.Category {
			rcChanged = true
		}
		out = append(out, DiffItem{
			Sample: s2.Sample, From: status(s1.Passed), To: status(s2.Passed),
			Change: change, RootCauseChanged: rcChanged,
		})
	}
	return out
}

// Gate is a CI gate definition.
type Gate struct {
	Metrics     []string          `json:"metrics,omitempty"`     // must all pass; empty = all
	MinScores   map[string]float64 `json:"min_scores,omitempty"` // per-metric minimum score
	MinPassRate float64           `json:"min_pass_rate"`         // global pass rate floor
}

// Evaluate returns whether the report passes the gate, plus failure reasons.
func (g Gate) Evaluate(r *Report) (bool, []string) {
	var failures []string
	// Global pass rate.
	passedSamples := 0
	for _, s := range r.Samples {
		if s.Passed {
			passedSamples++
		}
	}
	rate := 0.0
	if len(r.Samples) > 0 {
		rate = float64(passedSamples) / float64(len(r.Samples))
	}
	if rate < g.MinPassRate {
		failures = append(failures, fmt.Sprintf("pass rate %.0f%% < min %.0f%%", rate*100, g.MinPassRate*100))
	}

	// Per-metric checks (only when Metrics is explicitly non-empty).
	if len(g.Metrics) > 0 {
		for _, ms := range r.Summary {
			if !contains(g.Metrics, ms.Metric) {
				continue
			}
			if ms.PassRate < 1.0 {
				failures = append(failures, fmt.Sprintf("%s pass rate %.0f%%", ms.Metric, ms.PassRate*100))
			}
			if min, ok := g.MinScores[ms.Metric]; ok && ms.MeanScore < min {
				failures = append(failures, fmt.Sprintf("%s mean score %.2f < %.2f", ms.Metric, ms.MeanScore, min))
			}
		}
	}
	return len(failures) == 0, failures
}

// LoadReport reads a report JSON file.
func LoadReport(path string) (*Report, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r Report
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

func status(passed bool) string {
	if passed {
		return "PASS"
	}
	return "FAIL"
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}
