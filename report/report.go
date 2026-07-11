// Package report renders evaluation reports: machine-readable JSON and
// human-readable Markdown, plus summary computation.
package report

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/hermes/trajectory-eval/attribution"
	"github.com/hermes/trajectory-eval/trajectory"
)

// Schema is the fixed report schema version.
const Schema = "traj-eval-report/v1"

// SampleReport is one sample's metrics + attribution + trajectory.
type SampleReport struct {
	Sample      string                   `json:"sample"`
	Meta        map[string]string        `json:"meta,omitempty"`
	Steps       []trajectory.Step        `json:"steps,omitempty"`
	Results     []trajectory.Result      `json:"results"`
	Attribution *attribution.Attribution `json:"attribution,omitempty"`
	Passed      bool                     `json:"passed"`
}

// MetricSummary aggregates one metric across all samples.
type MetricSummary struct {
	Metric    string  `json:"metric"`
	PassRate  float64 `json:"pass_rate"`
	MeanScore float64 `json:"mean_score"`
	Passed    int     `json:"passed"`
	Total     int     `json:"total"`
}

// Usage records judge/token usage (filled by later milestones).
type Usage struct {
	LLMCalls  int     `json:"llm_calls,omitempty"`
	TotalCost float64 `json:"total_cost,omitempty"`
}

// Report is the top-level evaluation report.
type Report struct {
	Schema  string          `json:"schema"`
	Commit  string          `json:"commit,omitempty"`
	Samples []SampleReport  `json:"samples"`
	Summary []MetricSummary `json:"summary"`
	Usage   *Usage          `json:"usage,omitempty"`
	Failed  bool            `json:"failed"`
}

// Build assembles a Report from per-sample results.
// steps carries each sample's trajectory for the visualization UI.
func Build(commit string, perSample map[string][]trajectory.Result, attrib map[string]*attribution.Attribution, steps map[string][]trajectory.Step) Report {
	names := make([]string, 0, len(perSample))
	for n := range perSample {
		names = append(names, n)
	}
	sort.Strings(names)

	metricTotals := map[string]*MetricSummary{}
	r := Report{Schema: Schema, Commit: commit, Failed: false}

	for _, name := range names {
		results := perSample[name]
		passed := true
		for _, res := range results {
			if !res.Passed {
				passed = false
				r.Failed = true
			}
			ms, ok := metricTotals[res.Metric]
			if !ok {
				ms = &MetricSummary{Metric: res.Metric}
				metricTotals[res.Metric] = ms
			}
			ms.Total++
			ms.MeanScore += res.Score
			if res.Passed {
				ms.Passed++
			}
		}
		sr := SampleReport{Sample: name, Results: results, Passed: passed}
		if st, ok := steps[name]; ok {
			sr.Steps = st
		}
		if a, ok := attrib[name]; ok {
			sr.Attribution = a
		}
		r.Samples = append(r.Samples, sr)
	}

	// Compute summaries.
	names = make([]string, 0, len(metricTotals))
	for n := range metricTotals {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		ms := metricTotals[n]
		if ms.Total > 0 {
			ms.PassRate = float64(ms.Passed) / float64(ms.Total)
			ms.MeanScore = ms.MeanScore / float64(ms.Total)
		}
		r.Summary = append(r.Summary, *ms)
	}
	return r
}

// WriteJSON writes the report as indented JSON.
func (r Report) WriteJSON(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}

// WriteMarkdown writes a human-readable Markdown report.
func (r Report) WriteMarkdown(w io.Writer) error {
	var b strings.Builder
	fmt.Fprintf(&b, "# Trajectory Evaluation Report\n\n")
	fmt.Fprintf(&b, "- Schema: `%s`\n", r.Schema)
	if r.Commit != "" {
		fmt.Fprintf(&b, "- Commit: `%s`\n", r.Commit)
	}
	fmt.Fprintf(&b, "- Overall: **%s**\n\n", statusWord(!r.Failed))

	// Summary table.
	if len(r.Summary) > 0 {
		b.WriteString("## Summary\n\n")
		b.WriteString("| Metric | Pass Rate | Mean Score | Passed/Total |\n")
		b.WriteString("|--------|-----------|------------|--------------|\n")
		for _, ms := range r.Summary {
			fmt.Fprintf(&b, "| %s | %.0f%% | %.2f | %d/%d |\n",
				ms.Metric, ms.PassRate*100, ms.MeanScore, ms.Passed, ms.Total)
		}
		b.WriteString("\n")
	}

	// Per-sample details.
	for _, sr := range r.Samples {
		fmt.Fprintf(&b, "## %s — %s\n\n", sr.Sample, statusWord(sr.Passed))
		b.WriteString("| Metric | Score | Passed | Reason |\n")
		b.WriteString("|--------|-------|--------|--------|\n")
		for _, res := range sr.Results {
			fmt.Fprintf(&b, "| %s | %.2f | %s | %s |\n",
				res.Metric, res.Score, checkCross(res.Passed), escPipe(res.Reason))
		}
		b.WriteString("\n")
		if sr.Attribution != nil {
			writeAttribution(&b, sr.Attribution)
		}
	}
	_, err := io.WriteString(w, b.String())
	return err
}

func writeAttribution(b *strings.Builder, a *attribution.Attribution) {
	b.WriteString("### Attribution\n\n")
	fmt.Fprintf(b, "- Root cause: **%s** (confidence %.2f)\n", a.RootCause.Category, a.RootCause.Confidence)
	if a.RootCause.Detail != "" {
		fmt.Fprintf(b, "- Detail: %s\n", a.RootCause.Detail)
	}
	if len(a.CausalChain) > 0 {
		fmt.Fprintf(b, "- Causal chain: steps %v\n", a.CausalChain)
	}
	if len(a.KeyFailures) > 0 {
		b.WriteString("\nKey failing steps:\n\n")
		for _, f := range a.KeyFailures {
			fmt.Fprintf(b, "- **Step %d** (%s): %s\n", f.StepIndex, f.StepKind, f.Summary)
			for _, ev := range f.Evidence {
				fmt.Fprintf(b, "  - `%s`: %s — %s\n", ev.Source, ev.Assertion, ev.Detail)
			}
		}
	}
	b.WriteString("\n")
}

func statusWord(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}

func checkCross(ok bool) string {
	if ok {
		return "✅"
	}
	return "❌"
}

func escPipe(s string) string {
	return strings.ReplaceAll(s, "|", "\\|")
}
