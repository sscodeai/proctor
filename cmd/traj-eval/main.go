// Command traj-eval evaluates agent trajectories: load a dataset, run the
// deterministic agent metrics, attribute failures, and write JSON/Markdown
// reports. Usage:
//
//	traj-eval eval --dataset examples/golden.json --out report.json
//	traj-eval eval --dataset examples/golden.json --out report.md --format markdown
//
// Deterministic-only in M0 (no LLM judge); judge metrics arrive in M1.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/hermes/trajectory-eval/attribution"
	"github.com/hermes/trajectory-eval/ingest"
	"github.com/hermes/trajectory-eval/metrics"
	"github.com/hermes/trajectory-eval/metrics/agent"
	"github.com/hermes/trajectory-eval/report"
	"github.com/hermes/trajectory-eval/trajectory"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "traj-eval:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	// Support both `traj-eval eval --dataset ...` and `traj-eval --dataset ...`.
	if len(args) > 0 && args[0] == "eval" {
		args = args[1:]
	}
	fs := flag.NewFlagSet("traj-eval", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage: traj-eval eval [flags]\n\nFlags:\n")
		fs.PrintDefaults()
	}
	dataset := fs.String("dataset", "", "path to dataset file (.json or .jsonl)")
	out := fs.String("out", "", "output file path (default stdout)")
	format := fs.String("format", "json", "report format: json | markdown")
	commit := fs.String("commit", "", "version label (V1/V2) recorded in the report")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dataset == "" {
		return fmt.Errorf("--dataset is required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	samples, err := ingest.LoadSamples(*dataset)
	if err != nil {
		return fmt.Errorf("load dataset: %w", err)
	}
	if len(samples) == 0 {
		return fmt.Errorf("dataset %q contains no samples", *dataset)
	}

	// M0 metric set: deterministic only.
	metricSet := []metrics.Metric{
		metrics.Named("tool_correctness", agent.ToolCorrectness{}),
		metrics.Named("agent_loop_detection", agent.NewAgentLoopDetection()),
	}

	analyzer := attribution.Analyzer{Thresholds: attribution.DefaultThresholds()}

	perSample := map[string][]trajectory.Result{}
	attribs := map[string]*attribution.Attribution{}
	for i := range samples {
		s := samples[i]
		if s.Name == "" {
			s.Name = fmt.Sprintf("sample_%d", i)
		}
		results, err := metrics.RunAll(ctx, s, metricSet)
		if err != nil {
			return fmt.Errorf("evaluate %q: %w", s.Name, err)
		}
		perSample[s.Name] = results
		a, err := analyzer.Analyze(ctx, s, results)
		if err != nil {
			return fmt.Errorf("attribute %q: %w", s.Name, err)
		}
		if len(a.KeyFailures) > 0 || a.RootCause.Category != attribution.CauseUnknown {
			attribs[s.Name] = &a
		}
	}

	r := report.Build(*commit, perSample, attribs)

	var w *os.File
	if *out != "" && *out != "-" {
		f, err := os.Create(*out)
		if err != nil {
			return fmt.Errorf("create output: %w", err)
		}
		defer f.Close()
		w = f
	} else {
		w = os.Stdout
	}

	switch *format {
	case "json":
		err = r.WriteJSON(w)
	case "markdown", "md":
		err = r.WriteMarkdown(w)
	default:
		return fmt.Errorf("unknown format %q (json|markdown)", *format)
	}
	if err != nil {
		return err
	}

	if r.Failed {
		return fmt.Errorf("evaluation FAILED: %d/%d samples passed",
			passedCount(r), len(r.Samples))
	}
	fmt.Fprintf(os.Stderr, "ok: %d samples, %d metrics\n", len(r.Samples), len(metricSet))
	return nil
}

func passedCount(r report.Report) int {
	n := 0
	for _, s := range r.Samples {
		if s.Passed {
			n++
		}
	}
	return n
}
