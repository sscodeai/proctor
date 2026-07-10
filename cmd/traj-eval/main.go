// Command traj-eval evaluates agent trajectories: load a dataset, run agent
// metrics (deterministic + optional LLM-as-Judge), attribute failures, and
// write JSON/Markdown reports. Also provides V1-vs-V2 diff and CI gate.
//
// Usage:
//
//	traj-eval eval --dataset examples/golden.json [--format json|markdown] [--out f] [--commit v1] [--judge]
//	traj-eval diff --base v1.json --current v2.json [--out f]
//	traj-eval gate --current v2.json [--config gate.yaml]
//
// Judge metrics require LLM_BASE_URL / LLM_API_KEY / LLM_MODEL env vars
// (any OpenAI-compatible endpoint; deepseek works). Without them, only
// deterministic metrics run.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/hermes/trajectory-eval/attribution"
	"github.com/hermes/trajectory-eval/ingest"
	"github.com/hermes/trajectory-eval/llmjudge"
	"github.com/hermes/trajectory-eval/metrics"
	"github.com/hermes/trajectory-eval/metrics/agent"
	"github.com/hermes/trajectory-eval/metrics/judge"
	"github.com/hermes/trajectory-eval/report"
	"github.com/hermes/trajectory-eval/trajectory"
	"github.com/hermes/trajectory-eval/web"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "traj-eval:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: traj-eval <eval|diff|gate> [flags]")
	}
	cmd, rest := args[0], args[1:]
	switch cmd {
	case "eval":
		return runEval(rest)
	case "diff":
		return runDiff(rest)
	case "gate":
		return runGate(rest)
	case "serve":
		return runServe(rest)
	default:
		return fmt.Errorf("unknown command %q (eval|diff|gate|serve)", cmd)
	}
}

// runServe starts the visualization UI server for a report file.
func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	reportPath := fs.String("report", "", "path to report JSON to visualize")
	addr := fs.String("addr", "127.0.0.1:8787", "listen address")
	basePath := fs.String("base", "/", "base path prefix")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *reportPath == "" {
		return fmt.Errorf("--report is required")
	}
	srv := &web.Server{ReportPath: *reportPath, BasePath: *basePath}
	fmt.Fprintf(os.Stderr, "traj-eval: visualization UI at http://%s%s\n", *addr, *basePath)
	return http.ListenAndServe(*addr, srv.Handler())
}

// buildJudge constructs a JudgeFunc from env if LLM creds exist, else nil.
// When nil, only deterministic metrics run. When --judge is given but creds
// are missing, it's an error (explicit request).
func buildJudge(wantJudge bool) (judge.JudgeFunc, error) {
	if !wantJudge {
		return nil, nil
	}
	client, err := llmjudge.FromEnv()
	if err != nil {
		return nil, fmt.Errorf("--judge requested but LLM not configured: %w (set LLM_BASE_URL/LLM_API_KEY/LLM_MODEL)", err)
	}
	return client.Judge(), nil
}

func runEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	dataset := fs.String("dataset", "", "path to dataset file (.json or .jsonl)")
	out := fs.String("out", "", "output file path (default stdout)")
	format := fs.String("format", "json", "report format: json | markdown")
	commit := fs.String("commit", "", "version label (V1/V2) recorded in the report")
	wantJudge := fs.Bool("judge", false, "enable LLM-as-Judge metrics (requires LLM_* env)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dataset == "" {
		return fmt.Errorf("--dataset is required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	samples, err := ingest.LoadSamples(*dataset)
	if err != nil {
		return fmt.Errorf("load dataset: %w", err)
	}
	if len(samples) == 0 {
		return fmt.Errorf("dataset %q contains no samples", *dataset)
	}

	j, err := buildJudge(*wantJudge)
	if err != nil {
		return err
	}

	metricSet := buildMetricSet(j)

	analyzer := attribution.Analyzer{
		Judge:      j,
		Thresholds: attribution.DefaultThresholds(),
	}

	perSample := map[string][]trajectory.Result{}
	attribs := map[string]*attribution.Attribution{}
	stepsMap := map[string][]trajectory.Step{}
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
		stepsMap[s.Name] = s.Steps
		a, err := analyzer.Analyze(ctx, s, results)
		if err != nil {
			return fmt.Errorf("attribute %q: %w", s.Name, err)
		}
		if len(a.KeyFailures) > 0 || a.RootCause.Category != attribution.CauseUnknown {
			attribs[s.Name] = &a
		}
	}

	r := report.Build(*commit, perSample, attribs, stepsMap)

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

func runDiff(args []string) error {
	fs := flag.NewFlagSet("diff", flag.ExitOnError)
	base := fs.String("base", "", "base report JSON (V1)")
	current := fs.String("current", "", "current report JSON (V2)")
	out := fs.String("out", "", "output file (default stdout)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *base == "" || *current == "" {
		return fmt.Errorf("--base and --current are required")
	}
	v1, err := report.LoadReport(*base)
	if err != nil {
		return fmt.Errorf("load base: %w", err)
	}
	v2, err := report.LoadReport(*current)
	if err != nil {
		return fmt.Errorf("load current: %w", err)
	}
	items := report.Diff(v1, v2)

	var w *os.File = os.Stdout
	if *out != "" && *out != "-" {
		f, err := os.Create(*out)
		if err != nil {
			return err
		}
		defer f.Close()
		w = f
	}

	fmt.Fprintf(w, "%-24s %-6s -> %-6s %-10s %s\n", "Sample", "V1", "V2", "Change", "RC Changed")
	fmt.Fprintln(w, "----------------------------------------")
	for _, it := range items {
		fmt.Fprintf(w, "%-24s %-6s -> %-6s %-10s %v\n", it.Sample, it.From, it.To, it.Change, it.RootCauseChanged)
	}

	// Summary counts.
	regressed, fixed := 0, 0
	for _, it := range items {
		if it.Change == "regressed" {
			regressed++
		}
		if it.Change == "fixed" {
			fixed++
		}
	}
	fmt.Fprintf(w, "\n%d regressed, %d fixed\n", regressed, fixed)
	if regressed > 0 {
		return fmt.Errorf("diff: %d regressed sample(s)", regressed)
	}
	return nil
}

func runGate(args []string) error {
	fs := flag.NewFlagSet("gate", flag.ExitOnError)
	current := fs.String("current", "", "current report JSON")
	config := fs.String("config", "", "gate config YAML/JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *current == "" {
		return fmt.Errorf("--current is required")
	}
	r, err := report.LoadReport(*current)
	if err != nil {
		return fmt.Errorf("load report: %w", err)
	}

	var g report.Gate
	if *config != "" {
		data, err := os.ReadFile(*config)
		if err != nil {
			return err
		}
		// Accept JSON config for now (YAML later).
		if err := jsonUnmarshal(data, &g); err != nil {
			return fmt.Errorf("parse gate config: %w", err)
		}
	} else {
		g = report.Gate{MinPassRate: 1.0}
	}

	passed, failures := g.Evaluate(r)
	if !passed {
		for _, f := range failures {
			fmt.Fprintln(os.Stderr, "gate:", f)
		}
		return fmt.Errorf("gate FAILED")
	}
	fmt.Fprintln(os.Stderr, "gate: PASS")
	return nil
}

func buildMetricSet(j judge.JudgeFunc) []metrics.Metric {
	ms := []metrics.Metric{
		metrics.Named("tool_correctness", agent.ToolCorrectness{}),
		metrics.Named("agent_loop_detection", agent.NewAgentLoopDetection()),
	}
	if j != nil {
		ms = append(ms,
			metrics.Named("argument_correctness", agent.ArgumentCorrectness{Judge: j, PassThreshold: 0.7}),
			metrics.Named("task_completion", agent.TaskCompletion{Judge: j, PassThreshold: 0.7}),
			metrics.Named("step_efficiency", agent.StepEfficiency{Judge: j, PassThreshold: 0.7}),
			metrics.Named("plan_quality", agent.PlanQuality{Judge: j, PassThreshold: 0.7}),
			metrics.Named("plan_adherence", agent.PlanAdherence{Judge: j, PassThreshold: 0.7}),
			metrics.Named("tool_use", agent.ToolUse{Judge: j, PassThreshold: 0.7}),
		)
	}
	return ms
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

// jsonUnmarshal is a tiny alias to keep gate config parsing simple.
func jsonUnmarshal(data []byte, v any) error {
	return json.Unmarshal(data, v)
}
