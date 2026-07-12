// Command proctor evaluates agent trajectories: load a dataset, run agent
// metrics (deterministic + optional LLM-as-Judge), attribute failures, and
// write JSON/Markdown reports. Also provides V1-vs-V2 diff and CI gate.
//
// Usage:
//
//	proctor eval --dataset examples/golden.json [--format json|markdown] [--out f] [--commit v1] [--judge]
//	proctor diff --base v1.json --current v2.json [--out f]
//	proctor gate --current v2.json [--config gate.yaml]
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
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/sscodeai/proctor/align"
	"github.com/sscodeai/proctor/attribution"
	"github.com/sscodeai/proctor/compare"
	"github.com/sscodeai/proctor/ingest"
	"github.com/sscodeai/proctor/llmjudge"
	"github.com/sscodeai/proctor/metrics"
	"github.com/sscodeai/proctor/metrics/agent"
	"github.com/sscodeai/proctor/metrics/judge"
	"github.com/sscodeai/proctor/report"
	"github.com/sscodeai/proctor/trajectory"
	"github.com/sscodeai/proctor/web"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "proctor:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: proctor <eval|diff|gate> [flags]")
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
	case "align":
		return runAlign(rest)
	case "compare":
		return runCompare(rest)
	default:
		return fmt.Errorf("unknown command %q (eval|diff|gate|serve|align|compare)", cmd)
	}
}

// runAlign measures judge-vs-human agreement: dataset (with human labels) +
// report (judge scores) -> alignment report.
func runAlign(args []string) error {
	fs := flag.NewFlagSet("align", flag.ExitOnError)
	dataset := fs.String("dataset", "", "dataset file with human labels (samples[].labels)")
	reportPath := fs.String("report", "", "report JSON produced by eval --judge")
	threshold := fs.Float64("threshold", 0.5, "judge pass threshold for binarization")
	out := fs.String("out", "", "output file (default stdout)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dataset == "" || *reportPath == "" {
		return fmt.Errorf("--dataset and --report are required")
	}

	samples, err := ingest.LoadSamples(*dataset)
	if err != nil {
		return fmt.Errorf("load dataset: %w", err)
	}
	rep, err := report.LoadReport(*reportPath)
	if err != nil {
		return fmt.Errorf("load report: %w", err)
	}

	// Pair human labels with judge scores by (sample, metric).
	judgeScores := map[string]map[string]float64{} // sample -> metric -> score
	for _, s := range rep.Samples {
		m := map[string]float64{}
		for _, r := range s.Results {
			m[r.Metric] = r.Score
		}
		judgeScores[s.Sample] = m
	}

	var pairs []align.Pair
	for _, s := range samples {
		name := s.Name
		if name == "" {
			continue
		}
		js, ok := judgeScores[name]
		if !ok {
			continue
		}
		for metric, human := range s.Labels {
			judge, ok := js[metric]
			if !ok {
				continue
			}
			pairs = append(pairs, align.Pair{
				Metric: metric, Sample: name,
				HumanScore: human, JudgeScore: judge,
			})
		}
	}
	if len(pairs) == 0 {
		return fmt.Errorf("no matching (sample, metric) pairs between dataset labels and report")
	}

	a := align.Align(pairs, *threshold)

	var w *os.File = os.Stdout
	if *out != "" && *out != "-" {
		f, err := os.Create(*out)
		if err != nil {
			return err
		}
		defer f.Close()
		w = f
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(a); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "align: %d pairs across %d metrics\n", len(pairs), len(a.Metrics))
	return nil
}

// runCompare builds a model × dataset matrix from multiple reports.
// Usage: proctor compare --report model=a,dataset=d:path.json ...
func runCompare(args []string) error {
	fs := flag.NewFlagSet("compare", flag.ExitOnError)
	var reportFlags multiFlag
	fs.Var(&reportFlags, "report", "report entry as model=X,dataset=Y:path (repeatable)")
	out := fs.String("out", "", "output file (default stdout)")
	format := fs.String("format", "markdown", "output format: markdown | json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if len(reportFlags) == 0 {
		return fmt.Errorf("at least one --report is required (model=X,dataset=Y:path)")
	}

	var entries []compare.Entry
	for _, rf := range reportFlags {
		model, ds, path, err := parseReportFlag(rf)
		if err != nil {
			return err
		}
		rep, err := report.LoadReport(path)
		if err != nil {
			return fmt.Errorf("load %s: %w", path, err)
		}
		entries = append(entries, compare.Entry{Model: model, Dataset: ds, Report: rep})
	}

	matrix := compare.Build(entries)

	var w *os.File = os.Stdout
	if *out != "" && *out != "-" {
		f, err := os.Create(*out)
		if err != nil {
			return err
		}
		defer f.Close()
		w = f
	}

	switch *format {
	case "markdown", "md":
		_, err := io.WriteString(w, matrix.Markdown())
		if err != nil {
			return err
		}
		best, rate, mean := matrix.Best()
		fmt.Fprintf(w, "\n**Best model: %s** (pass rate %.0f%%, mean score %.2f)\n", best, rate*100, mean)
	case "json":
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		if err := enc.Encode(matrix); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown format %q (markdown|json)", *format)
	}
	return nil
}

// multiFlag collects repeated --report flags.
type multiFlag []string

func (m *multiFlag) String() string { return fmt.Sprint([]string(*m)) }
func (m *multiFlag) Set(v string) error {
	*m = append(*m, v)
	return nil
}

// parseReportFlag parses "model=X,dataset=Y:path".
func parseReportFlag(s string) (model, ds, path string, err error) {
	idx := strings.LastIndex(s, ":")
	if idx < 0 {
		return "", "", "", fmt.Errorf("invalid --report %q (want model=X,dataset=Y:path)", s)
	}
	meta, path := s[:idx], s[idx+1:]
	for _, kv := range strings.Split(meta, ",") {
		parts := strings.SplitN(kv, "=", 2)
		if len(parts) != 2 {
			return "", "", "", fmt.Errorf("invalid meta %q in --report %q", kv, s)
		}
		switch strings.TrimSpace(parts[0]) {
		case "model":
			model = strings.TrimSpace(parts[1])
		case "dataset":
			ds = strings.TrimSpace(parts[1])
		}
	}
	if model == "" || ds == "" {
		return "", "", "", fmt.Errorf("--report %q needs model and dataset labels", s)
	}
	return model, ds, path, nil
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
	fmt.Fprintf(os.Stderr, "proctor: visualization UI at http://%s%s\n", *addr, *basePath)
	return http.ListenAndServe(*addr, srv.Handler())
}

// buildJudge constructs a JudgeFunc from env if LLM creds exist, else nil.
// When nil, only deterministic metrics run. When --judge is given but creds
// are missing, it's an error (explicit request).
//
// The returned func is wrapped with on-disk Cache (reproducibility), optional
// RateLimit (avoid 429s), and a Meter (cost tracking) when enabled.
func buildJudge(wantJudge bool, cacheDir string, rps float64, meter *judge.Meter) (judge.JudgeFunc, error) {
	if !wantJudge {
		return nil, nil
	}
	client, err := llmjudge.FromEnv()
	if err != nil {
		return nil, fmt.Errorf("--judge requested but LLM not configured: %w (set LLM_BASE_URL/LLM_API_KEY/LLM_MODEL)", err)
	}
	j := client.Judge()

	// Meter first (innermost so it sees raw calls), then rate limit, then cache.
	if meter != nil {
		meter.UsageFn = client.LastUsage
		j = meter.Wrap(j)
	}
	if rps > 0 {
		j = judge.RateLimit(j, rps, 8)
	}
	if cacheDir != "" {
		c, err := judge.NewCache(cacheDir)
		if err != nil {
			return nil, fmt.Errorf("create judge cache: %w", err)
		}
		j = c.Wrap(j)
	}
	return j, nil
}

func runEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	dataset := fs.String("dataset", "", "path to dataset file (.json or .jsonl)")
	out := fs.String("out", "", "output file path (default stdout)")
	format := fs.String("format", "json", "report format: json | markdown")
	commit := fs.String("commit", "", "version label (V1/V2) recorded in the report")
	wantJudge := fs.Bool("judge", false, "enable LLM-as-Judge metrics (requires LLM_* env)")
	parallel := fs.Int("parallel", 4, "number of samples evaluated concurrently")
	cacheDir := fs.String("cache", "", "judge cache dir (reproducibility; default .cache/ when --judge)")
	rps := fs.Float64("rps", 0, "rate limit judge calls per second (0 = unlimited)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dataset == "" {
		return fmt.Errorf("--dataset is required")
	}
	if *parallel < 1 {
		return fmt.Errorf("--parallel must be >= 1")
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

	// Default cache dir when judge is on.
	cache := *cacheDir
	if cache == "" && *wantJudge {
		cache = ".cache"
	}
	meter := &judge.Meter{}
	j, err := buildJudge(*wantJudge, cache, *rps, meter)
	if err != nil {
		return err
	}

	metricSet := buildMetricSet(j)

	analyzer := attribution.Analyzer{
		Judge:      j,
		Thresholds: attribution.DefaultThresholds(),
	}

	// Concurrent evaluation with a worker pool.
	type sampleJob struct {
		idx int
		s   trajectory.Sample
	}
	type sampleResult struct {
		idx     int
		results []trajectory.Result
		attr    attribution.Attribution
		err     error
	}

	jobs := make(chan sampleJob)
	resultsCh := make(chan sampleResult, len(samples))

	// Workers.
	var wg sync.WaitGroup
	for w := 0; w < *parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				s := job.s
				results, err := metrics.RunAll(ctx, s, metricSet)
				if err != nil {
					resultsCh <- sampleResult{idx: job.idx, err: fmt.Errorf("evaluate %q: %w", s.Name, err)}
					continue
				}
				a, err := analyzer.Analyze(ctx, s, results)
				if err != nil {
					resultsCh <- sampleResult{idx: job.idx, err: fmt.Errorf("attribute %q: %w", s.Name, err)}
					continue
				}
				resultsCh <- sampleResult{idx: job.idx, results: results, attr: a}
			}
		}()
	}

	// Feed jobs.
	go func() {
		for i := range samples {
			s := samples[i]
			if s.Name == "" {
				s.Name = fmt.Sprintf("sample_%d", i)
			}
			jobs <- sampleJob{idx: i, s: s}
		}
		close(jobs)
	}()

	// Collect.
	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	perSample := map[string][]trajectory.Result{}
	attribs := map[string]*attribution.Attribution{}
	stepsMap := map[string][]trajectory.Step{}
	for res := range resultsCh {
		if res.err != nil {
			return res.err
		}
		s := samples[res.idx]
		perSample[s.Name] = res.results
		stepsMap[s.Name] = s.Steps
		if len(res.attr.KeyFailures) > 0 || res.attr.RootCause.Category != attribution.CauseUnknown {
			a := res.attr
			attribs[s.Name] = &a
		}
	}

	r := report.Build(*commit, perSample, attribs, stepsMap)

	// Report usage.
	if calls, _, inTok, outTok, cost := meter.Snapshot(); calls > 0 {
		r.Usage = &report.Usage{
			LLMCalls:  calls,
			TotalCost: cost,
		}
		fmt.Fprintf(os.Stderr, "judge: %d LLM calls, %d in + %d out tokens, cost $%.4f\n",
			calls, inTok, outTok, cost)
	}

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
		g, err = report.ParseGateConfig(data)
		if err != nil {
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
