// Package align measures how well LLM-as-a-Judge verdicts agree with human
// gold labels — the "who validates the validators" problem (arXiv:2404.12272).
//
// Given a set of samples with human labels (per metric: pass/fail or a
// 0..1 score) and the corresponding judge scores from a report, it computes:
//
//   - Pearson correlation (continuous agreement)
//   - Cohen's kappa (binary agreement, chance-corrected)
//   - Accuracy / precision / recall / F1 (binary)
//   - Bias (judge systematically higher/lower than humans)
//
// This lets you decide whether an automated judge can be trusted, tune
// thresholds, and pick the judge configuration that best matches experts.
package align

import (
	"fmt"
	"math"
)

// Pair is one (human_label, judge_score) observation for a metric.
type Pair struct {
	Metric     string  // metric name, e.g. "tool_correctness"
	Sample     string  // sample name (for reporting)
	HumanScore float64 // human gold: 0..1 (binary 0/1 or continuous)
	JudgeScore float64 // judge score from the report (0..1)
}

// MetricAlign is the alignment result for one metric.
type MetricAlign struct {
	Metric        string  `json:"metric"`
	N             int     `json:"n"`
	Pearson       float64 `json:"pearson"`   // -1..1, NaN if n<2 or zero variance
	Kappa         float64 `json:"kappa"`     // Cohen's kappa, -1..1
	Accuracy      float64 `json:"accuracy"`  // binary agreement fraction
	Precision     float64 `json:"precision"` // judge-pass given human-pass
	Recall        float64 `json:"recall"`    // judge detects human-passes
	F1            float64 `json:"f1"`
	Bias          float64 `json:"bias"` // mean(judge - human), >0 judge too lenient
	JudgePassRate float64 `json:"judge_pass_rate"`
	HumanPassRate float64 `json:"human_pass_rate"`
	Threshold     float64 `json:"threshold"` // pass threshold used for binarization
}

// Report is the full alignment result across metrics.
type Report struct {
	Schema  string        `json:"schema"` // "proctor-align/v1"
	Metrics []MetricAlign `json:"metrics"`
	// Overall is the mean of per-metric metrics (n-weighted).
	Overall MetricAlign `json:"overall"`
}

// Align computes alignment between human labels and judge scores.
// threshold is the judge-score pass threshold (default 0.5 if <=0).
func Align(pairs []Pair, threshold float64) Report {
	if threshold <= 0 {
		threshold = 0.5
	}
	// Group by metric preserving first-seen order.
	order := []string{}
	byMetric := map[string][]Pair{}
	for _, p := range pairs {
		if _, ok := byMetric[p.Metric]; !ok {
			order = append(order, p.Metric)
		}
		byMetric[p.Metric] = append(byMetric[p.Metric], p)
	}

	rep := Report{Schema: "proctor-align/v1"}
	allPairs := []Pair{}
	for _, m := range order {
		ma := computeMetricAlign(byMetric[m], threshold)
		rep.Metrics = append(rep.Metrics, ma)
		allPairs = append(allPairs, byMetric[m]...)
	}
	// Overall across all pairs (all metrics pooled).
	if len(allPairs) > 0 {
		rep.Overall = computeMetricAlign(allPairs, threshold)
	}
	return rep
}

func computeMetricAlign(pairs []Pair, threshold float64) MetricAlign {
	ma := MetricAlign{Metric: pairs[0].Metric, N: len(pairs), Threshold: threshold}

	// Continuous agreement.
	pearson := math.NaN()
	if len(pairs) >= 2 {
		if p, ok := pearsonCorr(pairs); ok {
			pearson = p
		}
	}
	ma.Pearson = pearson

	// Binary agreement using threshold for judge, 0.5 for human labels
	// (human labels are treated as binary pass/fail).
	var tp, fp, tn, fn int // judge-pass=positive
	judgePass, humanPass := 0, 0
	sumDiff := 0.0
	for _, p := range pairs {
		jPass := p.JudgeScore >= threshold
		hPass := p.HumanScore >= 0.5
		sumDiff += p.JudgeScore - p.HumanScore
		if jPass {
			judgePass++
		}
		if hPass {
			humanPass++
		}
		switch {
		case jPass && hPass:
			tp++
		case jPass && !hPass:
			fp++
		case !jPass && hPass:
			fn++
		default:
			tn++
		}
	}

	n := len(pairs)
	ma.Accuracy = float64(tp+tn) / float64(n)
	ma.Precision = safeDiv(float64(tp), float64(tp+fp))
	ma.Recall = safeDiv(float64(tp), float64(tp+fn))
	ma.F1 = safeDiv(2*float64(tp), 2*float64(tp)+float64(fp)+float64(fn))
	ma.Bias = sumDiff / float64(n)
	ma.JudgePassRate = float64(judgePass) / float64(n)
	ma.HumanPassRate = float64(humanPass) / float64(n)

	// Cohen's kappa with chance correction.
	ma.Kappa = kappa(tp, fp, fn, tn)

	return ma
}

// pearsonCorr computes Pearson correlation over pairs; ok=false if variance is 0.
func pearsonCorr(pairs []Pair) (float64, bool) {
	n := float64(len(pairs))
	var sx, sy, sxx, syy, sxy float64
	for _, p := range pairs {
		sx += p.HumanScore
		sy += p.JudgeScore
		sxx += p.HumanScore * p.HumanScore
		syy += p.JudgeScore * p.JudgeScore
		sxy += p.HumanScore * p.JudgeScore
	}
	denom := math.Sqrt((n*sxx - sx*sx) * (n*syy - sy*sy))
	if denom == 0 {
		return 0, false
	}
	return (n*sxy - sx*sy) / denom, true
}

// kappa computes Cohen's kappa for a 2x2 confusion matrix (judge vs human).
func kappa(tp, fp, fn, tn int) float64 {
	total := float64(tp + fp + fn + tn)
	if total == 0 {
		return math.NaN()
	}
	po := float64(tp+tn) / total
	// Expected agreement by chance.
	judgePass := float64(tp + fp)
	humanPass := float64(tp + fn)
	pe := (judgePass/total)*(humanPass/total) + ((total-judgePass)/total)*((total-humanPass)/total)
	if pe == 1 {
		return math.NaN()
	}
	return (po - pe) / (1 - pe)
}

func safeDiv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

// Format is a human-readable rendering of a MetricAlign.
func (ma MetricAlign) Format() string {
	pearson := "n/a"
	if !math.IsNaN(ma.Pearson) {
		pearson = fmt.Sprintf("%.3f", ma.Pearson)
	}
	kappa := "n/a"
	if !math.IsNaN(ma.Kappa) {
		kappa = fmt.Sprintf("%.3f", ma.Kappa)
	}
	return fmt.Sprintf(
		"  %-24s n=%-3d pearson=%-7s kappa=%-7s acc=%.2f prec=%.2f recall=%.2f f1=%.2f bias=%+.3f",
		ma.Metric, ma.N, pearson, kappa, ma.Accuracy, ma.Precision, ma.Recall, ma.F1, ma.Bias)
}
