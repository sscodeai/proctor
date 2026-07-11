package align

import (
	"math"
	"testing"
)

func TestPerfectAgreement(t *testing.T) {
	// Judge and human agree on everything: kappa=1, acc=1, pearson=1.
	pairs := []Pair{
		{Metric: "m", HumanScore: 1, JudgeScore: 0.9},
		{Metric: "m", HumanScore: 1, JudgeScore: 0.95},
		{Metric: "m", HumanScore: 0, JudgeScore: 0.1},
		{Metric: "m", HumanScore: 0, JudgeScore: 0.2},
	}
	rep := Align(pairs, 0.5)
	ma := rep.Metrics[0]
	if ma.Kappa != 1.0 {
		t.Errorf("expected kappa 1.0, got %.3f", ma.Kappa)
	}
	if ma.Accuracy != 1.0 {
		t.Errorf("expected accuracy 1.0, got %.3f", ma.Accuracy)
	}
	if math.Abs(ma.Pearson-1.0) > 0.01 {
		t.Errorf("expected pearson ~1.0, got %.3f", ma.Pearson)
	}
	if ma.Bias > 0.1 {
		t.Errorf("expected near-zero bias, got %.3f", ma.Bias)
	}
}

func TestNoAgreement(t *testing.T) {
	// Judge always passes, human always fails -> kappa ~0, precision 0.
	pairs := []Pair{
		{Metric: "m", HumanScore: 0, JudgeScore: 0.9},
		{Metric: "m", HumanScore: 0, JudgeScore: 0.8},
		{Metric: "m", HumanScore: 0, JudgeScore: 0.95},
		{Metric: "m", HumanScore: 0, JudgeScore: 0.85},
	}
	rep := Align(pairs, 0.5)
	ma := rep.Metrics[0]
	if ma.Accuracy != 0.0 {
		t.Errorf("expected accuracy 0, got %.3f", ma.Accuracy)
	}
	if ma.Precision != 0 {
		t.Errorf("expected precision 0, got %.3f", ma.Precision)
	}
	if ma.Bias <= 0.7 {
		t.Errorf("expected large positive bias (judge too lenient), got %.3f", ma.Bias)
	}
}

func TestPartialAgreement(t *testing.T) {
	pairs := []Pair{
		{Metric: "m", HumanScore: 1, JudgeScore: 0.9}, // tp
		{Metric: "m", HumanScore: 1, JudgeScore: 0.2}, // fn
		{Metric: "m", HumanScore: 0, JudgeScore: 0.1}, // tn
		{Metric: "m", HumanScore: 0, JudgeScore: 0.8}, // fp
	}
	rep := Align(pairs, 0.5)
	ma := rep.Metrics[0]
	if ma.Accuracy != 0.5 {
		t.Errorf("expected accuracy 0.5, got %.3f", ma.Accuracy)
	}
	if ma.Precision != 0.5 || ma.Recall != 0.5 || ma.F1 != 0.5 {
		t.Errorf("expected 0.5 metrics, got prec=%.2f rec=%.2f f1=%.2f", ma.Precision, ma.Recall, ma.F1)
	}
	if ma.Kappa != 0.0 {
		t.Errorf("expected kappa 0.0 (chance-level), got %.3f", ma.Kappa)
	}
}

func TestBiasDirection(t *testing.T) {
	// Judge systematically higher -> positive bias.
	pairs := []Pair{
		{Metric: "m", HumanScore: 0.5, JudgeScore: 0.9},
		{Metric: "m", HumanScore: 0.5, JudgeScore: 0.8},
		{Metric: "m", HumanScore: 0.5, JudgeScore: 0.7},
	}
	rep := Align(pairs, 0.5)
	ma := rep.Metrics[0]
	if ma.Bias < 0.2 {
		t.Errorf("expected positive bias, got %.3f", ma.Bias)
	}
}

func TestMultipleMetrics(t *testing.T) {
	pairs := []Pair{
		{Metric: "a", HumanScore: 1, JudgeScore: 0.9},
		{Metric: "a", HumanScore: 0, JudgeScore: 0.1},
		{Metric: "b", HumanScore: 1, JudgeScore: 0.1}, // judge wrong
		{Metric: "b", HumanScore: 0, JudgeScore: 0.9}, // judge wrong
	}
	rep := Align(pairs, 0.5)
	if len(rep.Metrics) != 2 {
		t.Fatalf("expected 2 metrics, got %d", len(rep.Metrics))
	}
	if rep.Metrics[0].Accuracy != 1.0 {
		t.Errorf("metric a should be 1.0 accuracy, got %.2f", rep.Metrics[0].Accuracy)
	}
	if rep.Metrics[1].Accuracy != 0.0 {
		t.Errorf("metric b should be 0.0 accuracy, got %.2f", rep.Metrics[1].Accuracy)
	}
	// Overall pools all pairs: 2/4 correct.
	if rep.Overall.Accuracy != 0.5 {
		t.Errorf("overall accuracy should be 0.5, got %.2f", rep.Overall.Accuracy)
	}
}

func TestSmallSample(t *testing.T) {
	// n=1: pearson undefined (NaN), kappa undefined.
	pairs := []Pair{{Metric: "m", HumanScore: 1, JudgeScore: 0.9}}
	rep := Align(pairs, 0.5)
	ma := rep.Metrics[0]
	if !math.IsNaN(ma.Pearson) {
		t.Errorf("expected NaN pearson for n=1, got %v", ma.Pearson)
	}
	if ma.Accuracy != 1.0 {
		t.Errorf("expected accuracy 1.0, got %.2f", ma.Accuracy)
	}
}
