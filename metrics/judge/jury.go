package judge

import (
	"context"
	"fmt"
	"sort"
	"sync"
)

// Jury aggregates verdicts from multiple judges (different models) for the
// same prompt, mitigating single-model bias. This implements the
// "Replacing Judges with Juries" idea (arXiv:2404.18796): a panel of diverse
// models is more robust than any single judge.
type Jury struct {
	Judges   []JudgeFunc
	Parallel bool // run judges concurrently (default true)
}

// NewJury builds a Jury from the given judges.
func NewJury(judges ...JudgeFunc) *Jury {
	return &Jury{Judges: judges, Parallel: true}
}

// JudgeFunc returns a JudgeFunc that aggregates the panel for a prompt.
func (j *Jury) JudgeFunc() JudgeFunc {
	return func(ctx context.Context, prompt string) (string, error) {
		return j.Ask(ctx, prompt)
	}
}

// Ask runs the panel and returns an aggregated JSON verdict string.
// Aggregation: majority vote on "passed", mean of scores, concatenated
// reasons (with per-judge attribution).
func (j *Jury) Ask(ctx context.Context, prompt string) (string, error) {
	if len(j.Judges) == 0 {
		return "", fmt.Errorf("jury has no judges")
	}
	if len(j.Judges) == 1 {
		return j.Judges[0](ctx, prompt)
	}

	results := make([]Verdict, len(j.Judges))
	errs := make([]error, len(j.Judges))

	run := func(i int) {
		results[i], errs[i] = CallVerdict(ctx, j.Judges[i], prompt)
	}

	if j.Parallel {
		var wg sync.WaitGroup
		wg.Add(len(j.Judges))
		for i := range j.Judges {
			go func(i int) {
				defer wg.Done()
				run(i)
			}(i)
		}
		wg.Wait()
	} else {
		for i := range j.Judges {
			run(i)
		}
	}

	// If any judge errored, report it but continue with the others.
	valid := 0
	passVotes := 0
	scoreSum := 0.0
	reasons := make([]string, 0, len(j.Judges))
	for i, v := range results {
		if errs[i] != nil {
			reasons = append(reasons, fmt.Sprintf("judge[%d]: error %v", i, errs[i]))
			continue
		}
		valid++
		if v.Passed {
			passVotes++
		}
		scoreSum += v.Score
		reasons = append(reasons, fmt.Sprintf("judge[%d]: %.2f %s", i, v.Score, v.Reason))
	}
	if valid == 0 {
		return "", fmt.Errorf("all judges failed: %v", errs[0])
	}

	passed := passVotes > valid/2 // strict majority
	score := scoreSum / float64(valid)

	// Compact reasons.
	reason := ""
	for i, r := range reasons {
		if i > 0 {
			reason += " | "
		}
		reason += r
	}
	if len(reason) > 400 {
		reason = reason[:400] + "..."
	}

	return fmt.Sprintf(`{"passed": %t, "score": %.4f, "reason": %q}`, passed, score, reason), nil
}

// Verdicts returns the individual (unaggregated) verdicts for inspection.
func (j *Jury) Verdicts(ctx context.Context, prompt string) ([]Verdict, error) {
	out := make([]Verdict, 0, len(j.Judges))
	for i := range j.Judges {
		v, err := CallVerdict(ctx, j.Judges[i], prompt)
		if err != nil {
			return out, err
		}
		out = append(out, v)
	}
	return out, nil
}

// SortedVerdicts returns individual verdicts sorted by score descending.
func SortedVerdicts(vs []Verdict) []Verdict {
	out := make([]Verdict, len(vs))
	copy(out, vs)
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out
}
