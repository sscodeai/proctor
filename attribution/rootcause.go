package attribution

import (
	"encoding/json"
	"fmt"
	"strings"
)

// classify maps located failures to a root cause category via deterministic
// rules. Falls back to CauseUnknown when no rule matches.
func classify(failures []FailureStep) RootCause {
	if len(failures) == 0 {
		return RootCause{Category: CauseUnknown, Detail: "no failure located", Confidence: 0}
	}

	// Gather the set of violated constraints and the earliest step.
	constraints := map[string]bool{}
	firstIdx := failures[0].StepIndex
	for _, f := range failures {
		constraints[f.Constraint] = true
		if f.StepIndex < firstIdx {
			firstIdx = f.StepIndex
		}
	}

	// Rule mapping (deterministic, confidence 1.0 when unambiguous).
	switch {
	case constraints["expected tool set must be called exactly"] && len(constraints) == 1:
		detail := "expected tool set mismatch"
		for _, f := range failures {
			if strings.Contains(f.Summary, "missing:") {
				detail = "missing expected tool call"
				break
			}
		}
		return RootCause{Category: CauseWrongTool, Detail: detail, Confidence: 1.0}
	case constraints["tool arguments must be correct and sufficient"]:
		return RootCause{Category: CauseBadArguments, Detail: "tool argument error", Confidence: 1.0}
	case constraints["execution must follow the plan"]:
		return RootCause{Category: CausePlanDeviation, Detail: "execution deviated from plan", Confidence: 1.0}
	case constraints["no loops / repeated work"]:
		return RootCause{Category: CauseRedundantLoop, Detail: "loop or repeated work detected", Confidence: 1.0}
	}

	// Multiple constraints: pick the earliest failing constraint as primary.
	// A judge could arbitrate here (later milestone); for now deterministic
	// priority: tool set > arguments > plan > loop.
	priority := []string{
		"expected tool set must be called exactly",
		"tool arguments must be correct and sufficient",
		"execution must follow the plan",
		"no loops / repeated work",
	}
	for _, c := range priority {
		if constraints[c] {
			switch c {
			case "expected tool set must be called exactly":
				return RootCause{Category: CauseWrongTool, Detail: "tool set mismatch (multi-constraint)", Confidence: 0.9}
			case "tool arguments must be correct and sufficient":
				return RootCause{Category: CauseBadArguments, Detail: "argument error (multi-constraint)", Confidence: 0.9}
			case "execution must follow the plan":
				return RootCause{Category: CausePlanDeviation, Detail: "plan deviation (multi-constraint)", Confidence: 0.9}
			case "no loops / repeated work":
				return RootCause{Category: CauseRedundantLoop, Detail: "loop (multi-constraint)", Confidence: 0.9}
			}
		}
	}

	return RootCause{Category: CauseUnknown, Detail: fmt.Sprintf("failed at step %d", firstIdx), Confidence: 0.5}
}

// parseJudgeJSON parses the judge fallback JSON response.
// Expected shape: {"step_index": int, "summary": string, "confidence": float}
func parseJudgeJSON(s string) (int, string, float64, error) {
	// Strip code fences if the model wrapped the JSON.
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		lines := strings.Split(s, "\n")
		if len(lines) > 2 {
			s = strings.Join(lines[1:len(lines)-1], "\n")
		}
	}
	var out struct {
		StepIndex  int     `json:"step_index"`
		Summary    string  `json:"summary"`
		Confidence float64 `json:"confidence"`
	}
	if err := json.Unmarshal([]byte(s), &out); err != nil {
		return 0, "", 0, fmt.Errorf("parse judge JSON: %w", err)
	}
	return out.StepIndex, out.Summary, out.Confidence, nil
}
