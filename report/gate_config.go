package report

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// ParseGateConfig parses a gate config from either JSON or a minimal YAML
// subset. The YAML subset covers the Gate shape:
//
//	min_pass_rate: 0.9
//	metrics:
//	  - tool_correctness
//	  - agent_loop_detection
//	min_scores:
//	  tool_correctness: 0.8
//
// Indentation-based nesting is flattened to one level (the Gate shape needs
// at most two levels). This keeps the core stdlib-only.
func ParseGateConfig(data []byte) (Gate, error) {
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "" {
		return Gate{}, fmt.Errorf("empty config")
	}
	// JSON first (it's also valid YAML, but our subset parser would mangle it).
	if strings.HasPrefix(trimmed, "{") {
		var g Gate
		if err := jsonUnmarshalGate([]byte(trimmed), &g); err != nil {
			return Gate{}, err
		}
		return g, nil
	}
	return parseYAMLGate(trimmed)
}

// jsonUnmarshalGate parses JSON into a Gate.
func jsonUnmarshalGate(data []byte, g *Gate) error {
	return json.Unmarshal(data, g)
}

// parseYAMLGate parses the minimal YAML subset into a Gate.
func parseYAMLGate(s string) (Gate, error) {
	var g Gate
	lines := strings.Split(s, "\n")

	// First pass: find list items under "metrics:".
	metricsMode := false
	for _, raw := range lines {
		line := strings.TrimRight(raw, " \t")
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		indent := len(line) - len(strings.TrimLeft(line, " "))
		if indent == 0 {
			metricsMode = false
		}
		if metricsMode && strings.HasPrefix(trimmed, "- ") {
			g.Metrics = append(g.Metrics, strings.TrimSpace(strings.TrimPrefix(trimmed, "- ")))
			continue
		}
		if strings.HasSuffix(trimmed, ":") {
			key := strings.TrimSuffix(trimmed, ":")
			if key == "metrics" {
				metricsMode = true
			}
			continue
		}
	}

	// Second pass: scalar keys and nested min_scores.
	minScoresMode := false
	for _, raw := range lines {
		line := strings.TrimRight(raw, " \t")
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") || strings.HasPrefix(trimmed, "- ") {
			continue
		}
		idx := strings.Index(trimmed, ":")
		if idx < 0 {
			continue
		}
		key := strings.TrimSpace(trimmed[:idx])
		val := strings.TrimSpace(trimmed[idx+1:])

		if key == "min_scores" {
			minScoresMode = true
			continue
		}
		if minScoresMode && key != "" && val != "" {
			if g.MinScores == nil {
				g.MinScores = map[string]float64{}
			}
			f, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return Gate{}, fmt.Errorf("min_scores.%s: %v", key, err)
			}
			g.MinScores[key] = f
			continue
		}
		minScoresMode = false

		switch key {
		case "min_pass_rate":
			f, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return Gate{}, fmt.Errorf("min_pass_rate: %v", err)
			}
			g.MinPassRate = f
		}
	}
	return g, nil
}
