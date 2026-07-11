package report

import (
	"testing"
)

func TestParseGateConfigJSON(t *testing.T) {
	g, err := ParseGateConfig([]byte(`{"metrics": ["tool_correctness"], "min_pass_rate": 0.9, "min_scores": {"tool_correctness": 0.8}}`))
	if err != nil {
		t.Fatal(err)
	}
	if g.MinPassRate != 0.9 {
		t.Errorf("min_pass_rate: %.2f", g.MinPassRate)
	}
	if len(g.Metrics) != 1 || g.Metrics[0] != "tool_correctness" {
		t.Errorf("metrics: %v", g.Metrics)
	}
	if g.MinScores["tool_correctness"] != 0.8 {
		t.Errorf("min_scores: %v", g.MinScores)
	}
}

func TestParseGateConfigYAML(t *testing.T) {
	yaml := `# gate config
min_pass_rate: 0.9
metrics:
  - tool_correctness
  - agent_loop_detection
min_scores:
  tool_correctness: 0.8
  agent_loop_detection: 0.5
`
	g, err := ParseGateConfig([]byte(yaml))
	if err != nil {
		t.Fatal(err)
	}
	if g.MinPassRate != 0.9 {
		t.Errorf("min_pass_rate: %.2f", g.MinPassRate)
	}
	if len(g.Metrics) != 2 {
		t.Errorf("metrics: %v", g.Metrics)
	}
	if g.MinScores["tool_correctness"] != 0.8 || g.MinScores["agent_loop_detection"] != 0.5 {
		t.Errorf("min_scores: %v", g.MinScores)
	}
}

func TestParseGateConfigYAMLSimple(t *testing.T) {
	g, err := ParseGateConfig([]byte("min_pass_rate: 1.0\n"))
	if err != nil {
		t.Fatal(err)
	}
	if g.MinPassRate != 1.0 {
		t.Errorf("min_pass_rate: %.2f", g.MinPassRate)
	}
}

func TestParseGateConfigEmpty(t *testing.T) {
	_, err := ParseGateConfig([]byte(""))
	if err == nil {
		t.Fatal("expected error for empty config")
	}
}
