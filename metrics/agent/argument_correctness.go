package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/sscodeai/proctor/metrics/judge"
	"github.com/sscodeai/proctor/trajectory"
)

// ArgumentCorrectness asks the judge, for EACH recorded tool call, whether the
// arguments are correct and sufficient to accomplish the input task. Score =
// correct calls / total. Catches right-tool-wrong-arguments failures.
//
// The verdicts carry the step index so the attribution layer can locate
// argument errors precisely.
type ArgumentCorrectness struct {
	Judge          judge.JudgeFunc
	PassThreshold  float64
	IncludeContext bool
}

func (m ArgumentCorrectness) Name() string { return "argument_correctness" }

func (m ArgumentCorrectness) Evaluate(ctx context.Context, s trajectory.Sample) (trajectory.Result, error) {
	calls := s.ToolCallsFromSteps()
	if len(calls) == 0 {
		return trajectory.Result{
			Metric: "argument_correctness", Passed: true, Score: 1,
			Reason: "no tool calls (skipped)",
		}, nil
	}

	prompt := m.buildPrompt(s, calls)
	var resp struct {
		Verdicts []struct {
			StepIndex int    `json:"step_index"`
			Verdict   string `json:"verdict"` // yes | no
			Reason    string `json:"reason"`
		} `json:"verdicts"`
	}
	if err := judge.CallJSON(ctx, m.Judge, prompt, &resp); err != nil {
		return trajectory.Result{}, err
	}

	correct := 0
	reasons := []string{}
	firstFailStep := -1
	for _, v := range resp.Verdicts {
		if v.Verdict == "yes" || v.Verdict == "pass" {
			correct++
		} else {
			reasons = append(reasons, fmt.Sprintf("step %d: %s", v.StepIndex, v.Reason))
			if firstFailStep == -1 || (v.StepIndex >= 0 && v.StepIndex < firstFailStep) {
				firstFailStep = v.StepIndex
			}
		}
	}

	score := float64(correct) / float64(len(resp.Verdicts))
	if len(resp.Verdicts) == 0 {
		score = 0
	}
	passed := score >= m.PassThreshold
	reason := fmt.Sprintf("%d/%d tool calls had correct arguments", correct, len(resp.Verdicts))
	if len(reasons) > 0 {
		reason += "; " + joinReasons(reasons, 3)
	}

	return trajectory.Result{
		Metric:  "argument_correctness",
		Passed:  passed,
		Score:   score,
		Reason:  reason,
		StepIdx: firstFailStep,
		Located: firstFailStep >= 0,
	}, nil
}

func (m ArgumentCorrectness) buildPrompt(s trajectory.Sample, calls []trajectory.ToolCall) string {
	// Render the trajectory with step indices so the judge can reference them.
	var b []byte
	b = append(b, []byte("Evaluate whether each tool call's arguments are correct and sufficient to accomplish the task.\n\nTask: "+s.Input+"\n\nTrajectory:\n")...)
	for _, st := range s.Steps {
		line := fmt.Sprintf("%d. [%s] %s", st.Index, st.Kind, st.Text)
		if st.ToolCall != nil {
			args, _ := json.Marshal(st.ToolCall.Args)
			line += fmt.Sprintf(" -> %s(%s)", st.ToolCall.Name, args)
		}
		b = append(b, []byte(line+"\n")...)
	}
	b = append(b, []byte("\nFor each tool_call step, return JSON:\n{\"verdicts\": [{\"step_index\": <int>, \"verdict\": \"yes|no\", \"reason\": \"<short>\"}]}")...)
	return string(b)
}

func joinReasons(reasons []string, max int) string {
	if len(reasons) <= max {
		return join(reasons, "; ")
	}
	return join(reasons[:max], "; ") + fmt.Sprintf("; +%d more", len(reasons)-max)
}

func join(parts []string, sep string) string {
	out := ""
	for i, p := range parts {
		if i > 0 {
			out += sep
		}
		out += p
	}
	return out
}
