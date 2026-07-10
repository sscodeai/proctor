package agent

import (
	"context"
	"fmt"

	"github.com/hermes/trajectory-eval/metrics/judge"
	"github.com/hermes/trajectory-eval/trajectory"
)

// TaskCompletion asks the judge whether the agent completed the input task.
// Score is a float 0..1 (DeepEval semantics); passes above the threshold.
type TaskCompletion struct {
	Judge         judge.JudgeFunc
	PassThreshold float64
}

func (m TaskCompletion) Name() string { return "task_completion" }

func (m TaskCompletion) Evaluate(ctx context.Context, s trajectory.Sample) (trajectory.Result, error) {
	if s.Input == "" {
		return trajectory.Result{Metric: "task_completion", Passed: true, Score: 1, Reason: "no task (skipped)"}, nil
	}
	prompt := fmt.Sprintf(
		"Evaluate whether the agent completed the user's task.\n\nTask: %s\n\nTrajectory:\n%s\n\n"+
			"Return JSON: {\"passed\": true|false, \"score\": <0..1>, \"reason\": \"<short>\"}",
		s.Input, formatStepsForPrompt(s.Steps))
	v, err := judge.CallVerdict(ctx, m.Judge, prompt)
	if err != nil {
		return trajectory.Result{}, err
	}
	return trajectory.Result{
		Metric: "task_completion",
		Passed: v.Passed && v.Score >= m.PassThreshold,
		Score:  v.Score,
		Reason: v.Reason,
	}, nil
}

// StepEfficiency asks the judge whether the agent used a minimal number of
// steps / no redundant work. Includes a deterministic redundancy pre-check.
type StepEfficiency struct {
	Judge         judge.JudgeFunc
	PassThreshold float64
}

func (m StepEfficiency) Name() string { return "step_efficiency" }

func (m StepEfficiency) Evaluate(ctx context.Context, s trajectory.Sample) (trajectory.Result, error) {
	// Deterministic pre-check: flag obviously redundant repeated tool calls.
	redundant := countRepeatedToolCalls(s)
	prompt := fmt.Sprintf(
		"Evaluate the step efficiency of this agent trajectory: did it use a minimal number of steps, or was there redundant/duplicate work?\n\nTask: %s\n\nTrajectory:\n%s\n\n"+
			"Return JSON: {\"passed\": true|false, \"score\": <0..1>, \"reason\": \"<short>\"}",
		s.Input, formatStepsForPrompt(s.Steps))
	v, err := judge.CallVerdict(ctx, m.Judge, prompt)
	if err != nil {
		return trajectory.Result{}, err
	}
	// If deterministic pre-check found redundancy, don't give a free pass.
	if redundant > 0 && v.Score > 0.5 {
		v.Score = 0.4
		v.Passed = false
		v.Reason = fmt.Sprintf("%d repeated tool calls (deterministic); %s", redundant, v.Reason)
	}
	return trajectory.Result{
		Metric: "step_efficiency",
		Passed: v.Passed && v.Score >= m.PassThreshold,
		Score:  v.Score,
		Reason: v.Reason,
	}, nil
}

// countRepeatedToolCalls counts tool calls with identical (name, canonical args)
// appearing more than once.
func countRepeatedToolCalls(s trajectory.Sample) int {
	seen := map[string]bool{}
	dup := map[string]bool{}
	for _, c := range s.ToolCallsFromSteps() {
		key := c.Name + "(" + canonicalArgs(c.Args) + ")"
		if seen[key] {
			dup[key] = true
		}
		seen[key] = true
	}
	return len(dup)
}

func formatStepsForPrompt(steps []trajectory.Step) string {
	out := ""
	for _, st := range steps {
		line := fmt.Sprintf("%d. [%s] %s", st.Index, st.Kind, st.Text)
		if st.ToolCall != nil {
			line += fmt.Sprintf(" -> %s", st.ToolCall.Name)
		}
		out += line + "\n"
	}
	return out
}
