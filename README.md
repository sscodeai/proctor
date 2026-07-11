# trajectory-eval

**Agent trajectory evaluation CLI — metrics + causal failure attribution, in pure Go.**

`trajectory-eval` evaluates recorded agent trajectories. It scores tool-call correctness, detects loops, checks plan adherence — and unlike most eval tools, **locates the exact step where a trajectory went wrong and explains why**, with an evidence chain and a root-cause classification.

> *Demo "Demos show capability, Eval builds trust" — but only if you can say *where* it failed.

![demo](docs/demo.gif)

## Why this exists

| Tool | Language | What it does | Gap |
|---|---|---|---|
| **DeepEval** (17.9k★) | Python | Scores agent trajectories | ❌ Scores failures but doesn't *locate* them |
| **AgentRx / TrajDebug** | Python | Causal attribution research | ❌ Research-grade, not a product |
| **Go ecosystem** | Go | — | ❌ Nothing in this space |

`trajectory-eval` = DeepEval's agent metric set (in Go) + AgentRx/TrajDebug-style causal attribution, shipped as a single self-contained binary with **zero third-party dependencies**.

## Features

### Metrics (8)
| Metric | Type | Description |
|---|---|---|
| `tool_correctness` | deterministic | Jaccard overlap vs expected tool set, with missing/extra diff |
| `agent_loop_detection` | deterministic | 3 sub-signals: tool repetition / reasoning stall / call-graph cycle |
| `argument_correctness` | LLM judge | Per-call argument validity, located to step |
| `task_completion` | LLM judge | Did the agent finish the task? |
| `step_efficiency` | LLM judge + deterministic pre-check | Minimal steps / redundant work |
| `plan_quality` | LLM judge | Is the plan sound and complete? |
| `plan_adherence` | LLM judge + weighted-LCS | Did execution follow the plan? Locates first deviation |
| `tool_use` | LLM judge | Multi-turn tool selection + argument correctness |

### Causal attribution (differentiator)
- **Locate** key failing steps (with step index) via earliest-deviation backtracking
- **Evidence chain** — every failure carries source, assertion, detail
- **Root cause** — 7 categories (`wrong_tool`, `bad_arguments`, `plan_deviation`, `missing_step`, `redundant_loop`, `insufficient_info`, `unknown`) with confidence
- **Causal chain** — ordered failing step indices
- **Deterministic-first**: zero-LLM attribution works; LLM judge is only a fallback

### Extras
- **Multi-model jury** — panel of judges with majority voting (mitigates single-model bias)
- **Trajectory ingestion** — OpenAI chat format & LangGraph exports → Sample (record-then-evaluate, no runtime instrumentation)
- **Reports** — JSON (`traj-eval-report/v1`, CI artifact) + Markdown
- **V1 vs V2 diff** — regressed / fixed / unchanged + root-cause change detection
- **CI gate** — `--config gate.yaml`, exit code 0/1
- **Visualization UI** — zero-dependency web dashboard (embedded in the binary)
- **OpenAI-compatible LLM judge** — works with any endpoint (deepseek, openrouter, ...)

## Quick start

```bash
go build ./cmd/traj-eval

# Deterministic evaluation (zero LLM cost)
./traj-eval eval --dataset examples/golden.json --format markdown

# With LLM-as-Judge metrics (any OpenAI-compatible endpoint)
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_API_KEY=sk-...
export LLM_MODEL=deepseek-chat
./traj-eval eval --dataset examples/golden.json --format json --out report.json --judge --commit v1

# Visualize
./traj-eval serve --report report.json --addr 127.0.0.1:8787

# Version compare + CI gate
./traj-eval diff --base v1.json --current v2.json
./traj-eval gate --current v2.json --config gate.yaml
```

Exit code: `0` if all samples pass, `1` otherwise (CI gate semantics).

## Example output

```
## order_lookup_wrong_tool — FAIL

| Metric | Score | Passed | Reason |
|--------|-------|--------|--------|
| tool_correctness | 0.33 | ❌ | 1/2 expected tools called; missing: search_orders; unexpected: delete_order |

### Attribution
- Root cause: **wrong_tool** (confidence 1.00)
- Causal chain: steps [2]
Key failing steps:
- **Step 2** (tool_call): 1/2 expected tools called; missing: search_orders; unexpected: delete_order
  - `metric:tool_correctness`: ... — score=0.33 step=2
```

## Dataset formats

```json
{
  "samples": [
    {
      "name": "order_lookup_ok",
      "input": "Find all orders for user 42",
      "expected_tools": ["get_user", "search_orders"],
      "steps": [
        {"index": 0, "kind": "reasoning", "text": "I need the user record first"},
        {"index": 1, "kind": "tool_call", "tool_call": {"name": "get_user", "args": {"user_id": 42}}},
        {"index": 2, "kind": "observation", "text": "user found: Alice"}
      ]
    }
  ]
}
```

Also accepts JSONL (`#` comments allowed) and **external trajectory exports**: OpenAI chat-completions message lists and LangGraph node/edge graphs are auto-detected and converted.

## Architecture

```
trajectory/   core data model (Sample, Step, ToolCall, Span) — stdlib only
metrics/      metric interface + deterministic agent metrics
  judge/      LLM-as-Judge plumbing (Jury, Cache, RateLimit, Meter)
attribution/  causal failure attribution (locate → evidence → root cause)
ingest/       dataset loading + OpenAI/LangGraph trace conversion
report/       JSON + Markdown rendering, diff, gate
web/          zero-dependency visualization UI (embedded)
llmjudge/     OpenAI-compatible LLM client (stdlib only)
cmd/traj-eval CLI
```

Core packages have **zero third-party dependencies** — deterministic evaluation is fully offline.

## Roadmap

- [x] M0: deterministic metrics + causal attribution
- [x] M1: judge-based metrics + LLM adapter + diff/gate
- [x] M2: trajectory ingestion, multi-model jury, visualization UI
- [ ] Multi-target compare grid (matrix of models × datasets)
- [ ] AlignMetric (judge vs human agreement calibration)

## License

MIT
