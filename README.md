# trajectory-eval

**Agent trajectory evaluation CLI — metrics + causal failure attribution in pure Go.**

`trajectory-eval` evaluates recorded agent trajectories: it scores tool-call correctness, detects loops, and — unlike most eval tools — **locates the exact step where the trajectory went wrong and explains why**, with an evidence chain and a root-cause classification.

> Demo "Demos show capability, Eval builds trust — but only if you can say *where* it failed."

## Why this exists

- **DeepEval** (17.9k★) is the most mature agent-eval framework, but it's Python and it *scores* failures without *locating* them.
- **AgentRx / TrajDebug** do causal attribution, but they're research-grade Python, not products.
- **Go ecosystem** has nothing in this space.

`trajectory-eval` = DeepEval's metric set (Go) + AgentRx/TrajDebug-style causal attribution, shipped as a single self-contained binary.

## Features

### M0 (current) — deterministic, zero-LLM
- `tool_correctness` — Jaccard overlap vs expected tool set, with missing/extra diff
- `agent_loop_detection` — 3 sub-signals (tool repetition / reasoning stall / call-graph cycle), DeepEval-aligned weights
- **Causal attribution** — locate key failing steps (with step index), build evidence chain, classify root cause (7 categories), trace causal chain via earliest-deviation backtracking
- **Judge fallback** — when deterministic signals are empty but the run failed, an LLM judge (optional) identifies the most suspicious step
- Reports: JSON (`traj-eval-report/v1`) + Markdown, CI-ready exit codes

### M1 (planned)
- Judge-based metrics: `argument_correctness`, `task_completion`, `step_efficiency`, `plan_quality`, `plan_adherence`
- OpenAI-compatible LLM judge adapter (works with any endpoint)
- Multi-turn `tool_use` metric
- V1 vs V2 diff + CI gate

## Quick start

```bash
go build ./cmd/traj-eval
./traj-eval eval --dataset examples/golden.json --format markdown
./traj-eval eval --dataset examples/golden.json --format json --out report.json --commit v1
```

Exit code: `0` if all samples pass, `1` otherwise (CI gate semantics).

## Example output (Markdown)

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

## Dataset format

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

Also accepts JSONL (one sample per line, `#` comments allowed).

## Architecture

```
trajectory/   core data model (Sample, Step, ToolCall, Span) — stdlib only
metrics/      metric interface + deterministic agent metrics
attribution/  causal failure attribution (locate → evidence → root cause)
ingest/       dataset loading (JSON/JSONL)
report/       JSON + Markdown rendering, summary
cmd/traj-eval CLI
```

Core packages have **zero third-party dependencies** — deterministic evaluation is fully offline.

## Roadmap

- [x] M0: deterministic metrics + causal attribution
- [ ] M1: judge-based metrics + LLM adapter + diff/gate
- [ ] M2: multi-turn tool_use, trajectory ingestion from LangGraph/OpenAI

## License

MIT
