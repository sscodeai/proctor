# Proctor

[English](README.md) | [日本語](README.ja.md)

**Agent Eval — the exam system for AI agents: score, audit, and explain their trajectories.**

`Proctor` is an **agent evaluation (Agent Eval / LLMOps) tool**. It evaluates recorded agent trajectories, scores tool-call correctness, detects loops, checks plan adherence — and unlike most eval tools, **locates the exact step where an agent went wrong and explains why**, with an evidence chain and a root-cause classification.

> **Agent Eval** — what this is: measuring how well an AI agent performs a task, *and diagnosing why it failed*. Not just "did it pass" — *where* and *why* it failed. That's the difference between trusting an agent and betting on it.

> *Demos show capability, Eval builds trust* — but only if you can say *where* it failed.

![demo](docs/demo.gif)

## Why this exists

| Tool | Language | What it does | Gap |
|---|---|---|---|
| **DeepEval** (17.9k★) | Python | Scores agent trajectories | ❌ Scores failures but doesn't *locate* them |
| **AgentRx / TrajDebug** | Python | Causal attribution research | ❌ Research-grade, not a product |
| **Go ecosystem** | Go | — | ❌ Nothing in this space |

`Proctor` = DeepEval's agent metric set (in Go) + AgentRx/TrajDebug-style causal attribution, shipped as a single self-contained binary with **zero third-party dependencies**.

## What Proctor answers

Three questions every agent team needs answered — and most eval tools only answer the first:

| Question | How Proctor answers it |
|---|---|
| **Did the agent do the right thing?** | 8 metrics: correctness, loop detection, completion, efficiency, plan quality/adherence… |
| **Where did it go wrong?** | Causal attribution: locates the exact failing step, earliest-deviation backtracking |
| **Why?** | Evidence chain + 7-category root cause (`wrong_tool`, `bad_arguments`, `plan_deviation`, `missing_step`, `redundant_loop`, `insufficient_info`, `unknown`) with confidence |

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
- **Root cause** — 7 categories with confidence
- **Causal chain** — ordered failing step indices
- **Deterministic-first**: zero-LLM attribution works; LLM judge is only a fallback

### Judge integrity (LLM-as-a-Judge, done right)
- **Multi-model jury** — panel of judges with majority voting (mitigates single-model bias from MT-Bench)
- **Judge-human alignment** (`align`) — Pearson / Cohen's kappa / accuracy / bias vs human gold labels ("who validates the validators")

### Engineering loop
- **Trajectory ingestion** — OpenAI chat format & LangGraph exports → Sample (record-then-evaluate, no runtime instrumentation)
- **Reports** — JSON (`proctor-report/v1`, CI artifact) + Markdown
- **V1 vs V2 diff** — regressed / fixed / unchanged + root-cause change detection
- **CI gate** — `--config gate.yaml`, exit code 0/1
- **Visualization UI** — zero-dependency web dashboard (embedded in the binary)
- **Model × dataset matrix** (`compare`) — compare multiple models across datasets, auto-picks the best
- **OpenAI-compatible LLM judge** — works with any endpoint (deepseek, openrouter, ...)

## Quick start

```bash
go build ./cmd/proctor

# Deterministic evaluation (zero LLM cost)
./proctor eval --dataset examples/golden.json --format markdown

# With LLM-as-Judge metrics (any OpenAI-compatible endpoint)
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_API_KEY=sk-...
export LLM_MODEL=deepseek-chat
./proctor eval --dataset examples/golden.json --format json --out report.json --judge --commit v1 \
  --parallel 4 --cache .cache --rps 10   # concurrent, cached, rate-limited

# Visualize
./proctor serve --report report.json --addr 127.0.0.1:8787

# Version compare + CI gate
./proctor diff --base v1.json --current v2.json
./proctor gate --current v2.json --config gate.yaml

# Judge-human alignment (needs human labels in the dataset)
./proctor align --dataset golden_labels.json --report report.json

# Model × dataset matrix
./proctor compare \
  --report model=deepseek-flash,dataset=cs:report_a.json \
  --report model=deepseek-pro,dataset=cs:report_b.json
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
cmd/proctor   CLI
```

Core packages have **zero third-party dependencies** — deterministic evaluation is fully offline.

## Research grounding

Proctor's judge-based metrics and attribution methodology draw from published papers, public preprints, and open-source evaluation systems:

| Work | Venue / source | What Proctor uses |
|---|---|---|
| **G-Eval** ([arXiv 2303.16634](https://arxiv.org/abs/2303.16634)) | EMNLP 2023 | LLM-as-a-Judge origin — rubric-grounded scoring aligned with human judgment; the foundation for all 6 judge metrics |
| **Judging LLM-as-a-Judge with MT-Bench** ([arXiv 2306.05685](https://arxiv.org/abs/2306.05685)) | NeurIPS 2023 | Documented judge biases (position / verbosity / self-preference) — motivates Proctor's multi-model jury |
| **Replacing Judges with Juries** ([arXiv 2404.18796](https://arxiv.org/abs/2404.18796)) | 2024 | Multi-model panel reduces single-judge bias — the jury voting design |
| **RAGAS** ([arXiv 2309.15217](https://arxiv.org/abs/2309.15217)) | EACL 2023 | Decomposing quality into computable metrics — metric-design reference |

### Attribution lineage

| Source | What Proctor borrows |
|---|---|
| **DeepEval** (confident-ai/deepeval) | The 8 agent metric semantics (tool correctness, loop detection weights 0.40/0.35/0.25, weighted-LCS plan adherence) |
| **AgentRx** (microsoft) | "constraint → check → locate" pipeline: which step violated which constraint |
| **TrajDebug** (THU-KEG) | Expected-vs-actual diff + backtracking to the earliest deviation step, evidence chain |

Honest note: no single paper defines Proctor — it is an engineering fusion of judge methodology (G-Eval/MT-Bench), metric sets (DeepEval), and causal attribution (AgentRx/TrajDebug), plus a deterministic-first attribution design that works with zero LLM cost.

## Roadmap

- [x] M0: deterministic metrics + causal attribution
- [x] M1: judge-based metrics + LLM adapter + diff/gate
- [x] M2: trajectory ingestion, multi-model jury, visualization UI
- [ ] Multi-target compare grid (matrix of models × datasets)
- [ ] AlignMetric (judge vs human agreement calibration)

## License

MIT
