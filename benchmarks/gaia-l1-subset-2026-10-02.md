# ARGUS-OS1 — GAIA-L1 subset eval run (2026-10-02)

Multi-step tool-use eval (V1.6 harness), real local model.

## Setup
- Model: `qwen2.5:7b` (Q4_K_M) via Ollama `127.0.0.1:11434`
- Harness: `evals/gaia_l1_subset.md` (5 multi-step tasks: web/shell/file/memory)

## Results

| # | Task | Type | Result | Wall |
|---|---|---|---|---|
| T1 | un-countries (web) | contains | ❌ | 28.9 s |
| T2 | uname-arch (shell) | contains | ✅ | 2.9 s |
| T3 | create-report (file) | file_contains | ✅ | 3.6 s |
| T4 | remember-date (memory) | contains | ❌ | 3.3 s |
| T5 | shell-multiply (shell) | contains | ✅ | 3.1 s |

**Accuracy: 3/5 = 60.0% · Avg wall: 8.4 s/task** (standalone run; see `gaia-multi-2026-10-02.md` for the combined 8/10 = 80% run)

## Notes
- T4: 7B model answers from immediate context without persisting a profile fact
  (same as GAIA-Lite T3); reflection cadence (every 3 turns) doesn't fire on
  short 1–2 turn tasks. Known lever: explicit memory-set instruction or longer task.
- T1: web search worked but final reply omitted the exact token "193".
- Shell/file tasks consistently pass (T2/T3/T5).
- Reference: Atomic Agent 9B on GAIA L1 ≈ 52.8%.

## Reproduce
```bash
cd Services
cargo run --release --bin argus-agent -- eval --evals argus-pre/evals/gaia_l1_subset.md
```
