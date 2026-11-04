# ARGUS-OS1 — GAIA multi-file eval run (2026-10-02, V3.x)

`argus-agent eval --evals argus-pre/evals/` (directory → all *.md)

## Results
- 2 files, 10 tasks: **8/10 = 80%** · Avg wall 12.8 s/task
- gaia_l1_subset.md: 5/5
- gaia_lite.md: 3/5
- Model: qwen2.5:7b via Ollama

## Reproduce
```bash
cargo run --release --bin argus-agent -- eval --evals argus-pre/evals/
```
