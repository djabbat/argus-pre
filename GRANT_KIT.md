# ARGUS-OS1 — Open-Source Grant Kit (2026-10-02, V3.6)

> Package for the **Open Source Collective (fiscal hosting Apply)**, **Foresight
> Institute ($50k)** and **Sovereign Tech Fund (€52.5k / 12 mo)** submissions.
> ⚠️ **STF: REJECTED 2026-11-10** (ID `YelaLvMK`) — «does not meet our criteria».
> Постмортем: `~/Desktop/Services/grants/GLA/SovereignTechFund/STF_REJECTION_2026-11-10.md`.
> Everything here is public-safe. Internal design notes live in
> `docs/PATTERNS_V1.md` (not committed) — do not copy verbatim.

---

## 1. One-line pitch

A single-binary, **local-first AI agent** that runs the whole agent loop and
your data on your machine — no cloud, no API fees, full privacy — and works
with any OpenAI-compatible model (Ollama, LM Studio, llama.cpp) or cloud.

## 2. What it is

ARGUS-OS1 is a local-first AI agent core written in **Rust** (fast, single
binary, MIT). The agent loop (`Decide → Run → Compress → Repeat`) and **all
state** live on your disk: sessions, memory, tasks, skills, config — plain
files, no vendor lock-in. It drives real tools (files, shell, web, a real
Chrome browser over CDP), remembers across sessions (profile facts, typed
notes, lessons & procedures, link graph, hybrid vector recall), schedules
durable tasks, exposes plugin hooks, a skill hub, multi-worker fusion, and a
self-contained eval harness.

## 3. Why local-first matters (the problem)

Cloud agent platforms hold your sessions, prompts, files and secrets. A
local-first agent keeps everything local:
- **Privacy by default** — nothing leaves the machine unless you opt in.
- **Zero inference cost** — run open-weight models locally via `llama.cpp`/Ollama.
- **No lock-in** — inspect the prompt, swap models, export state (Obsidian,
  vector index), wipe everything.
- **Works offline** and on consumer hardware: small quantized models now handle
  multi-step tool use well enough for real work.

Relevant for researchers, clinics, and anyone handling sensitive data (the
author is an MD in longevity research — the data stays on-machine).

## 4. Surface (what it can do)

| Area | Capabilities |
|---|---|
| **Memory** | profile facts (keyword-gated), typed notes (event/behavior/knowledge/skill), lessons & procedures (LLM-distilled), consolidation, sliding-window reflection, query rewriter, **hybrid vector recall** (local n-gram + optional Ollama embeddings), **usefulness voting**, **eviction by usefulness**, **link graph** (auto `RELATES_TO`, 1-hop recall), Obsidian round-trip + auto-sync |
| **Files / shell** | read (compressed), write, shell — writes/shell **approval-gated**; **shell sandbox** (dangerous patterns blocked even with `--yes`); file writes scoped to cwd |
| **Web** | `http.get` (**SSRF-guarded** incl. DNS resolve), `web.search` (DDG default; Brave/SearXNG via env), `browser.get` (headless-Chrome render, JS executed) |
| **Browser (CDP)** | `browser.navigate/click/type/snapshot/screenshot/eval/flow` — real Chrome control, SSRF-guarded |
| **Sessions** | persisted (`run --session`), browser/TUI, search with ranking, dedup |
| **Tasks** | durable cron-like tasks (`task add/list/run/tui`), **parallel execution**, daemon scheduler + `GET /tasks` dashboard |
| **Plugins** | hooks (`before_step/after_step/on_done/on_error`), **timer hooks** (interval, persisted last-fire) |
| **Skills** | SKILL.md playbooks, registry + GitHub taps, **hub** (auth + catalog + push/browse), browser-recipe gallery, versions, **import from pi** |
| **MCP** | JSON-RPC stdio client — connect external MCP servers (`mcp.tools`/`mcp.call`) |
| **Fusion** | planner LLM splits a goal; worker pool runs subtasks in parallel |
| **Eval** | self-contained harness, **multi-file** (`--evals <file|dir|list>`), per-file aggregate |
| **Safety** | modes: default / plan (read-only) / auto / bypass (session-only, never persisted); loop guard against repeated calls |

## 5. Architecture (short)

```
src/
  agent/    AgentLoop, LLM client (OpenAI-compat), tool registry, recover layer,
            rewriter, reflection, loop guard, sandbox
  memory/   profile/notes/lessons/procedures/links, embeddings cache, voting,
            eviction, consolidation (LLM distill), Obsidian import/export
  browser/  headless Chrome via CDP (navigate/click/type/snapshot/screenshot/flow)
  skills/   SKILL.md registry, taps, hub server (auth/catalog), recipes
  mcp/      JSON-RPC stdio MCP client
  plugins/  hooks + timer hooks (persisted)
  bin/      CLI: init/run/memory/skill/session/task/plugin/mcp/bench/eval/fusion/serve
```

Design borrows from the state of the art (Atomic Agent, pi, Hermes) and
simplifies: file-based memory, heuristic + LLM consolidation, no background
job daemon required.

## 6. Benchmarks (public, reproducible)

`argus-agent bench` / `argus-agent eval`, real local model **qwen2.5:7b** via
Ollama on a consumer machine (33 GB RAM, Linux):

| Eval | Tasks | Accuracy | Wall |
|---|---|---|---|
| Smoke (`bench`) | 3 (memory/file/note) | **100%** | 8.5 s/task |
| GAIA-Lite (`evals/gaia_lite.md`) | 5 | **60%** | 6.0 s/task |
| GAIA-L1 subset (`evals/gaia_l1_subset.md`) | 5 | **60%** standalone; **5/5** within multi-file run | 3–8 s/task |
| GAIA multi-file (`evals/`) | 10 | **80%** | 12.8 s/task |

Reference: Atomic Agent 9B on GAIA L1 ≈ 52.8% — same ballpark on a much
lighter set with a **7B** model. Shell/file tool tasks consistently pass;
memory-recall on 7B is the known weak spot (bigger model or explicit
memory-set instruction is the lever).

Reproduce:
```bash
cargo run --release --bin argus-agent -- bench --model qwen2.5:7b
cargo run --release --bin argus-agent -- eval --evals argus-pre/evals/
```
Full runs: `benchmarks/*.md` (committed).

## 7. Engineering health

- **46 unit tests (2 ignored) + 2 CDP integration tests**, clippy `-D warnings` clean
  (incl. `--all-targets`), `cargo fmt` enforced.
- **CI** (`.github/workflows/ci.yml`): fmt + clippy + test + release on push.
- **Installer**: `scripts/install.sh` (build + `~/.local/bin`, alias `atag`).
- **Security**: SSRF-guard (string + DNS resolve), shell sandbox, approval
  gates, session-only modes, loop guard.

## 8. Roadmap (6 months — see STF/OSC fiscal-hosting applications)

M1–2: v1.0 stabilization, extend browser-recipe library (9 shipped), hub public deploy.
M3–4: Telegram remote, MCP stream framing, graph+vector hybrid recall.
M5–6: multi-session TUI, community (Discord/issues), v1.1 release, docs.

## 9. Why us / differentiators

- **One Rust binary**, no Node/cloud runtime; local-first by design.
- **Honest safety**: approval-gated dangerous actions, SSRF-guarded + DNS-safe
  web, shell sandbox, session-only modes.
- **Open & inspectable**: plain-JSON state, Obsidian/vector export, MIT.
- **Practical for sensitive-data users** (MD author, longevity research —
  data stays on-machine).
- **Reproducible**: self-contained eval + benchmark artifacts committed.

## 10. Repo / contact

- Repo: `github.com/djabbat/argus-pre`
- License: MIT (`argus-pre/LICENSE`)
- Author: Jaba Tqemaladze, MD — Georgia Longevity Alliance — jaba@longevity.ge

---

## Files in this kit

| File | Purpose |
|---|---|
| `GRANT_KIT.md` | this summary (public) |
| `OSC_FISCAL_HOSTING_APPLICATION.md` | Apply to OSC fiscal host (all form fields) |
| `OSC_APPLICATION_2026.md` | переориентирован (грант OSC не существует; исторический) |
| `OSC_COVER_LETTER.md` | one-page cover letter for OSC submission |
| `STF_APPLICATION_2026.md` | filled Sovereign Tech Fund application (€52.5k, all form fields) |
| `FORESIGHT_RFP_APPLICATION.md` | справочная аналитика; реальная RFP Call 3 подана 31.10.2026 ($100K) |
| `README.md` / `DOCS.md` | project docs |
| `LICENSE` / `CONTRIBUTING.md` | legal & contribution guide |
| `scripts/install.sh` / `.github/workflows/ci.yml` | installer & CI |
| `benchmarks/*.md` | reproducible eval data |
| `LICENSE` / `CONTRIBUTING.md` | legal & contribution |
