# ARGUS-OS1 — argus-pre

Local-first AI agent core (V3.6) for ARGUS-OS1. Runs the agent loop entirely on Runs the agent loop entirely on
your machine: local or cloud LLM, file-based memory, safe tool execution.

> **Status: developer preview.** APIs and behavior are moving. Pin a release if
> you need a stable integration point.

## What it is

An agent loop (`Decide → Run → Compress → Repeat`) that:

- keeps **all state on disk** (JSON files: profile, notes, lessons) — no cloud;
- works with any **OpenAI-compatible LLM** (Ollama, LM Studio, llama-server);
- runs real tools: `file.read`, `file.write`, `shell` (approval-gated),
  `memory.set/note/lesson/recall`, `http.get` (SSRF-guarded), `web.search`,
  `browser.get` + CDP `browser.navigate/click/type/snapshot/eval/flow` (real Chrome);
- has **modes**: `default`, `plan` (read-only), `auto`, `bypass` — session state,
  never persisted grants;
- remembers across sessions (profile facts with keyword gating, typed notes,
  lessons), with a sliding-window reflection every N turns and a query rewriter
  for short referential follow-ups.

## Build & test

```bash
cd Services
cargo build --bin argus-agent
cargo test -p argus-pre --lib      # 46 unit tests (recover, SSRF, reflection, consolidate, heuristics)
```

Requires a running LLM server. Default: Ollama on `http://127.0.0.1:11434`
(model `qwen2.5:7b`). Also works with LM Studio (`--url http://127.0.0.1:1234`).

## Usage

```bash
# one-shot agent run against a goal
argus-agent run "Remember: my favorite color is blue, then read /etc/hostname" --cwd .
argus-agent run "…" --url http://127.0.0.1:1234 --model gemma-4-e4b-it   # LM Studio
argus-agent run "…" --mode plan      # read-only (write tools refused)
argus-agent run "…" --yes            # auto-approve dangerous calls (file.write, shell)

# memory
argus-agent init                     # create memory/ (profile.json, notes.json, lessons.json)
argus-agent memory add "note text" --kind knowledge
argus-agent memory list

# benchmark (self-contained, GAIA-L1-style smoke)
argus-agent bench --model qwen2.5:7b
```

State lives in `~/.argus-agent/` by default (override with `ARGUS_STATE_DIR`).

## Benchmark

| Eval | Tasks | Accuracy | Wall | Model |
|---|---|---|---|---|
| Smoke (`bench`) | 3 (memory/file/note) | **100%** | 8.5 s/task | qwen2.5:7b |
| GAIA-Lite (`evals/gaia_lite.md`) | 5 (web/file/shell/memory) | **60%** | 6.0 s/task | qwen2.5:7b |
| GAIA-L1 subset (`evals/gaia_l1_subset.md`) | 5 (multi-step) | **60%** | 8.4 s/task | qwen2.5:7b |
| GAIA multi-file (`evals/`) | 10 | **80%** | 12.8 s/task | qwen2.5:7b |

Shell/file tool tasks consistently pass; memory/web recall on 7B is the known
weak spot (reference: Atomic Agent 9B on GAIA L1 ≈ 52.8%).

```bash
argus-agent bench --model qwen2.5:7b
argus-agent eval --evals evals/gaia_lite.md
argus-agent eval --evals evals/gaia_l1_subset.md
```

### GAIA-Lite eval (tool-use)

5 tool-use tasks (web, file, shell, memory): **3/5 = 60% · 6.0 s/task** on
`qwen2.5:7b`. Full run: `benchmarks/gaia-lite-2026-10-02.md`.

```bash
argus-agent eval --evals evals/gaia_lite.md
```

### Obsidian export

```bash
argus-agent memory export --vault <path>   # notes/ lessons/ procedures/ as Markdown + [[wikilinks]]
argus-agent memory import --vault <path>   # round-trip: read vault .md (frontmatter tags) into memory
argus-agent serve --port 8797              # skill hub: GET /skills, GET /skills/<name>
argus-agent skill browse --remote http://127.0.0.1:8797
```

## Architecture

```
src/
  lib.rs            # library root (agent + memory)
  agent/mod.rs      # AgentLoop (P1), modes (P4), LlmModel, tool registry,
                    #   recover layer, rewriter (P3A), reflection (P3B)
  memory/mod.rs     # file-based memory: profile/notes/lessons + reflection parser
  bin/argus_agent.rs# CLI: init / run / memory / bench
  main.rs           # argus-pre pre-experimental studies (cell-micromanipulation)
```

Design notes: the loop sends a compact prompt (stable prefix + memory pointers)
to the model; one inference yields one JSON array of tool calls; results are
compressed rather than re-encoded every turn (KV-cache friendly). See
`docs/PATTERNS_V1.md` (internal) for the full pattern list.

## Security

- `file.write` and `shell` are **dangerous**: refused without `--yes` (or
  `auto`/`bypass` mode).
- `http.get` is **SSRF-guarded**: private/loopback/link-local hosts blocked,
  http(s) only.
- `plan` mode is strictly read-only.
- Modes are session state only — a `bypass` never survives a restart.

## Roadmap

- **V1.2 (done)** — web (`http.get` SSRF-guarded, `web.search`), benchmark, OSS
  packaging (MIT, CONTRIBUTING, README).
- **V1.3 (done)** — `browser.get`, fuzzy call recovery (typo-tolerant), loop guard;
  memory consolidation.
- **V1.4 (done)** — GAIA-Lite eval harness, Obsidian export, CDP browser
  (`navigate/click/type/snapshot`); grant kit (`GRANT_KIT.md`).
- **V1.5 (done)** — plugin skills (SKILL.md playbooks + scripts), multi-worker
  fusion (`argus-agent fusion`), OSC application draft (`OSC_APPLICATION_2026.md`, submission on hold).
- **V1.6 (done)** — procedures from consolidation; skill registry (`skill search/install`);
  GAIA-L1 subset eval (60% · 8.4 s/task).
- **V1.7 (done)** — CDP browser flows (`browser.eval` + multi-step `browser.flow`);
  skills taps (`skill tap add/remove/list`, install from GitHub owner/repo).
- **V1.8 (done)** — Obsidian round-trip (`memory import --vault`); browser recipes
  (web-research/form-fill/login-flow with ready `browser.flow` templates);
  skill hub server (`serve` — `/skills`, `/skills/<name>`; `skill browse --remote`).
- **V1.9 (done)** — hub push/registry (`skill push --remote`, POST /skills on
  serve); browser recipe library (9 recipes); `DOCS.md` full user guide.
- **V2.0 (done)** — sessions persistence (`--session <id>`, `session list`),
  durable tasks (`task add/list/run`, cron-like); Obsidian polish (inline #tags,
  dedup import, `date` in frontmatter).
- **V2.1 (done)** — `serve --tick S` daemon (skill hub + task scheduler); Obsidian
  polish: `kind` from frontmatter, skip `_`/`.` template dirs.
- **V2.2 (done)** — hybrid vector recall (offline n-gram embeddings + cosine,
  no external services); `memory recall`, used in `memory.recall` tool.
- **V2.3 (done)** — plugin hooks (`plugin add/list/remove`, events
  before_step/after_step/on_done/on_error, env ARGUS_*); session browser
  (`session show/tail/search/rm/tui`).
- **V2.4 (done)** — optional Ollama embeddings (`memory recall --vector ollama`,
  auto-detect + fallback to local ngram; `memory embed` check); Obsidian export
  `title:` from first heading. Full audit passed (build/test/clippy/CLI).
- **V2.5 (done)** — embedding cache (`embeddings.json`, ngram always + ollama
  opt-in, computed on add); `task tui` (interactive: run/delete/add); session
  TUI tail/delete.
- **V2.6 (done)** — Obsidian round-trip polish (import `date:` → created_at,
  `src:<file>` tag, export `source:`); session search context (`↳` match line,
  `--n` limit).
- **V2.7 (done)** — vector-recall persist (`rebuild_ngram_cache`, auto-fix on load,
  `memory embed-build`); timer hooks (`plugin add --event timer --interval N`,
  ticked each loop step, fire-and-forget).
- **V2.8 (done)** — timer last-fire persisted to `plugins.json` (no refire across
  runs); session search ranking (score = transcript matches×10 + goal + freshness).
- **V2.9 (done)** — eval harness multi-file (`--evals <file|dir|list>`, per-file
  aggregate); skill versions (`skill info`, `skill update` force reinstall).
- **V3.0 (done)** — procedures LLM-distilled (Model::distill_procedure, JSON
  activation+steps, heuristic fallback); web-search providers (Brave/SearXNG via
  env, DDG default + fallback).
- **V3.1 (done)** — lessons LLM-distilled (Model::distill_lesson, JSON
  activation+principle); vector index export (`memory export-vectors`).
- **V3.2 (done)** — Obsidian auto-sync (`memory sync --vault --mode`, auto-sync
  after run via `ARGUS_VAULT_SYNC`); browser-recipe gallery (`skill recipes`).
- **V3.3 (done)** — skill hub auth (`ARGUS_HUB_TOKEN`, 401 without Bearer);
  multi-session concurrency (`task run` executes due tasks in parallel threads).
- **V3.4 (done)** — sessions dedup (`session dedup`, save-skip unchanged transcript,
  drop empty); serve task dashboard (`GET /tasks`).
- **V3.5 (done)** — memory voting + eviction (`memory vote`, limits by usefulness);
  hub catalog (`?q=`, `/version`); SSRF DNS resolve + shell sandbox; `browser.screenshot`;
  installer `scripts/install.sh`, CI workflow; src-dedup import; more tests (14).
- **V3.6 (done)** — memory link graph (auto RELATES_TO by tags, 1-hop recall,
  `memory links/link`); import from pi (`skill import-pi`, copies `~/.pi/agent` skills);
  MCP client (`mcp add/tools/call`, JSON-RPC stdio, mcp.* agent tools).
- **V3.7** — Telegram remote; graph-vector hybrid recall.

See [GRANT_KIT.md](GRANT_KIT.md) for the open-source / grant package and
[DOCS.md](DOCS.md) for the full user guide.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Jaba Tqemaladze.
