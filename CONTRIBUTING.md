# Contributing to ARGUS-OS1 (argus-pre)

Thanks for considering a contribution. This is a developer preview — small,
focused PRs move fastest.

## Getting started
- Fork, branch (`feat/…`, `fix/…`), open a PR against `main`.
- Keep the diff small; one concern per PR.

## Rules
- **Tests colocated with code.** Any behavioral change ships with `#[cfg(test)]`
  unit tests in the same module (see `src/agent/mod.rs`). Run:
  `cargo test -p argus-pre --lib`.
- **Behavioral changes behind flags.** New memory/loop features go behind a
  `LoopConfig` flag with a documented default; toggling it off must not change
  existing behavior.
- **Security first.** Anything that writes, executes, or reaches the network is
  `dangerous`/approval-gated or SSRF-guarded. Never relax these for convenience.
- **No secrets.** Keys and tokens never enter the repo or the code.
- Comments in Russian (project convention); identifiers/docs in English.

## Benchmark
When touching the agent loop, run `argus-agent bench --model <model>` before/after
and report the accuracy/wall-time delta in the PR description.

## License
By contributing you agree your work is licensed under MIT (see LICENSE).
