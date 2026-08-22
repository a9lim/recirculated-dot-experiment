# AGENTS.md

Recirculated dots: can a frozen pretrained Gemma3-1B be taught to
think in opaque `<t>` tokens, with a recirculated activation wire
(Mozer et al. 2026) carrying serial state between dots (Pfau et al.
2024)? Trainable surface = one embedding row + one gate MLP; the
thesis (H3) is that this minimal surface suffices. This file is the
operating manual; read [docs/design.md](docs/design.md) before
changing anything substantive.

## Docs discipline

- `docs/design.md` — authoritative and **current-only**: idea,
  architecture, task design, training recipe, gates, decision
  register (D-numbers are stable anchors; code cites them).
- `docs/findings.md` — **current** experimental data and its
  interpretation.
- `docs/journal.md` — dated, chronological, **append-only** lab
  notebook. When canon moves: update design/findings in place, append
  the story (what was tried, rejected, superseded) to the journal.
  Never rewrite old journal entries.
- `docs/literature.md` — paper digests.

## Layout

- `recirculated_dot_experiment/wire.py` — the two-pass recirculation
  engine (inference): FA2 custom ops, branch-decomposed cache, manual
  steady-step CUDA graph, packed projections
  (`pack_model_projections`, shared with training), `load_model`.
- `recirculated_dot_experiment/tasks.py` — serialization + forced-
  choice eval. The only model-free module: importable on the Mac.
- `recirculated_dot_experiment/train.py` — think-scope BPTT
  (`think_outputs`), dots-alone control (`parallel_outputs`), naive
  reference for the gradient gate, `Surface`, execution-plan policy,
  eval adapters, `gate`/`run` CLI.
- `recirculated_dot_experiment/g0.py` — the wire's gate runner
  (identity, perplexity repro). Not a fourth module.
- `tests/` — shape/policy/serialization contracts; model-facing tests
  skip without flash-attn.
- `scripts/` — operator tooling, not modules: `lab.sh` (gates,
  probed detached runs and run queues with a DONE marker, scoring,
  status/watch/kill) and `score.py` (post-hoc scorer: untrained nulls,
  home/cross-arm cells, snapshot trajectories, knob transfer — driven
  by each checkpoint's own saved args). `example-queue.txt` is the
  current planned run.

Anti-cruft clause: exactly three modules (wire, tasks, train) until
something concrete forces a fourth. Task instance generators live in
the root package (`transformer_experiments.dot_tasks`) — import,
never re-implement.

## Hard rules

- Everything model-facing is CUDA + bf16 + flash-attn only (D10) and
  runs on the 4090 box: `ssh jobe`, repo at
  `~/Work/transformer-experiments/recirculated-dot-experiment`. There
  is no Mac/CPU/fp32 model path; don't add one.
- Sequences are composed in **id space** and never re-tokenized;
  answers are bare space-free single tokens (D8/D11).
- Per-shape compiled specialization is canon; warm largest-first
  before any clock; a timed loop that compiles is a bug (the audits
  raise).
- transformers is pinned `~=5.15.1`; the identity gate is the
  contract test for any bump.
- bf16 gates are null-calibrated: when a tolerance is needed, measure
  the self-noise null first, don't guess.
- bf16 is the frozen base's precision only. The trainable surface, its
  optimizer state, the gate's arithmetic, the loss, and every readout
  are fp32 (D2/D11): bf16 parameters silently drop AdamW updates and a
  bf16 sigmoid saturates to exactly 1. Don't cast the surface down.

## Gates (the contract — run after touching model-facing code)

```
python -m recirculated_dot_experiment.g0 identity      # wire = HF at alpha=0
python -m recirculated_dot_experiment.train gate       # BPTT vs naive reference
python -m recirculated_dot_experiment.g0 repro         # only when wire semantics change
python -m pytest tests -q && ruff check .              # both machines
```

Thresholds and their calibration are in design.md → Gates; current
witnessed numbers in findings.md. No green gates, no trust — and no
training run on an ungated tree.

## Workflow notes

- Edit on either machine; jobe runs. Sync through git (both ends are
  clones; keep them at the same commit before runs).
- Long jobe runs: detached (`nohup`/`tmux`), with a durable
  completion signal (DONE marker or notification) — in-session
  watchers have silently missed completions before.
- Training checkpoints default to `data/train/surface.pt` (ignored);
  `--resume` is exact (RNG-complete, step-addressable schedule).
- Runs go through `scripts/lab.sh train|queue`: every run is probed
  (`--steps 4`) before it is trained, logs land in `logs/TAG.log`
  (+ `logs/queue.log` for `watch`), the marker is `logs/queue-STATUS`,
  and each finished run is scored to `logs/TAG.score.log`. `--cosine`
  is a period, not the run length — pass it explicitly per run.
