# Design

Decisions ledger. One dated entry per decision, rationale in one line.
Anti-cruft clause: the package holds exactly three modules — `wire`,
`tasks`, `train` — until something concrete forces a fourth.

## Idea

Combine recirculation (Mozer et al. 2026) with filler-token thinking
(Pfau et al. 2024): a frozen pretrained model thinks in opaque `<t>`
tokens while a recirculated activation wire carries serial state
between positions.

```
normal CoT:      text  text  text  answer
coconut:         text  neuralese    answer
dots:            text  <t>   <t>    answer          (no wire between dots)
recirculation:   text+s text+s      answer+s        (wire, no extra time steps)
this project:    text[+s] <t>+s <t>+s ... answer+s  (wire x dots)
```

Each `<t>` is one application of a recurrent cell whose body is frozen
layers dest→source; the dot count is the unrolled time axis.

## Hypotheses

- **H1 (expressivity).** Dots+wire solves inherently serial tasks
  (S5 word problems, parity, reachability) with accuracy scaling in
  dot budget k. Dots alone stay flat (Pfau et al.'s TC0 bound);
  wire alone is capped by content-token count.
- **H2 (learnability).** BPTT through the wire gives answer-only
  supervision a credit-assignment path into the dot span — no CoT
  decomposition data, no coconut curriculum. (Fallback: curriculum;
  see D6 lesson.)
- **H3 (naturalness).** The minimal trainable surface — one embedding
  row plus a gating MLP, base frozen — suffices; the model's own
  circuits do the compute. This is the thesis. A clean failure of H3
  (gating insufficient, adapter needed) is a finding, not a rewrite:
  the prior full-adapter experiment showed the wire can carry serial
  thought; the question is whether it does so *natively*.

## Decisions

**D1 — Wire = two-pass, paper-faithful.** (2026-08-20) Each column
computed twice; readout (and loss) on first-pass logits; second pass
of column j runs alongside first pass of column j+1 and refreshes
column j's KV cache from the destination layer up; destination input
mixed per Eq. 1 with norm-matched source. Rationale: inherits the
paper's validated out-of-box behavior and layer-pair priors.

**D2 — Trainable surface = rung (a) only.** (2026-08-20) Learned
`<t>` embedding row + the paper's α,β-MLP (2 hidden GELU layers,
hidden = d_model, LN at input, input = concat(source, dest), sigmoid
output, vector-valued, init α=0.1/β=0.9). Base frozen. Full-adapter
rung documented as fallback only — tried in the prior experiment:
worked, but destroyed the naturalness the project exists to test.

**D3 — Task suite = discriminating 2×2.** (2026-08-20)
{dots, none} × {wire, none}, CoT topline. Theory-clean serial rows:
S5 word problems (NC1-complete, outside Pfau et al.'s filler bound)
and graph reachability (NL-complete, likewise, modulo standard
separations). Parity is in TC0 — the bound is formally silent on it —
so it rides as the empirical learnability probe, not an expressivity
claim. Parallelizable control: 3SUM (fillers-sufficient per Pfau et
al.). Money plot: accuracy vs k at eval (inference-time scaling with
zero legible tokens). Generators are shared with the sibling in the
root package, `transformer_experiments/dot_tasks.py` — import, don't
re-implement; `min_path` on reachability scales serial depth and
suppresses shortcut heuristics; `render_cot` is the topline/curriculum
trace.

**D4 — Halting is sampling.** (2026-08-20) `<t>` is a real vocab
token; the LM head chooses dot-vs-answer natively. v0 trains
teacher-forced fixed k and evals a k-sweep; adaptive dotting later,
length penalty only if needed.

**D5 — Wire scope is a config flag.** (2026-08-20) `full` (all
positions, paper-style; serial prefill) vs `think` (dot span + answer
only; parallel prefill, dots become the sole serial channel — forces
utilization and gives a clean test-time-compute knob). Note: in
`full` scope the token-conditional gate can *learn* to concentrate α
on dots; where α mass migrates is an interp readout in its own right.

**D6 — Training mechanics.** (2026-08-20) BPTT serial over positions,
parallel over batch — fat batches on short formal sequences (the
prior run's ~20% GPU utilization was serial-loop-with-small-batch).
Truncated BPTT (detach every m dots) as escape hatch, noting it caps
the credit-assignment horizon. Curriculum fallback kept ready.
Hard-won lesson (a9, prior experiment): **the LR schedule must reset
at every curriculum stage** — each transition is a distribution
shift, and a single global decay leaves the model unable to adapt
(one such run fell apart 8 hours in). Per-stage warmup/decay is fine;
checkpoint at stage boundaries.

**D7 — Model = Gemma3 1B PT.** (2026-08-20) The paper's
best-characterized receptive model: source/dest {11,4}, convex
norm-ratio mix (β=1−α), α ramped over the first 10 positions.
Configurable; 270M as a possible fast-iteration mode would need its
own sweep gate (uncharacterized in the paper).

**D8 — `<t>` token and serialization pins.** (2026-08-20) `<t>` :=
Gemma3's reserved `<unused0>` slot — already a single token with an
existing untrained embedding row, tied with the LM head; no vocab
resize, and training the one row shapes both how the dot is read and
when it is emitted. Tokenizer facts that pin the format: Gemma3
splits `" 1"` into two tokens (standalone space + digit) but `"1"`,
`"yes"`, `"no"` are single tokens — so sequences are composed with
**no space before the answer** (prompt ends `:`/`?`, then dots, then
the bare answer token). `render` answers are already space-free;
composition is consumer-side.

**Gate G0 — implementation correctness.** Before any training:
(i) α=0 two-pass run reproduces plain-forward logits exactly (cache
surgery no-op check); (ii) reproduce the paper's untrained perplexity
reduction on Gemma3 1B, {11,4}, α=0.15, convex norm-ratio + ramp —
clearly positive, order 10% on PG19/arXiv slices, ~4% on C4;
(iii) a bad layer pair shows ~no gain (we see their landscape, not an
artifact). 512-token windows so every sliding-window layer (window
512) is effectively global and the KV cache stays a plain
index-assignable tensor — a deliberate deviation from the paper's
1024 (this is a validation gate, not a replication). No repro, no
trust. PG19 comes from the emozilla/pg19 parquet mirror (datasets>=5
refuses the script-based deepmind/pg19).

**D9 — training-path design.** (2026-08-20, incorporating the Codex
perf consult) The BPTT path gets a *functional token-chunked cache*
(the wire's in-place lanes are inference-only): per layer, an
immutable detached prompt prefix + a tuple of refreshed one-token
tensors + the newest first-pass tensor; dual views built by cat, no
tensor participating in the graph ever overwritten. Persistent KV at
B=64, span 40 is ~52 MiB — the real memory is saved attention
operands (GQA repeat under masks → up to ~8 GiB), so checkpoint each
decoder-layer call (`use_reentrant=False`). lm_head runs ONLY on
supervised answer positions — never on dots. Before any training
run: a gradient gate (B=1, span 3–4) comparing the functional dual
cache against an out-of-place sequential reference for loss and
grads of the embedding row + gate MLP. Curriculum LR rule per D6.
torch.compile belongs to THIS path, not the inference wire: the
functional cache is mutation-free, so the graph is pure and inductor
has nothing to functionalize — the measured compile losses on the
inference wire (broadcast materialization, non-re-inplaced mutations;
findings 2026-08-20 round 2) do not apply. Keep the `_wire_sdpa`
stride-0 GQA expansion in the training attention path too.

**D10 — canonical wire: one path.** (2026-08-20, a9 ratified) The
compiled-FA2 configuration is not the default but the *only* path:
CUDA, half precision, FlashAttention kvcache for the dual pass,
torch.compile(fullgraph) over the serial slab. The sdpa dual-pass
implementation, mask machinery, and backend/compile flags are
stripped (git history keeps them; findings keeps the measurements).
Rationale: training and eval share one set of numerics — no mid-run
surprises from a path switch — and the Mac is not a target. Cost
knowingly accepted: the fp32 identity gate retired with the sdpa
path; G0's identity now runs bf16 with thresholds calibrated against
the measured null (plain HF forward vs itself under a kernel-tiling
change) — the engine at α=0 sits *inside* that null (mean |Δlogit|
5.3e-2 vs 5.5e-2; top-1 0.982 vs 0.965; ppl rel 7e-4). The last fp32
proof of the same algorithm: max |Δlogit| 1.2e-4 (2026-08-20).

## Open

- Reachability negatives are unconstrained; if shortcut heuristics
  show up in the 2×2, add near-miss negatives to the shared module.
