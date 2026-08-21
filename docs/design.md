# Design

The authoritative current state: idea, architecture, task design,
training recipe, gates, and the decision register (stable D-numbers,
cited from code). History, superseded detail, and the story of how
each piece got here live in [journal.md](journal.md); measurements in
[findings.md](findings.md). Anti-cruft clause: the package holds
exactly three modules — `wire`, `tasks`, `train` — until something
concrete forces a fourth (`g0` is the wire's gate runner, not a
module).

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
  see D6.)
- **H3 (naturalness).** The minimal trainable surface — one embedding
  row plus a gating MLP, base frozen — suffices; the model's own
  circuits do the compute. This is the thesis. A clean failure of H3
  (gating insufficient, adapter needed) is a finding, not a rewrite:
  the prior full-adapter experiment showed the wire can carry serial
  thought; the question is whether it does so *natively*.

## Architecture: the wire

**The recurrence (D1, D7).** Gemma3-1B PT, source/dest {11,4}
(0-indexed, empirically resolved). Each column is computed twice:
readout (and loss) come from first-pass logits; the second pass of
column t−1 runs alongside the first pass of column t and refreshes
t−1's KV for layers dest+1..top, with the destination input mixed per
the paper's Eq. 1 — `β·h_d + α·(‖h_d‖/‖h_s‖)·h_s`, norm-matched,
α ramped over the first 10 positions untrained (α=0.15 convex) and
produced by the gate MLP when trained. Snapshot rule: both passes of a
step read the *same* visible past; the refresh joins it only after
both have run, and the first pass sees column t−1 only at first-pass
fidelity.

**Canonical execution (D10, D13).** One path: CUDA, bf16, one
attention library end to end — models load with
`attn_implementation="wire_attention"`, whose fallback tier defers to
HF's flash_attention_2 with a registered mask-interface pair, so
baselines run the engine's kernels. There is no fp32 or CPU path.
Structure of the inference engine (`wire.RecirculationEngine`):

- **Pass A**: layers 0..dest never see refreshed state, so the bottom
  slab prefills the whole sequence in parallel (fixed B=64 compiled
  chunks, final chunk row-index padded).
- **Serial slab**: layers dest+1..top, the dual pass batched as one
  [2B] call through the custom op `wire::fa2_kvcache`
  (per-row `cache_seqlens` express the per-lane prefix lengths).
- **Branch-decomposed cache**: both branches read one physical
  refreshed prefix 0..t−2 plus a one-slot side buffer via duplicate
  `cache_batch_idx`; a compiled two-key softmax with fp32 LSE merge
  adds each branch's distinct keys. Halves slab KV storage; its bf16
  association shift is ratified and inside the identity gate.
- **Steady step**: one position-independent manual CUDA graph with
  device-resident counters; an adaptive `alpha_fn` falls back to the
  compiled eager controller.
- **Packed projections** (`wire.pack_model_projections`): Q/K/V and
  gate/up packed once per layer, the original Parameters kept as
  disjoint views (a state-dict hook clones only for serialization).
  Shared with the training path.
- Regional `torch.compile(fullgraph)` on the tensor-heavy regions with
  measured wins (prefill, slab steps, two-key merge, fused NLL); the
  position loop stays a small eager state machine, and the
  logits-only head stays eager (compilation never amortizes).

**Compilation is setup, not experiment runtime.** Per-shape
specialization is canon (dynamic shapes were rejected after they
changed forced-choice cells). Every distinct execution shape warms
before any clock starts, largest shape first, so cache growth cannot
recompile cache-dependent graphs; timed or trained loops audit
Dynamo's unique-graph counter and fail on any escape. Final partial
batches are duplicate-row padded to the established shape and the
padding discarded before scoring. Defaults: B=512 (32.7k tok/s at
10.15 GiB on the 4090 at T=512).

**Version pin.** transformers `~=5.15.1` — the wire drives private
layer/cache contracts; the identity gate is the contract test for any
bump. torch 2.8 + flash-attn 2.8.3 per the machine constraints.

## Tasks and evaluation

**Suite (D3).** The discriminating 2×2 {dots, none} × {wire, none}
plus a CoT topline. Serial rows: S5 word problems (NC1-complete,
outside the filler bound) and graph reachability (NL-complete,
likewise, modulo standard separations); parity rides as the empirical
learnability probe (TC0 — the bound is formally silent). Parallel
control: 3SUM (fillers-sufficient per Pfau et al.). Generators are
shared with the sibling experiment in
`transformer_experiments.dot_tasks` — import, don't re-implement.
Money plot: accuracy vs k at eval — inference-time scaling with zero
legible tokens, against the untrained zero line in findings.

**Serialization (D8, D11).** `<t>` := Gemma3's reserved `<unused0>` —
a single token with a tied, untrained embedding row; no vocab resize.
Sequences are composed in id space and never re-tokenized:
`[BOS] prompt-ids | think-span → answer`, the answer token never part
of the eval input. Gemma3 splits `" 1"` but not `"1"`/`"yes"`/`"no"`,
so composition is space-free (the prompt ends with its own cue
token). Task knobs are pinned so every rendered surface form is one
token (reachability at nodes=10 — "10"/"11" split), giving exactly one
token length per (task, k): s5 105, parity 68, reachability 82,
threesum 53, plus k. CoT is one natural space-bearing string,
truncated by actual token count when budgeted.

**Evaluation (D4, D11).** Forced-choice at the final position: argmax
over the task's label tokens (chance = 1/|labels|); soft-everywhere,
full-vocab legality (does the unrestricted argmax land in the label
set) and the gold answer's full-vocab logprob ride along. Instance
sets are paired across conditions. Scope labels are explicit: `none`,
`full-wire`, `dots`, `dots+full-wire`, `dots+think-wire`, `cot`; the
training condition `dots+wire` denotes think scope, and its untrained
null is `dots+think-wire`. Dot sweeps run max-k once and read every
smaller k as a causal prefix — mathematically the same prefix, so the
max-k shape is the canonical sweep definition; both trained arms use
the identical live-row bf16 readout.

**Free running (D14).** The forced k-sweep never measures D4/D12's
actual claim — that the model chooses when to stop dotting — so a
free-running readout rides beside it, *derived* rather than rolled
out: content-free identical dots make a free rollout's prefix equal
to the teacher-forced one (while the model emits `<t>`, the forced
input is exactly what it would have generated), and every position's
readout is causal, so free-running behavior is computed from the same
max-k run. Two arms from the per-position full-vocab logits:

- *Greedy*: halt at the first position whose unrestricted argmax is
  not `<t>`; score the emitted token (accuracy, legality, halt
  position). A trajectory that never halts within the budget scores
  as wrong and is reported as such, never hidden.
- *Sampled, in closed form*: with p_t(·) the position-t softmax, the
  halt-time law is P(halt at t) = (1 − p_t(`<t>`))·∏_{s<t} p_s(`<t>`)
  and the mass on emitting the gold answer as the halting token is
  Σ_t p_t(gold)·∏_{s<t} p_s(`<t>`) — an exact marginalization over
  all rollouts, no Monte Carlo. Soft-everywhere: the halt mass, gold
  mass, legal mass, and expected halt time are all reported as
  masses, not thresholded.

Exact up to the sweep's max k; applies to every dot arm in any scope.
CoT free-running would need true ragged generation and stays
teacher-forced with the frozen topline. Beyond honesty about halting,
this adds a second discriminating axis to the 2×2: without the wire,
per-dot emission logits see the prompt but nothing accumulates
"doneness" across dots, so instance-conditional halting correlated
with serial difficulty is itself wire evidence.

## Training recipe

**Surface (D2).** The entire trainable state: the `<t>` embedding row
(tied — it shapes both how a dot is read and when it is emitted;
synced into the model embedding for plain-forward arms) plus the
paper's gate MLP (LN on concat(source, dest), two hidden GELU layers
at d_model, sigmoid vector output; zero-init output layer with logit
biases so training starts exactly at α=0.1, β=0.9). BF16 throughout;
only Gemma's own RMSNorm fp32 reduction accumulator is wider. Base
frozen — enforced: the loader freezes before any graph exists, and
the first step raises if any base Parameter accumulates a gradient.
The full-adapter rung is documented fallback only.

**Scope and schedule (D5, D12).** Think-first: the prompt is
prefilled once, frozen and detached; the dot span is the only serial
region, forcing wire utilization and keeping the credit chain short.
Full scope follows in a later phase (the wire-alone 2×2 cell is
inherently full-scope and joins then; in full scope, where the gate
concentrates α is an interp readout in its own right). k is sampled
per step from the eval sweep set, homogeneous within a batch; task
sampled per step likewise (mixture is the goal, per-task the
benchmark). Sampling is online and fresh, train seeds disjoint from
the eval seed, addressable by `(seed, step)` for exact resume.

**Forward (D9).** The functional think-scope wire: one parallel
bottom-slab call over the whole dot span (exact — dots' layers
0..dest never see refreshed state and the dot embedding is
input-known), then the serial two-pass slab with the gate MLP at each
mix, under the same snapshot rule as the inference wire. Per slab
layer the cache is piecewise — detached prompt KV, settled one-token
refreshed KVs, the previous column's first-pass frontier — and
nothing in the graph is ever overwritten; the refresh and first-pass
branches share one packed projection and one differentiable
FA2-varlen call (the inference kvcache op has no backward). KV views
are assembled inside non-reentrant checkpointed layer calls with RNG
preservation off, so no prefix copy survives the forward.

**Supervision (D12).** lm_head over the whole emission span: the last
prompt position and every dot target `<t>`; the last dot targets the
answer. Loss `CE(answer) + λ·mean(CE(emission))` (default λ=1) — the
task gradient is k-independent, the dot targets carry no task
content (H2 stays clean), and with sampled k the emission term
teaches a stopping *hazard*, activating D4's halting-by-sampling in
v0. Interior prompt positions stay unsupervised. Full-vocab CE via
one-shape compiled 512-row slabs with a live tied `<t>` column.

**Optimization (D6).** AdamW, weight decay 0, lr 1e-3, linear warmup
(default 100 steps) then flat — no global decay, so later curriculum
stages (if H2 needs them) each reset their own schedule; the prior
experiment's hard-won lesson is that a single decaying schedule
across stage transitions falls apart. Checkpoints are atomic and
RNG/optimizer-complete, resumable by step (`--resume`), default
`data/train/surface.pt`.

**Execution policy (D13).** `--batch` is the effective optimizer
batch (default 512). `checkpoint=auto` from measured 4090 knees:
retain activations while B·k ≤ 2048 (splitting into equal-shape
microbatches up to k=8), at B=512/k=16 retain four evenly spaced
recurrent layers, at k=32 recompute everything. Batch production is a
bounded one-worker deterministic pipeline with pinned nonblocking
transfer. Every structural graph, prompt family, and eval shape warms
before the clock; a step that compiles anything raises. Periodic eval
during training reuses the max-k sweep with a bounded pinned-host LRU
of frozen prompt state.

**Recorded limitation.** The CoT topline stays frozen — nothing is
trainable under a frozen base, so it is the *in-context* legible
reference, not a trace-supervised ceiling (that would need
unfreezing; out of scope).

## Gates

No green gates, no trust. Re-run after touching anything model-facing.

- **G0 identity** (`g0 identity`): the α=0 two-pass engine must match
  the plain HF forward inside the measured bf16 null — thresholds
  mean |Δlogit| < 0.15, top-1 > 0.95, ppl rel < 3e-3, calibrated
  against kernel-tiling self-noise (~5.5e-2 / 0.965) with
  machinery-bug scale ~20× higher; the ppl bound is deliberately
  narrow around the ratified two-key kernel's drift. The retired fp32
  reference proved the algorithm exact (max |Δlogit| 1.2e-4).
- **G0 repro** (`g0 repro`): the paper's untrained perplexity
  reduction at {11,4}, α=0.15 — order −9% PG19, −5% C4, with
  `--pairs` controls ~null. 512-token windows keep every
  sliding-window layer effectively global (deliberate deviation from
  the paper's 1024; validation, not replication).
- **Gradient gate** (`train gate`): the functional BPTT path vs a
  naive reference (per-column sequential bottom, unbatched dual
  passes, visibility re-derived by column index, no checkpointing —
  deliberately independent, after shared bookkeeping once hid a real
  bug) on loss and every surface gradient, in both the zero-init and
  a deterministic perturbed-gate state (zero-init blocks the hidden
  MLP layers); plus an HF cross-check of the span drive with the row
  synced in. Thresholds: loss rel < 5e-3, grad max-rel < 0.1,
  HF mean |Δlogit| < 0.15, top-1 > 0.95 — pinned in the gap between
  measured kernel noise (~2e-2) and semantic-bug scale (O(0.5+)).
- **Runtime audits**: unique-graph counters in timed/trained loops;
  the frozen-base gradient check at the first step; `pytest tests`
  (shape/policy/serialization contracts, Mac-runnable) and Ruff.

## Decision register

Stable anchors; one line each, current resolution only. Dates,
alternatives, and the paths not taken are in the journal.

- **D1** Wire = two-pass, paper-faithful; readout on first-pass
  logits; refresh from dest up; Eq. 1 norm-matched mix.
- **D2** Trainable surface = `<t>` row + α,β gate MLP only; base
  frozen; full adapter is fallback, not the experiment.
- **D3** Task suite = discriminating 2×2 (S5, parity, reachability;
  3SUM control) + CoT topline; money plot accuracy vs k.
- **D4** Halting is sampling: `<t>` is a real vocab token; the head
  chooses dot-vs-answer natively; v0 trains fixed sampled k.
- **D5** Wire scope is a config axis: think (v0) vs full (later);
  full-scope α migration is an interp readout.
- **D6** Fat batches, BPTT serial over positions; flat LR after
  warmup; any curriculum stage resets its schedule.
- **D7** Model = Gemma3-1B PT, pair {11,4} 0-indexed, α=0.15 ramped
  over 10 positions untrained.
- **D8** `<t>` := `<unused0>` (single token, tied row, no resize);
  space-free id-space composition.
- **D9** Training path = functional piecewise cache, differentiable
  FA2, checkpointed layer calls; gradient gate mandatory.
- **D10** One canonical wire path: CUDA + bf16 + FA2 + regional
  compile; no fp32/CPU/sdpa tier; bf16 gates are null-calibrated.
- **D11** Id-space serialization, single-token pins, forced-choice
  eval with legality and gold_lp; explicit scope labels.
- **D12** Training v0: think-first, k sampled per batch,
  emission-span supervision `CE(ans) + λ·mean(CE(emit))`, per-task
  benchmark then mixture; online sampling; bf16 surface.
- **D13** Audited CUDA training: packed shared projections, regional
  compile, FA2-varlen dual branch, `checkpoint=auto` knees, max-k
  sweep eval, warm-everything + zero-compile-in-step, atomic
  RNG-complete resume; whole-step CUDA graphs rejected.

## Open

- Reachability negatives are unconstrained; if shortcut heuristics
  show up in the 2×2, add near-miss negatives to the shared module.
- Adaptive-k evaluation (sampled halting at inference) once a trained
  surface exists — the stopping hazard is already trained in v0.
