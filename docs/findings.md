# Findings

Current experimental data and its interpretation, on the canonical
path at today's code. Superseded measurements and the optimization
history live in [journal.md](journal.md); what the numbers gate and
why lives in [design.md](design.md). All hardware numbers are the
RTX 4090.

## Wire correctness (Gate G0)

**Identity — PASS** (re-witnessed 2026-08-22 on the precision-split
tree, unchanged to print resolution): the α=0 two-pass engine vs the
plain HF forward measures mean |Δlogit| 5.80e-2, top-1 0.9746, ppl
27.0229 → 27.0575 (rel 1.3e-3) — inside the calibrated bf16 null
(kernel-tiling self-noise ~5.5e-2 / 0.965; machinery bugs sit ~20×
higher and scramble top-1). The retired fp32 reference proved the same
algorithm exact at max |Δlogit| 1.2e-4.

**Perplexity repro — PASS** (re-witnessed 2026-08-22, unchanged).
Untrained recirculation, 100×512-token windows, α=0.15, ramp 10:
**PG19 −9.00%, C4 −5.09%** against the paper's −14.4% (at 1024-token
windows) and −3.9%. Controls: {10,3} (the 1-indexed reading) keeps
only ~−1.5%, adjacent {8,7} ~−0.5% — we are on the paper's landscape,
the pair is 0-indexed, and the landscape is sharp. Each dataset scores
in 2.1 s at B=512 with zero timed recompiles.

The untrained wire is real, clearly positive, in the paper's ballpark,
and specific to the characterized layer pair.

## Wire throughput (inference path)

Teacher-forced scoring at T=512: **B=512 — 8.01 s per 100×512
windows, 32.7k tok/s, 10.15 GiB peak**; B=128 — 2.35 s, 27.9k tok/s,
5.35 GiB. Five unique compiled graphs per shape family, zero during
timed evaluation; cold compile+capture ~25 s from an empty Inductor
cache, ~6.5 s with the persistent cache. The full untrained task-grid
preparation (largest shape first, eight prefill lengths) is 49.4 s
cold at B=512 and passes Dynamo's default recompile guard. (The
answer-position readout is now fp32; the teacher-forced NLL path these
numbers time is unchanged.)

## Tasks: untrained baseline (the money plot's zero line)

Full grid 2026-08-20, 4 tasks × k ∈ {1,2,4,8,16,32}, n=512, seed 0
(labels: `none`, `full-wire`, `dots`, `dots+full-wire`, `cot`); the
readout is fp32 (below):

- **Accuracy at chance everywhere.** No untrained condition computes
  anything. Constant-in-k cells (threesum 0.518, reachability ~0.525)
  are the degenerate majority pick: the forced choice lands on one
  fixed label, so acc = that label's empirical split.
- **Legality 0.000 in every non-CoT cell**: the pretrained model
  never spontaneously emits a bare space-free answer token. Claiming
  the answer surface is precisely the trained row's job (D2/D8); this
  is the null it is measured against.
- **CoT toplines split by content**: parity legal 1.000, gold_lp
  −0.78 ≈ ln ½ (fully in answer space; knows it's a bit, not which);
  s5 legal 0.990 but acc 0.148 — *below* chance 0.20, it continues
  the digit pattern instead of reading off the final state;
  reachability acc 0.568 with legal 0.000 — the BFS trace leaks the
  answer (last node = target iff reachable) yet untrained it barely
  helps.
- **The wire's ppl gain shows through the task lens**: gold_lp
  improves under the wire in nearly every matched pair (s5 −8.54 →
  −7.18; reachability −8.52 → −7.92), while untrained dots cost
  gold_lp (settling near −18 for k≥2: untrained `<unused0>` rows push
  the readout off-distribution).
- **The untrained margin is near-linear in the bits.** For parity the
  label-logit margin regresses on the input bits at R² 0.94 (length
  16), corr 0.53 with the number of ones — so short-length "above
  chance" is that linear readout landing on the right side, not
  parity. At length 4 the base model reaches acc 0.57 (k=1) to 0.63
  (k=32); every short-length accuracy must be read as Δ over the per-k
  untrained row, not over 0.5.

Every trained gain is read against this row. The think-scope training
null (`dots+think-wire`) shares the structural facts — untrained row,
legality 0, chance accuracy — and is recorded per-run by each training
job's step-0-equivalent eval.

## Readout precision (D11)

Every readout — forced-choice label logits, the legality argmax, and
gold_lp — is fp32 over the bf16 hidden (`tasks.fp32_logits`, chunked
over the vocabulary). This is load-bearing at the trained head's logit
scale: trained readouts push label logits to ~20, where a bf16 logit's
ulp is 0.125 — wider than the forced-choice margins it decides. A bf16
sweep of a trained surface produced ~8 distinct margins and ~40% exact
ties on 512 instances (`argmax` breaks ties toward label index 0); the
fp32 sweep restores full resolution. Re-scoring past checkpoints
through the fp32 head did not move any reported accuracy (the ties fell
on both sides), but the instrument had been one ulp wide.

## Training path correctness (gradient gate)

**PASS** (re-witnessed 2026-08-22 on the precision-split tree: fp32
surface and optimizer state, 1/d-scaled gate output, fp32 CE, λ=1 per
emission position; B=2, parity len 4, k=3): functional loss 67.254745
= rerun (bitwise-deterministic at this shape) vs reference 67.276848
(rel 3.3e-4 — the loss sums three emission positions at weight 1);
grad max-rel vs the independent reference 3.637e-2 zero-init and
3.491e-2 with the perturbed gate (whose std scales with d so its
pre-sigmoid effect matches the calibration; that state activates the
hidden MLP layers zero-init blocks); span drive vs the plain HF
forward mean |Δlogit| 9.42e-2, top-1 1.0000, compared inside the HF
head's own bf16 rounding. All inside the pinned thresholds (design.md:
Gates). The max-k sweep readout agrees with standalone execution at
100% full-vocab top-1, mean |Δlogit| 0.047–0.056 (the established
tiling null). All 18 project tests pass on jobe; the Mac runs the 8
model-free ones and skips the rest.

## Training path throughput

**Stale as of 2026-08-22** — the peak-memory column predates the fp32
surface and the CE-slab checkpointing (which cut the largest-k peak to
~12.5 GiB); step times are approximately unchanged but unremeasured
under the new recipe. One optimizer step, parity, forward + full-vocab
emission-span loss + backward + AdamW, effective B=512 under the
internal automatic activation-checkpoint policy:

| k | plan | step | throughput | peak (pre-fix) |
|---|---|---|---|---|
| 8 | 2 × B=256, retained | 1.06 s | 483 ex/s | 13.1 GiB |
| 16 | B=512, 4 layers retained | 2.03 s | 253 ex/s | 18.0 GiB |
| 32 | B=512, full recompute | 4.02 s | 127 ex/s | 17.8 GiB |

The 1.86 GiB of gradients the pre-audit baseline accumulated for
frozen base Parameters is structurally zero (checked at step 1).
Periodic evaluation uses the max-k sweep (2.92× over per-k executions)
with the pinned-host prompt-state LRU (a further 1.48× on repeated
sweeps, bitwise-equal hiddens). The batch producer never starves the
GPU (worst observed wait 18 µs behind a 150 ms step). Training warms
every configured k before the clock and compiles nothing inside steps.

## Tasks: training to date (all under a dead gate)

Every training run through 2026-08-22 was made before the precision
fixes below, and post-hoc probing showed the gate MLP had saturated to
a constant, input-independent binary mask within ~500 steps in all of
them (α, β exactly 0/1 per dim, cross-instance std of α exactly
0.0000; ~45% pass-through, ~22% +source, ~9% source-only, ~23% zeroed;
the mechanism is in the journal, 2026-08-22). So the prior results
characterize the reduced model "trained `<t>` row + fixed mask", not
H3's adaptive gate — which has not yet been exercised on a task. Read
them accordingly:

- **parity lengths 16 and 32: no task signal.** Accuracy at chance at
  every k and every training length (n=512), answer CE on the ln-2
  calibrated-ignorance floor; the length-16 margin correlates −0.06
  with parity. Everything *around* the task trained — legality → 1.0,
  a wire-dependence signature (the same surface scored wire-free:
  gold_lp −0.72 → −8..−9, greedy halting 1 → never), and self-halting
  — but the parity function did not emerge, and at length 16 the
  trained readout even *removed* the base model's bit-linear structure
  (R² 0.94 → 0.15) toward calibrated ignorance.
- **parity length 4: a memorization signal, wire-only.** wire-trained
  + wire-run reached acc 0.752 at k=32 (12 of the 16 distinct length-4
  instances; one-sided p ≈ 0.04), monotone in k, the one cell above
  the ln-2 floor (gold_lp −0.62), holding under the fp32 readout. No
  other arm beats the per-k untrained baseline, and there is no length
  transfer (5/6/8 fall to the null) — a length-4 lookup. The mechanism
  still counts: the surface sees the input only through recirculated
  hiddens, so even a lookup certifies input-dependent routing through
  the wire.

Two caveats now on the record: at length 4 the 512 draws are 16
distinct instances, so accuracy is quantized to 1/16 (±0.125 under a
coin) and the n=512 SE never applied; and the halting "at k~4" in
every run was loss-weighting arithmetic (`λ·mean` cancels the fat tail
against the answer term), not learned timing — the per-position `λ·Σ`
form (D15) is what makes the hazard track the training k distribution.

## Precision + gate fixes (2026-08-22): in, gate-verified, unproven on task

The fixes — fp32 surface/optimizer/gate-math/CE/readout over the bf16
base; the gate pre-sigmoid scaled by 1/d; `λ·Σ CE(emit)` at λ=1; and a
per-group gate lr (`--gate-ratio`, default 0.1) — are on the working
tree with G0 identity/repro and the gradient gate re-witnessed above. A
200-step timeline confirms the gate is alive for the first time: at
gate-ratio 1.0 the cross-instance std of α rises 0 → 0.0035 (exactly 0
in every prior run) with the median pre-sigmoid held at the paper init
and a small saturation tail by step 200; at ratio 0.1 it stays 100%
unsaturated but had not engaged by step 200 (slower pacing, same
headroom). No task run has been made under the corrected recipe yet —
checkpoint version 2 does not load the pre-fix surfaces.

## Next

Rerun parity4 under the corrected recipe (live gate, exact hazard,
fp32 readout; ~30 min) — the cheapest test of whether the gate
mattered — then parity16. Length 16 under answer-only supervision also
faces a task-design question the gate fix does not touch: uniform-bit
parity has no partial credit (a scan that XORs 15 of 16 bits is
uncorrelated with the label), so at chance the gradient is noise unless
the features already depend on all bits jointly; options are biased
bits, mixed-length batches that force a scan, or a task with partial
credit. Then the remaining tasks per-task, the mixture (F4), and full
scope (wire-alone arm + α-migration readout).
