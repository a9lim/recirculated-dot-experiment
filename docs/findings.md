# Findings

Current experimental data and its interpretation, on the canonical
path at today's code. Superseded measurements and the optimization
history live in [journal.md](journal.md); what the numbers gate and
why lives in [design.md](design.md). All hardware numbers are the
RTX 4090.

## Wire correctness (Gate G0)

**Identity — PASS** (re-witnessed 2026-08-21 at `e80af1c`): the α=0
two-pass engine vs the plain HF forward measures mean |Δlogit|
5.80e-2, top-1 0.9746, ppl 27.0229 → 27.0575 (rel 1.3e-3) — inside
the calibrated bf16 null (kernel-tiling self-noise ~5.5e-2 / 0.965;
machinery bugs sit ~20× higher and scramble top-1). The retired fp32
reference proved the same algorithm exact at max |Δlogit| 1.2e-4.

**Perplexity repro — PASS.** Untrained recirculation, 100×512-token
windows, α=0.15, ramp 10: **PG19 −9.00%, C4 −5.09%** against the
paper's −14.4% (at 1024-token windows) and −3.9%. Controls from the
first measurement round: {10,3} (the 1-indexed reading) keeps only
~−1.5%, adjacent {8,7} ~−0.5% — we are on the paper's landscape, the
pair is 0-indexed, and the landscape is sharp. Each dataset scores in
2.1 s at B=512 with zero timed recompiles.

Interpretation unchanged since day one: the untrained wire is real,
clearly positive, in the paper's ballpark, and specific to the
characterized layer pair.

## Wire throughput (inference path)

Teacher-forced scoring at T=512: **B=512 — 8.01 s per 100×512
windows, 32.7k tok/s, 10.15 GiB peak**; B=128 — 2.35 s, 27.9k tok/s,
5.35 GiB. Five unique compiled graphs per shape family, zero during
timed evaluation; cold compile+capture ~25 s from an empty Inductor
cache, ~6.5 s with the persistent cache. The full untrained task-grid
preparation (largest shape first, eight prefill lengths) is 49.4 s
cold at B=512 and passes Dynamo's default recompile guard.

## Tasks: untrained baseline (the money plot's zero line)

Full grid 2026-08-20, historical full-scope wire arms (today's
labels: `none`, `full-wire`, `dots`, `dots+full-wire`, `cot`),
4 tasks × k ∈ {1,2,4,8,16,32}, n=512, seed 0:

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
  −7.18; reachability −8.52 → −7.92; dots vs dots+wire likewise at
  small k), while untrained dots cost gold_lp (settling near −18 for
  k≥2: untrained `<unused0>` rows push the readout
  off-distribution).

Every trained gain will be read against this row. The think-scope
training null (`dots+think-wire`) shares the structural facts —
untrained row, legality 0, chance accuracy — and is recorded per-run
by each training job's step-0-equivalent eval.

## Training path correctness (gradient gate)

**PASS** (re-witnessed 2026-08-21 at `e80af1c`, B=2, parity len 4,
k=3): functional loss 29.500000 = rerun (bitwise-deterministic at
this shape) = reference at bf16 print resolution; grad max-rel vs the
independent reference 2.271e-2 zero-init and 5.176e-2 with the
perturbed gate (the second state activates the hidden MLP layers that
zero-init blocks); span drive vs the plain HF forward mean |Δlogit|
9.42e-2, top-1 1.0000. All inside the pinned thresholds (design.md:
Gates), which sit in the gap between measured kernel noise and the
O(0.5+) scale a deliberately broken visibility set produces. The
max-k sweep readout agrees with standalone execution at 100%
full-vocab top-1, mean |Δlogit| 0.047–0.056 (the established tiling
null). All 10 project tests pass on jobe; the Mac runs the 4
model-free ones and skips the rest.

## Training path throughput

One optimizer step, parity, forward + full-vocab emission-span loss +
backward + AdamW, effective B=512 under `checkpoint=auto`:

| k | plan | step | throughput | peak |
|---|---|---|---|---|
| 8 | 2 × B=256, retained | 1.06 s | 483 ex/s | 13.1 GiB |
| 16 | B=512, 4 layers retained | 2.03 s | 253 ex/s | 18.0 GiB |
| 32 | B=512, full recompute | 4.02 s | 127 ex/s | 17.8 GiB |

Against the pre-audit baseline (B=64, k=4): 290.9 → 98.7 ms/step
(**2.95×**), peak 6.38 → 3.54 GiB, and the 1.86 GiB of gradients the
baseline accumulated for frozen base Parameters is now structurally
zero (checked at step 1). Periodic evaluation uses the max-k sweep
(2.92× over per-k executions) with the pinned-host prompt-state LRU
(a further 1.48× on repeated sweeps, bitwise-equal hiddens). The
batch producer never starves the GPU (worst observed wait 18 µs
behind a 150 ms step). Training warms every configured k before the
clock (38 graphs for dots+wire, 16 for dots, full default k set) and
compiles nothing inside steps — the audit that enforces this caught
two real warm-coverage holes at launch (journal 2026-08-21).

## Training smoke (pre-scale sanity)

Parity, dots+wire (think scope), 150 steps, B=64, k ∈ {1,2,4}: loss
29.4 → 1.7; emission CE 19.5 → ~0.01 and eval legality 0.000 → 1.00 —
the trained row fully claims the answer surface the untrained
baseline lacked; answer CE settles at ~ln 2, the
calibrated-but-ignorant parity floor, with accuracy still at chance
as expected at this scale. The surface trains; whether the wire lets
it *compute* is the first real run's question (H2).

## Next

Real runs: parity dots+wire vs the dots-alone control (the H2
learnability probe), then the remaining tasks per-task, then the
mixture (F4), then full scope (brings the wire-alone arm and the
α-migration readout).
