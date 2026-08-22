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

**PASS** (re-witnessed 2026-08-21 at `3572c0a` under D15's λ=0.125,
B=2, parity len 4, k=3): functional loss 12.875000 = rerun
(bitwise-deterministic at this shape) = reference at bf16 print
resolution; grad max-rel vs the independent reference 2.698e-2
zero-init (2.271e-2 at λ=1) and 5.176e-2 with the perturbed gate (the
second state activates the hidden MLP layers that zero-init blocks);
span drive vs the plain HF forward mean |Δlogit| 9.42e-2, top-1
1.0000. All inside the pinned thresholds (design.md:
Gates), which sit in the gap between measured kernel noise and the
O(0.5+) scale a deliberately broken visibility set produces. The
max-k sweep readout agrees with standalone execution at 100%
full-vocab top-1, mean |Δlogit| 0.047–0.056 (the established tiling
null). All 15 project tests pass on jobe; the Mac runs the 7
model-free ones and skips the rest.

## Training path throughput

One optimizer step, parity, forward + full-vocab emission-span loss +
backward + AdamW, effective B=512 under the internal automatic
activation-checkpoint policy:

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

## First H2 probe: parity at v0 scale (2026-08-21)

Both arms trained at defaults from `8020e51`: parity (length 32),
think scope, 2000 steps × effective B=512 (~1M examples), k sampled
uniformly from {1,2,4,8,16,32}, lr 1e-3, λ=1; wire arm ~50 min, dots
~22 min, clean audits throughout. In-run eval sweeps: accuracy at
chance every 500 steps in both arms. Post-hoc scoring of the final
surfaces (n=512, eval seed 0; `logs/posthoc-parity.log` on jobe) —
sweep with gold_lp plus the D14 free-running readout, against the
untrained think-scope null and a transfer cell (wire-trained surface
executed *without* the wire):

| surface / arm | acc | legal (k≥4) | gold_lp (k≥4) | free-running (greedy) |
|---|---|---|---|---|
| untrained / either | chance | 0.000 | −9..−11 | halts at k=1, 0% legal |
| **wire / think** | chance | 1.000 | **−0.72..−0.79** | halt 1.00, k~4, 100% legal |
| **wire / dots** (transfer) | chance | 0.000 | −3.7..−9.1 | never halts |
| dots / dots | chance | 1.000 | −0.72..−0.79 | halt 1.00, k~4, 100% legal |

Readings:

1. **H2 signal absent at this scale.** Accuracy is chance in every
   trained cell; answer CE sits on the ln 2 calibrated-ignorance
   floor (gold_lp ≈ −0.69). The surfaces learned everything *about*
   answering — emission, calibration, stopping — and nothing about
   parity.
2. **The learned surface is wire-dependent** — the probe's genuinely
   new fact. With the same wire-trained surface, gold_lp is identical
   across wire/no-wire arms at k≤2 (−1.570/−1.586 — structurally
   forced: the first refreshed column only becomes visible to a
   readout at t=3, so this equality doubles as a live semantics
   cross-check), then collapses without the wire at k≥4: −0.72 →
   −3.7..−9.1, legality 1.0 → 0.0, greedy halting 100% → never. The
   gate and row learned to keep the readout calibrated *through
   recirculated state*, not through the row alone. Dependence, not
   yet superiority: the dots-trained arm reaches the same floor
   wire-free.
3. **Halting works (D4/D14).** Both trained arms self-halt greedily
   at k~4 with 100% legality (soft E[k|halt] ≈ 2.0–2.3), against an
   untrained null that "halts" immediately on an illegal token.
   Trained on uniform k, the learned hazard is front-loaded — the
   forced-sweep legality zeros at k≤2 are this hazard seen from the
   other side.

Interpretation: the machinery is green end to end — gradients flow,
the stopping hazard trains, the gate routes wire state — but the task
computation did not emerge in 2000×512 at serial depth 32, which is
the maximally hard depth for the k≤32 budget. Next probes, in order
of information per GPU-hour: task-difficulty scaling (shorter parity
lengths — a curriculum over difficulty, not trace supervision, so H3
stays intact) and longer optimization.

## Parity length 4: first learned signal (2026-08-21)

Difficulty-scaling probe: `--knobs length=4`, standard v0 recipe
(flat LR, uniform k over {1..32}), both arms, 2000×512, snapshots
every 500. Post-hoc at n=512 seed 0 (`logs/posthoc-parity4.log`,
`logs/posthoc-parity4-transfer.log` on jobe). Three results:

1. **The untrained baseline is not at chance at length 4.** Frozen
   Gemma, forced choice: acc 0.568 at k=1 up to 0.631 at k=32
   (SE ≈ 0.022) — with legality 0.00 and gold_lp −7..−10. The base
   model carries parity-of-4-bits signal in its label-logit margins
   while its output distribution is garbage. Every len-4 accuracy
   must be read as Δ over the per-k untrained row, not over 0.5.
2. **Wire-trained + wire-run beats that baseline; nothing else
   does.** wire/think at step 2000: acc 0.568/0.568/0.629/0.752 at
   k=4/8/16/32 (monotone in k), legality 1.00, and gold_lp −0.62 at
   k=32 — the only cell above the ln 2 calibrated-ignorance floor
   (~54% gold mass; argmax amplifies to 0.75). dots/dots matches the
   *untrained* accuracy pattern with legality dressed on top and
   gold_lp never above floor. Transfers fail both directions:
   wire-trained scored wire-free collapses to the null (gold_lp
   −3..−6); dots-trained given the wire *degrades* at high k.
3. **The signal emerged entirely in steps 1500→2000** (k=32
   trajectory: 0.486 → 0.492 → 0.492 → 0.752) — a late phase change
   after ~1500 steps in the calibration/legality/hazard basin, cut
   off mid-climb. Undertraining is confirmed, not just suspected.

**No length transfer.** The step-2000 wire surface scored at lengths
5/6/8: accuracy at the untrained null, gold_lp back *below* floor
(−0.8..−2.1 — confidently miscalibrated off-distribution), legality
partially collapsed. With 2⁴ = 16 distinct instances (B=512 sees the
whole space ~32× per step — effectively full-batch GD), the learned
solution is a length-4-specific lookup. The mechanism is still the
point: the surface never sees the input except through recirculated
hiddens, so even memorization certifies input-dependent routing
through the wire — H2's mechanism at work, without H2's algorithm.

Free-running texture: the trained wire arm greedily halts at k~3.8
(100% legal, free acc 0.568 ≈ its forced k=4 accuracy) — it halts
well before its own competence peak at k=32. The hazard trained on
uniform k learned "halt around 4"; the capability lives at 32. That
mismatch motivated D15's fat-tailed training k.

Watch-items: legality at small k is *bistable* over training (both
arms oscillate 0 ↔ 1.0 across evals under flat lr 1e-3) — any single
eval's legality is a snapshot of that oscillation; and one stray
transfer cell (dots at len5/k32: 0.637) has no k- or length-trend
and awaits replication before it means anything.

## Next

Rerun parity4 under D15 (cosine to floor, train-k {2..32} with
P∝k, longer budget) — does the phase change move earlier, stabilize,
saturate; does transfer ever emerge (grokking watch). Then len 8/16
with proper budgets (2⁸/2¹⁶ instances progressively close the
memorization escape), the remaining tasks per-task, the mixture
(F4), and full scope (wire-alone arm + α-migration readout).
