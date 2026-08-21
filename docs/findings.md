# Findings

## Train: gradient gate + smoke (2026-08-21)

D12 landed (think-first, sampled k, emission-span supervision,
per-task-then-mixture; forks a9-ratified). The training path is the
functional think-scope wire: frozen parallel prompt prefill (detached
per-layer KV), one parallel bottom-slab call over the whole dot span
(exact — dots' layers 0..dest never see refreshed state and the dot
embedding is input-known), then the serial two-pass slab with the
gate MLP at each mix, K/V cat'd inside checkpointed layer calls,
attention via differentiable `flash_attn_func` (D9 amendment: the
inference kvcache op has no backward).

**Gradient gate: PASS** (B=2, parity len 4, k=3). Functional path
rerun is *bitwise* identical (deterministic at this shape); vs the
naive reference (per-column sequential bottom, unbatched dual passes,
visibility re-derived by column index, no checkpointing): loss
rel-diff 6.9e-4, grad max-rel 1.67e-2 — bf16 kernel-order noise;
semantic bugs sit O(0.5+). During construction the gate's design
caught a real one: an early draft merged the refresh into the visible
set before the first pass ran (same-snapshot violation) — in both
paths identically, which is why the reference derives visibility
independently by index. HF cross-check: the span drive with the row
synced into the tied embedding matches the plain forward at
mean|dlogit| 8.2e-2, top-1 1.0000 (G0-null scale).

**Training smoke** (parity, dots+wire, 150 steps, B=64, k∈{1,2,4}):
loss 29.4 → 1.7; emission CE 19.5 → ~0.01 and eval legality
0.000 → 1.00 — the trained row fully claims the answer surface that
the untrained baseline lacked entirely; answer CE settles at ~ln 2
(calibrated-but-ignorant parity floor), accuracy still chance as
expected at this scale. ~0.33 s/step eager; real runs want the D9
compile pass.

## G0 — wire implementation correctness (2026-08-20)

**Identity check: PASS.** Two-pass engine at α=0 vs plain HF forward,
Gemma3 1B PT, fp32 on jobe: max |Δlogit| 1.35e-4, mean 4.8e-6,
perplexity identical to 4 decimals (27.0329). The cache-overwrite /
rope / manual-layer-drive machinery is a verified no-op at α=0.
(`python -m recirculated_dot_experiment.g0 identity`)

**Perplexity repro: PASS.** Untrained recirculation, Gemma3 1B PT
bf16, 100 windows × 512 tokens, α=0.15, convex norm-ratio mix,
10-step ramp (`g0 repro --windows 100 --pairs "10,3;8,7"`):

| pair (0-indexed) | PG19 | C4 |
|---|---|---|
| baseline ppl | 23.03 | 17.99 |
| **{11,4}** (paper) | **−8.81%** | **−5.05%** |
| {10,3} (1-indexed reading) | −1.35% | −1.62% |
| {8,7} (adjacent control) | −0.44% | −0.52% |

Reads: (i) the wire works untrained, clearly and in the paper's
ballpark — PG19 −8.8% vs their −14.4% at 1024-token windows, C4
−5.1% vs their −3.9% (we keep only ≥512-token C4 docs, and longer
docs benefit more, consistent with their lag analysis); (ii) the
paper's {11,4} is 0-indexed — empirically resolved, and the landscape
is sharper than the smooth-heatmap assumption suggested ({10,3} loses
most of the effect); (iii) the adjacent control is ~null, so we are
seeing their landscape, not a generic perturbation artifact.
Engine cost: ~73 s per 100×512 windows (bf16, batch 16, 4090).

**Gate G0 is passed.** The wire is trustworthy; tasks and training
build on it.

## Wire optimization pass (2026-08-20)

73s → 5s per 100×512-token windows (14.6×), perplexity invariant at
every step (20.99–21.02, bf16 noise; identity gate exact throughout):

1. **Pass A** — layers 0..dest never see refreshed state (refresh only
   writes dest+1..top), so the bottom slab prefills in parallel for
   the whole sequence, exactly. Serial work drops to the 21-layer slab.
2. **Dual-pass batching** — first pass of column t and second pass of
   column t−1 share a cache snapshot (the paper's concurrency), so
   they run as one [2B] call: 48 → 21 layer calls per step. 73→38s.
3. **Deferred chunked readout** — per-step fp32 softmax over the 262k
   vocab replaced by end-of-run chunked lm_head/NLL (~1.5 GB bound).
4. **Batch scaling** — the loop is launch-bound, so batch rides free:
   38s (B=16) → 10s (B=64) → 5s (B=100). Same-code B=16 is 33s.
5. **Two-lane DualCache** — per-layer lanes in one [2B] buffer; views
   are zero-copy, data movement is four one-slot writes + one commit
   per layer per step (replaces two full-prefix cats). Modest now
   (launch-bound), but it is the shape-stability groundwork for
   torch.compile.

Codex consults (sol tier, both ran their own reproductions on jobe):

- *Correctness (adversarial, fp32 sequential reference, agreement
  2.76e-7 across edge configs)*: snapshot/commit ordering, dual-pass
  visibility, rope alignment, GQA (kv_heads 1/2/4), boundaries, ramp
  association, NLL chunking — all held. Two real catches, both fixed:
  pass A is causal only under sdpa (eager attends bidirectionally →
  engine now asserts sdpa), and `final_logit_softcapping` was bypassed
  (None on Gemma3-1B, latent for other variants → now applied).
  transformers pinned `~=5.15.1` (the wire drives private layer/cache
  contracts); the identity gate is the contract test for bumps.
- *Performance*: compiled 21-layer slab measured 11.5 → 2.4 ms (4.8×)
  with fixed shapes — torch.compile with bucketed KV lengths is the
  documented next lever, deferred until training throughput actually
  binds. Not worth it at our scale: streams, FlashAttention, custom
  fused kernels, quantization, full-length eager attention.

## Push-it-further pass (2026-08-20, second round)

**GQA under masks — the real 4× kernel win.** FlashAttention benched
4× faster than sdpa at the serial shape (117 vs 464 µs; q_len=1,
kv=512, 200 rows) — but the whole gap was native GQA: HF's sdpa path
repeat_kv's 1 KV head to 4 query heads whenever a mask is present
(materialized copy + 4× traffic; attention here is memory-bound). At
the causal pass-A shape FA2 and sdpa tie exactly. Fix: `_wire_sdpa`
registered via HF's AttentionInterface — **stride-0 expand views**
into the same fused kernel: 464 → 119 µs, bitwise-identical output,
zero copies. Two traps documented from measurement: `enable_gqa=True`
with a dense mask silently falls back to the math backend (3354 µs),
and FA2 itself cannot express the per-lane dual-pass mask. FA2 is
empirically closed: nothing left for it to win.

**torch.compile — implemented, gate-exact, measured slower, parked.**
Slab step compiles fullgraph with zero recompiles and passes the
identity gate (1.22e-4 fp32). But the compiled graph does 4.5× the
GPU work (37.4 vs 8.3 ms/step at B=100): inductor materializes the
stride-0 broadcast feeding the extern attention kernel (un-doing the
GQA win), runs static full-T attention, and fails to re-inplace the
lane-buffer mutations (functionalization copy tax). The mutation-
heavy inference cache is structurally compile-hostile. `--compile`
stays as a documented experiment flag; compile belongs to the
training path, whose D9 functional cache is mutation-free (a pure
graph) by design.

**Where the wire landed.** B=100 eager: ~10 ms/step wall vs 8.3 ms
GPU work — ~83% GPU-bound with efficient kernels; weight traffic
alone floors the slab near ~2 ms/step, so remaining headroom is ≤2×
at high effort/risk. Final canonical numbers: PG19 −8.75%, C4 −5.11%,
5 s per 100×512 windows each, identity gate exact. Journey: 73 → 5 s
(14.6×), semantics-preserving at every step.

## Round 3: FA2 kvcache × compile (2026-08-20, a9's call)

Round 2's "FA2 cannot express the dual-pass mask" was true only of
the HF integration path. Correction: the dual-pass "mask" is really
*per-lane prefix lengths*, and `flash_attn_with_kvcache` expresses
exactly that via per-row `cache_seqlens` — lane 0 appends first-pass
at slot t, lane 1 overwrites its recompute at t-1, both in-kernel.
Measured 43 vs 119 µs against the fixed masked sdpa (it only reads
the live prefix; no mask tensor exists), output matching to bf16
noise. The `fa2` backend (`DualCacheFA2`, stacked lanes, one fused
lane1→lane0 commit per step at step end) rides HF's kwargs
passthrough — still no layer forking.

FA2 and compile then compose exactly as hoped — better than
orthogonal: the kvcache interface removes both structures that made
inductor lose in round 2 (no mask-driven broadcast to materialize;
cache writes hidden inside an opaque op; the commit lives outside
the graph). One wrinkle: flash-attn 2.8 exposes `fwd_kvcache` as a
raw PyCapsule that dynamo cannot trace — wrapped as a torch custom
op with `mutates_args=("k_cache","v_cache")` and a fake kernel, after
which the slab compiles fullgraph. `reduce-overhead` adds nothing
(cudagraphs decline the mutating custom op and fall back).

**Landing: 3 s per 100×512 windows** at B=100 (~5.9 ms/step, at the
estimated weight-traffic floor), ppl invariant across all backends ×
compile combinations (21.001–21.016 PG19; identity gate exact on the
sdpa fp32 path). Journey: 73 → 38 → 5 → **3 s (24×)**. Defaults:
`attn_backend="auto"` (fa2 when available + half precision, else
sdpa; the Mac falls back cleanly), compile opt-in via `--compile`.
The custom op and stacked-lane design carry over to the training
path (D9).

## Canonicalization (2026-08-20, D10)

Compiled-FA2 ratified as the only path; sdpa dual-pass, mask
machinery, and all flags stripped (git history keeps them). Repro on
the canonical path: PG19 −8.81%, C4 −5.11%, 5 s/3 s per 100×512
windows (first dataset amortizes compile).

The fp32 identity gate retired with the sdpa path (FA2 is
half-precision-only). Rather than guessing a bf16 tolerance, G0 now
calibrates against a measured **null**: the plain HF forward compared
with *itself* under a kernel-tiling change (batch-4 vs row-by-row)
disagrees at mean |Δlogit| 5.45e-2, top-1 0.9648 — and the α=0 engine
vs the plain forward sits *inside* that null (5.26e-2, 0.9824, ppl
rel 7e-4). The wire adds zero divergence beyond intrinsic bf16
kernel-order noise; thresholds (mean < 0.15, top-1 > 0.95, ppl rel
< 2e-3) live in the ~20× gap between the null and machinery-bug
scale. Method worth keeping: when a gate loses its exact reference,
measure the self-noise null before pinning tolerances.

**FA2-everywhere addendum (same day).** Stock sdpa retired from the
fallback tier too: pass A and plain forwards defer to HF's
flash_attention_2 interface, with the paired mask-interface
registration (`AttentionMaskInterface`) so mask building matches.
One wrinkle: HF's delegate resolves the flash package by reading
`module.config._attn_implementation`, which says "wire_attention" —
the name is scoped to "flash_attention_2" strictly inside the
delegated call. FA2 null: mean 5.48e-2, top-1 0.9785 (same floor as
sdpa's); identity PASS inside it; repro PG19 −8.93%, C4 −5.03%
(baselines shifted ~1e-3 relative with the kernel change, as
expected), 5 s/3 s unchanged. One attention library end to end.

## Compile-everything and packed-projection pass (2026-08-20)

Live RTX 4090 profiling at B=100, T=512 split the 3.08 s warm wire
into 0.18 s pass A/setup, 2.40 s serial slab, and 0.50 s readout.
Matrix multiplies were 53% of CUDA time and FA2 kvcache attention 28%.

The landed path fullgraph-compiles pass A, the first slab step, the
recurrent norm-ratio mix plus slab, and both readout modes. Q/K/V and
gate/up are packed once from frozen weights; per-layer GEMMs fall from
seven to four. RoPE, dual-lane RoPE, sequence-length schedules, KV, and
top-hidden scratch are reused by shape. The only eager region left is
the position state machine and its exact end-of-step cache commit.
Python alpha/beta values are normalized to tensors; the default ramp
is device-resident, preventing value-specialized recompilation.

At B=128, T=512 the final warm time is **2.84 s** (23.1k token/s),
versus 3.52 s (18.6k token/s) before this pass: **1.24x throughput**.
The phase split is 0.127 s prefill, 2.442 s serial slab, and 0.268 s
commit plus readout. Persistent packed weights raise post-compile peak
allocation to 9.36 GiB, still leaving ample 4090 headroom. Compile-cold
time is 13.65 s and is amortized by repeated batches.

Gates: bf16 identity PASS (mean |dlogit| 5.81e-2, top-1 0.9902,
plain/engine ppl 27.0229/27.0260); full 100x512 reproduction at the new
default batch gives PG19 **-9.00%** and C4 **-5.11%**. `max-autotune`,
FA2 `num_splits`, and larger readout chunks remained measured nulls.

## Higher-effort CUDA kernel pass (2026-08-20)

Each candidate was first microbenchmarked at Gemma3-1B's live recurrent
shape and then admitted only by paired B=128, T=512 whole-wire timing on
the RTX 4090. Quantization and multi-device parallelism were excluded.

**Manual steady-step CUDA graph — landed.** A single capture uses
device-resident current/previous indices and stable hidden, RoPE,
sequence-length, recurrent-state, KV, and top-state buffers. Position
selection, state threading, shared-cache writes, and counter increments
all happen inside the graph; the host loop issues one replay per steady
position. The first call warms and captures the fixed-buffer signature
after the unique first step, restores that step's side state, and uses
the graph immediately; it does not also compile the default eager
signature. Warm latency fell from 2.836–2.838 s to 2.776–2.784 s (about
2%); arbitrary `alpha_fn` calls correctly retain the eager controller.

**Exact shared-prefix dual-branch attention — landed.** At step `t`, the
branches share refreshed KV `0..t-2` and differ only in their tail: the
first pass adds `{first-pass(t-1), first-pass(t)}`, while the refresh adds
`{refresh(t-1)}`. The new cache stores the prefix once plus one side KV.
FA2 reads it through interleaved duplicate `cache_batch_idx` entries; a
separately compiled closed-form two-key softmax and fp32 LSE merge
reconstruct the two attentions before the refresh write. Across all 511
steady positions, one layer's attention sweep fell from 35.04 to 25.94 ms
(1.35x). Whole-wire warm latency with the CUDA graph is **2.346–2.359 s,
27.8–27.9k token/s**, a **1.21x throughput gain** over the 2.84 s prior
path. Warm peak allocation falls from 9.36 to **8.20 GiB** because the
slab KV history is no longer duplicated.

The closed-form two-key specialization was explicitly ratified after an
adversarial gate caught its changed bf16 association: mean logit error
and top-1 stayed at the calibrated kernel-order floor, while identity PPL
relative drift moved from 1.32e-3 to 2.59e-3. The PPL threshold therefore
changes narrowly from 2e-3 to **3e-3**; the independent mean-error and
top-1 guards are unchanged.

The other requested routes failed the whole-path gate:

- **FA2-fused RoPE:** the isolated recurrent attention improved 3.23x at
  position 1, 2.02x at 255, and 1.17x at 511, and saved about 0.25 GiB;
  nevertheless the full wire regressed to 2.852 s (about 0.5%). Removed.
- **RMSNorm-linear/residual fusion:** commuting the frozen RMS scale into
  QKV and gate/up weights made their representative kernels 2–3% slower
  and introduced larger bf16 rounding drift. NVIDIA Transformer Engine
  had no wheel for the pinned torch 2.8/cu128/Python 3.12 stack and its
  source extension failed on an unavailable CUDA header; every temporary
  package was removed and `uv pip check` returned clean. The residual
  norm chains are already fused by Inductor. Removed.
- **Readout overlap:** issuing each completed NLL chunk on a second CUDA
  stream produced 2.775–2.783 s, indistinguishable from graph-only, with
  bitwise-identical NLL. The LM-head GEMM contends with rather than hides
  behind the recurrent slab. Removed.

**Compile lifecycle and item 8 — landed.** The default path now compiles
only the fixed-buffer recurrent signature and uses its CUDA graph on the
first call; it no longer compiles a redundant eager signature first.
Readout scratch pads the last chunk and discards its extra results, so the
B=128,T=512 NLL path needs **five unique graphs**, down from six. A truly
empty Inductor cache takes **24.66 s** for compile+capture plus its first
execution; the ordinary persistent-cache run takes about **6.5–6.7 s**.
Both are reported outside experiment timing. G0 also pads only the final
execution batch with duplicate rows, excludes them from scoring, reuses
one engine across same-shaped datasets, and raises if Dynamo's unique
graph count changes while the evaluation clock is running. The final
PG19 and C4 runs each took **2.10 s with zero recompiles**.

Final gates on the shared-prefix algorithm: bf16 identity PASS (mean
|dlogit| 5.86e-2, top-1 0.9746, plain/engine ppl 27.0229/26.9528), exact
repeatability under graph replay, T=1..4 in bf16 and fp16, and arbitrary
alpha fallback all finite. The 100x512 reproduction remains PG19
**-9.00%** and C4 **-5.09%**. The accepted identity shift is below the
recalibrated G0 boundary and remains jointly constrained by all three
metrics.

## Tasks: serialization + untrained baseline 2×2 (2026-08-20)

D11 landed: id-space composition over the shared generators, pinned
single-token surfaces (reachability at nodes=10 — "10"/"11" split into
digit pairs), forced-choice readout via the engine's new
`answer_logits` (identity gate re-passed at the recorded numbers after
the `_run` extraction — pure code motion). At this stage one Dynamo
guardrail was raised: `recompile_limit` 8 → 256. The then-standalone task
grid visited ~30 distinct lengths, and the default limit hard-aborted under
fullgraph at the 9th shape (hit mid-grid at parity T=69). The later max-k
sweep and hindsight pass supersede this global change.

Full untrained grid (historical full-scope labels: 4 tasks × {none,
wire, dots, dots+wire} + cot,
k ∈ {1,2,4,8,16,32}, n=512, B=512, ~8 min wall including compiles):

- **Accuracy at chance everywhere** — no untrained condition computes
  anything. Constant-in-k cells (threesum 0.518, reachability ~0.525)
  are the degenerate majority pick: the forced choice lands on one
  fixed label, so acc = that label's empirical split.
- **Legality 0.000 in every non-CoT cell**: the pretrained model never
  spontaneously emits a bare space-free answer token. Putting mass on
  the answer surface is precisely the trained embedding row's job
  (D2/D8); this is the null it is measured against.
- **CoT toplines split by content**: parity legal 1.000 / gold_lp
  −0.78 (≈ln ½ — fully in answer space, knows it's a bit, not which);
  s5 legal 0.990 / gold_lp −2.42 but acc 0.148 *below* chance 0.20 —
  it continues the digit pattern rather than reading off the final
  state; reachability 0.568 / legal 0 — the BFS trace leaks the answer
  (last node = target iff reachable) but untrained it barely helps.
- **The wire's ppl gain shows through the task lens**: gold_lp
  improves under wire in nearly every matched pair (s5 none −8.54 →
  wire −7.18; reachability −8.52 → −7.92; dots vs dots+wire likewise
  at small k), while untrained dots cost gold_lp (k=1 mild, k≥2
  settling near −18: untrained `<unused0>` rows push the readout
  off-distribution).

This is the pre-training reference row for the money plot (accuracy
vs k): every trained gain has its null here.

## Memory and throughput scaling pass (2026-08-20)

Four compatible changes now define the canonical memory layout and batch
policy. Pass A runs in fixed B=64 chunks, including a row-index-padded
final chunk with the same dispatch keys, and copies each output directly
into pooled destination storage. The original projection Parameters are
disjoint views of the packed QKV and gate/up tensors, retiring **0.857
GiB** of duplicate storage while leaving plain-model logits bit-identical.
A state-dict post-hook clones only those views during export; strict reload
preserved the live aliases and full safetensors `save_pretrained` passed.

The recurrent step no longer retains expanded dual-lane RoPE. It gathers
only positions `[t,t-1]` from the compact tables inside the compiled graph
and broadcasts them over branch lanes. The persistent RoPE footprint is
**0.001 GiB**, replacing about 0.25 GiB at B=128 and 1.00 GiB at B=512.
This is deliberately distinct from pushing RoPE into FA2 itself, whose
earlier whole-wire timing failed.

On the RTX 4090, the final B=128,T=512 path measured **2.350–2.351 s**
(27.87–27.89k token/s), unchanged in speed from the prior path, while peak
allocation fell from 8.20 to **5.35 GiB**. The new default B=512,T=512
measured **8.008–8.019 s**, **32.69–32.74k token/s** (about 17% more
throughput than B=128), at **10.15 GiB** peak. Both shapes use five unique
compiled graphs and trigger zero graphs during timed replay; a nonmultiple
B=100 reproduction also stays at five through the uniform prefill gather.

Final gates: bf16 identity PASS (mean |dlogit| 5.80e-2, top-1 0.9746,
plain/engine ppl 27.0229/27.0575); deterministic graph replay; T=1..4
finite in bf16 and fp16; adaptive-alpha fallback finite; PG19 **-9.00%**
and C4 **-5.09%** over 100x512 windows. Compilation/capture remained
outside experiment timing, and both dataset evaluations took 2.13 s with
zero recompiles.

## Training/task correctness and CUDA pass (2026-08-21)

The Claude training/task implementation received a full contract audit
before optimization. Eight issues were corrected together: the 340 base
Parameters are frozen before graph construction; the head is a flattened
2-D GEMM; D12's untrained think-scope null is explicit and the older
full-scope arm is separately named; CoT is tokenized as one natural
space-bearing string; Torch/CUDA seeding and optimizer/RNG-complete atomic
resume replaced surface-only checkpoints; a perturbed-state gate exercises
the MLP hidden layers that zero output initialization normally blocks; both
trained arms use one BF16 live-row readout; and project-owned tests plus
Ruff are green. Surface row, gate, layer I/O, attention, and readout are
BF16; only Gemma's specified RMSNorm reduction accumulator remains FP32
internally before returning BF16.

Seven compatible performance changes landed:

1. `checkpoint=auto` retains activations through `B*k <= 2048` and
   recomputes above it.
2. CE flattens to a 512-row full-vocab slab, compiles fullgraph, and pads
   its tail to one shape.
3. Evaluation performs one max-k causal run and reads every requested
   prefix position.
4. Frozen prompt, parallel span, and recurrent layer math use common
   packed, tensor-only regional compilation.
5. Refresh and first-pass branches share one projection and one
   differentiable FA2-varlen attention call.
6. Training reuses the inference wire's packed projections; original
   Parameters are storage views, not duplicate allocations.
7. A bounded deterministic producer overlaps generation/tokenization
   with pinned, nonblocking host-to-device transfer.

Paired RTX 4090 evidence against commit `f4b4325`, parity B=64,k=4,
five warm full train steps (forward + full-vocab loss + backward + AdamW):
median **290.91 → 98.72 ms (2.95×)**, peak **6.38 → 3.54 GiB
(−44.5%)**. The baseline accumulated gradients for all 340 base
Parameters (**1.862 GiB**); the new path accumulated zero. Isolated
512-row head forward+backward improved **8.42 → 6.63 ms (1.27×)**.

The adaptive checkpoint tradeoff at B=256,k=8 was **531.0 ms / 13.05
GiB** without recomputation versus **619.5 ms / 5.52 GiB** with it; auto
therefore takes the 16.7% faster retained path while leaving more than
10 GiB headroom. At the then-default stress point B=256,k=32, the
piecewise checkpointed path measured **2.32 s / 10.17 GiB**. A rejected
intermediate that persisted assembled prefixes reached 22.76 GiB; no
such cache remains.

Max-k evaluation over k={1,2,4,8}, B=64 measured **0.081 s versus 0.237
s (2.92×)** for four standalone executions. Sweep-vs-standalone mean
logit differences were 0.047–0.056 for smaller k, inside the established
BF16 kernel-tiling null; full-vocab top-1 agreed 100%. Natural S5 CoT now
decodes as `3 3 2 ...` (15 actual tokens for eight states), not `332...`.
For the heaviest CPU generator (threesum B=256), synchronous preparation
was 69.4 ms median; after the first fill, the bounded producer's maximum
observed wait was 18 microseconds behind a 150 ms simulated GPU step.

Final gates: full zero-init and perturbed-gate gradient checks PASS
(functional/reference losses equal at BF16 resolution; max relative
gradient differences 2.27e-2 and 5.18e-2), span/HF top-1 1.000 with mean
|dlogit| 9.42e-2, inference G0 identity PASS at its canonical 5.80e-2 /
0.9746 / 27.0229→27.0575 measurements, seven project tests on Jobe, seven
shared task tests on both machines, and zero train-step graphs after the
15-graph structural warmup. Compile/warmup is reported separately and
excluded from the experiment clock.

## Training/task throughput follow-up (2026-08-21)

Five remaining opportunities were timed on the RTX 4090. Four produced a
useful canonical change; manual whole-step CUDA graphs did not.

1. **K-aware effective batches.** Training now defaults to effective B=512.
   For k=8, auto mode runs two retained B=256 microbatches: the final
   committed candidate measured **1.0604 s / 482.8 examples/s / 13.08 GiB**.
   One checkpointed B=512 step was **1.1789 s, 434.3 examples/s, 9.07 GiB**,
   so the equal-shape accumulation is 11.1%
   faster. For k=16 and k=32, the larger checkpointed GEMMs win: B=512
   measured **2.0674 s / 247.7 examples/s / 11.58 GiB** and, on the final
   candidate, **4.0236 s / 127.2 examples/s / 17.79 GiB**. B=640,k=32
   reached 21.77 GiB with no throughput gain (125.0 examples/s), fixing
   B=512 as the long-k knee.
2. **Selective activation checkpointing.** At B=512,k=16, retaining four
   evenly spaced recurrent layers and checkpointing 22 improved **2.0674 →
   2.0286 s (1.9%)**, at **18.00 GiB**; the final committed candidate
   remeasured **2.0252 s / 252.8 examples/s / 18.01 GiB**. Retaining eight
   layers OOMed. At k=32, retaining even one recurrent layer OOMed once the
   compiled head was live, so the safe all-layer plan remains authoritative.
   Checkpoint calls no longer preserve CUDA RNG state because the captured
   region contains no stochastic operation.
3. **Compiled adaptive surface (historical candidate).** Gate MLP, sigmoid
   alpha/beta, norm-matched mixing, final RMSNorm, packed decoder math, and
   train CE were made regional fullgraph compilations. Gate/mix was whole-step neutral at B=256,k=8
   (**530.4 ms compiled vs 531.6 ms eager**) but eliminates the last material
   trainable pointwise island; the mandatory zero-init and perturbed-surface
   gradient gate still passes. Compiling the standalone eval readout was
   rejected (**3.535 vs 3.517 ms**) and fused AdamW was also rejected for this
   small parameter list (**0.283 vs 0.112 ms**). The hindsight pass below
   retains the decoder/CE wins and retires the three tiny boundaries.
4. **Frozen prompt-state cache.** Repeated task evaluations share a bounded
   packed BF16 pinned-host LRU. On parity B=256, its entry is 0.432 GiB. After
   warming both tensor layouts outside timing, a compiled k=32 sweep segment
   fell from **435.4 to 295.1 ms (1.48×; 140.3 ms saved)** with bitwise-equal
   hiddens. The single packed transfer avoids 53 separate CUDA allocations.

Manual forward+backward CUDA graphs helped the old small B=64,k=4 shape
(**97.55 → 90.87 ms, 1.07×**), but not the throughput contract: B=512,k=4
improved only **0.35%**, B=256,k=8 **1.06%**, and selective B=512,k=16
**0.81%**, while reserving **14.82, 15.23, and 20.34 GiB** respectively in
private graph pools. Persisting one graph per task/shape would consume the
memory needed by other shapes; shared pools impose output-lifetime and replay
ordering constraints disproportionate to the measured gain. The manual graph
path was therefore not implemented. Profiler attribution (42.9% GEMM, 30.8%
FA2 backward; task heads 3.0% of max-k evaluation) likewise gave no case for
installing an external kernel package.

## Cross-module compile hindsight pass (2026-08-21)

Compilation was re-audited region by region after the wire, training, and
max-k evaluation paths had reached their current form. The heavy wire regions
remain clear wins on the RTX 4090: prefill 111.54 → 63.32 ms, first slab
10.06 → 2.60 ms, recurrent step 12.11 → 4.73 ms, two-key merge 0.186 →
0.051 ms, and fused NLL 6.16 → 3.25 ms. The manual steady graph also remains
canonical: 6.6% faster at B=128 and 1.1–1.8% at B=512 with no measured peak
memory increase.

Two smaller compilation classes did not pay. The wire's logits-only B=512
head saved 0.023 ms per call but cost 1.01 s from an empty cache, a roughly
44,000-call break-even; it is now eager while fused NLL remains compiled.
Training's adaptive gate/mix, standalone final norm, and final norm+concat
cost 4.44 s cold together while changing a complete B=256,k=8 step by less
than 0.2%; the two norm boundaries were individually slower compiled. Those
three regions are now eager, while packed decoder layers and fused CE remain
compiled.

Max-k evaluation reduced the default full-wire task grid to exactly eight
prefill lengths. The old process-global `recompile_limit=256` is therefore
retired; the exact current sequence passes Dynamo's default limit. More
importantly, preparing its longest T=137 input first prevents the cache growth
from T=105 from recompiling cache-dependent slabs. With empty Inductor caches,
the complete preparation sequence fell from 59.29 to 42.08 s and 15 to 12
graphs at B=64; at the canonical B=512 it fell from 65.18 to **49.40 s**
(−24.2%), again 15 to 12 graphs, despite the explicit extra warmup execution.

Dynamic prefill was rejected in favor of this ordering fix. It collapsed
four task prefill shapes to one graph and shifted PG19/C4 perplexity only
+0.051%/+0.025%, but changed several n=64 forced-choice cells by one example.
Largest-first static preparation captures the useful lifecycle gain without
changing the canonical numerical path. Sweep construction is now shared by
the task CLI and periodic training evaluation through one `encode_dot_sweep`
helper in `tasks`; no fourth generic utilities module was introduced.
