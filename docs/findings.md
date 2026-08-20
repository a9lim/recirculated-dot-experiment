# Findings

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
separately compiled two-key softmax and fp32 LSE merge reconstruct the
two exact attentions before the refresh write. Across all 511 steady
positions, one layer's attention sweep fell from 35.04 to 25.94 ms
(1.35x). Whole-wire warm latency with the CUDA graph is **2.355–2.361 s,
27.8k token/s**, a **1.20x throughput gain** over the 2.84 s prior path.
Warm peak allocation falls from 9.36 to **8.20 GiB** because the slab KV
history is no longer duplicated.

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

Final gates on the shared-prefix algorithm: bf16 identity PASS (mean
|dlogit| 5.82e-2, top-1 0.9824, plain/engine ppl 27.0229/27.0587), exact
repeatability under graph replay, T=1..4 in bf16 and fp16, and arbitrary
alpha fallback all finite. The 100x512 reproduction remains PG19
**-9.00%** and C4 **-5.09%**. The small identity shift is inside the
pre-calibrated bf16 kernel-order null and below every G0 threshold.
