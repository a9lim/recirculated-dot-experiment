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
