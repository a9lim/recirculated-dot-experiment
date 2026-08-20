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
