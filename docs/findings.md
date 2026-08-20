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
