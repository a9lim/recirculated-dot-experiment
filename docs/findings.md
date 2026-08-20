# Findings

## G0 — wire implementation correctness (2026-08-20)

**Identity check: PASS.** Two-pass engine at α=0 vs plain HF forward,
Gemma3 1B PT, fp32 on jobe: max |Δlogit| 1.35e-4, mean 4.8e-6,
perplexity identical to 4 decimals (27.0329). The cache-overwrite /
rope / manual-layer-drive machinery is a verified no-op at α=0.
(`python -m recirculated_dot_experiment.g0 identity`)
