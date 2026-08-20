# Literature synthesis

PDFs in `references/`, manifest in `references/refs.yaml`.

## Recirculation (Mozer, Siddiqui, Sawyer, Sanyal, Liu — DeepMind, arXiv:2608.17981)

Inference-time recurrence for off-the-shelf models. At input step t+1,
the residual stream at a shallow **destination** layer d mixes in the
norm-matched **source** activation from a deep layer s of step t's
column: `d' = (1−α)d + α(‖d‖/‖s‖)s`, α≈0.1–0.15.

Orchestration (the subtle part): each column is computed **twice**.
Token t is sampled after its first pass (no latency); its second pass
runs concurrently with token t+1's first pass and **refreshes its KV
cache** from layer d up. Cross-position state flows through attention
over refreshed same-layer KVs — layer d holds z(t) and z(t+1) in
adjacent columns, a true z(t+1)=f(z(t),x(t)) no feedforward stack can
express. Generation ~free (two stacks in parallel); prefill serial.

Results to remember:
- Untrained, on Gemma3 PT: 8–23% ppl reduction across ten datasets;
  GSM8k 4B pass@1 29.3→30.6 (→35.5 adaptive). Larger models benefit
  more. Instruction following and contextualization (Racing Thoughts)
  improve; single-token benchmarks only marginally.
- **Gemma-specific in magnitude.** Qwen3/Pythia/Phi2/Ministral show
  the same qualitative sweep region but <0.5% untuned. Suspected
  cause: Gemma's Peri-LN (output LN keeps late-layer contributions
  alive). We are studying "frozen Gemma3 + wire," and say so.
- **Adaptive recirculation**: frozen base + MLP (2 hidden GELU,
  hidden=d_model, LN-in, input concat(s,d), sigmoid out) producing
  per-token *vector* α,β; init α=0.1/β=0.9. 100 steps, bs 32, AdamW
  lr 3e-4, wd 1e-4, BPTT. Matches full fine-tuning (23.0% vs 21.6%
  ppl reduction). Vector > scalar; conditional > constant.
- Hyperparameters: 1B {s=11,d=4} convex mix β=1−α, α ramped
  min(t/10,1)·α over first 10 positions; 4B {18,9} and 12B {35,16}
  need non-convex β=1. Normalization scheme affects robustness of the
  sweep landscape, not the peak.
- Their future-directions list includes blockwise recirculation
  (parallel prefill in chunks), r>1 iterations, multiple paths —
  all knobs we may want.

## Let's Think Dot by Dot (Pfau, Merrill, Bowman — NYU, arXiv:2404.15758)

Filler tokens ('......') can replace CoT on 3SUM/2SUM-Transform, but:
- **Expressivity bound**: fillers add parallel width only; answers
  stay in TC0. No information flows between filler positions through
  the token channel (identical input embeddings), so no serial state.
- **Learnability barrier**: models learn to exploit fillers only with
  dense, parallelizable-decomposition supervision; ordinary CoT data
  doesn't transfer.

## The complementarity (why this project exists)

Both of Pfau et al.'s negative results are about a missing wire, and
recirculation is exactly that wire:
- No serial state between dots → the wire carries z(t)→z(t+1);
  each dot becomes one recurrent step of frozen layers d→s (H1).
- No gradient path crediting a dot with serial progress → BPTT
  through the wire creates it (H2).
Versus coconut: the token channel stays on-distribution (dots are
real tokens, sampled normally) and the neuralese rides as a small
additive mid-stack leak the pretrained model provably tolerates —
rather than replacing the input channel outright.

## Adjacent (not yet pulled)

- Coconut — Hao et al., arXiv:2412.06769 (latent CoT via curriculum).
- Pause tokens — Goyal et al., arXiv:2310.02226 (trained-from-start dots).
- Catch Your Breath — Galashov et al., arXiv:2510.13879 (adaptive
  pause insertion; the adaptive-dot-count axis, and a Mozer paper).
- Merrill & Sabharwal CoT expressivity line (theory frame for H1).
- Full-bandwidth transformer (FBT) — Wang et al., arXiv:2608.08888
  (the sibling project's method: top-layer state GLU-fused into the
  next input, natively pretrained; the native-training counterpart to
  our frozen-model wire).
