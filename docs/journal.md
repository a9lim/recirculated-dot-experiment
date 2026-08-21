# Journal

Lab notebook: dated, chronological, append-only. The authoritative
*current* state lives in [design.md](design.md) (architecture, recipe,
decision register) and [findings.md](findings.md) (data and
interpretation); those are pruned freely as canon moves. This file is
where the history goes when they shed it — what was tried, measured,
broken, rejected, and superseded, in the order it happened. Rewriting
old entries defeats the point; append.

---

## 2026-08-20 — Project start; wire v0; Gate G0

Design laid down (D1–D8): two-pass paper-faithful wire, minimal
trainable surface, discriminating 2×2 task suite, halting as sampling,
scope flag, training mechanics, Gemma3-1B PT with pair {11,4},
`<t>` := `<unused0>`. Public repo created as a submodule of the
private root; task generators shared with the sibling chain-of-dots
experiment via `transformer_experiments.dot_tasks`.

First engine built directly on HF Gemma3 layer drives, fp32 sdpa.
**Identity gate exact**: α=0 two-pass vs plain forward, max |Δlogit|
1.35e-4, ppl identical to 4 decimals (27.0329). **Perplexity repro**
(100×512-token windows, α=0.15, convex norm-ratio mix, 10-step ramp):

| pair (0-indexed) | PG19 | C4 |
|---|---|---|
| baseline ppl | 23.03 | 17.99 |
| **{11,4}** (paper) | **−8.81%** | **−5.05%** |
| {10,3} (1-indexed reading) | −1.35% | −1.62% |
| {8,7} (adjacent control) | −0.44% | −0.52% |

Three reads: the untrained wire works in the paper's ballpark (their
−14.4% PG19 was at 1024-token windows); the paper's {11,4} is
**0-indexed**, resolved empirically ({10,3} loses most of the effect —
the landscape is sharper than a smooth heatmap suggests); the adjacent
control is ~null, so this is their landscape, not a perturbation
artifact. G0 declared passed. PG19 via the emozilla parquet mirror
(datasets ≥5 refuses script-based deepmind/pg19).

## 2026-08-20 — Wire optimization, rounds 1–3 (73 s → 3 s)

Round 1 (73→5 s per 100×512 windows, ppl invariant throughout):
pass A — layers 0..dest never see refreshed state, so the bottom slab
prefills in parallel, exactly; dual-pass batching — first pass of t and
second pass of t−1 share a snapshot, one [2B] call; deferred chunked
readout; batch scaling (the loop is launch-bound); two-lane DualCache.

Codex correctness consult (adversarial fp32 sequential reference,
agreement 2.76e-7): snapshot/commit ordering, visibility, rope, GQA,
boundaries all held; two real catches fixed — pass A is causal only
under sdpa (eager attends bidirectionally), and
`final_logit_softcapping` was bypassed (None on Gemma3-1B, latent for
other variants). transformers pinned `~=5.15.1` (the wire drives
private layer/cache contracts; the identity gate is the contract test
for bumps).

Round 2: the 4× FA2-vs-sdpa gap at the serial shape was entirely
HF's `repeat_kv` materialization under masks — fixed in-place with
stride-0 GQA expand views (`_wire_sdpa`, 464→119 µs, bitwise
identical). Two measured traps: `enable_gqa=True` with a dense mask
silently falls back to the math backend (3354 µs); FA2's standard API
can't express the per-lane dual-pass mask. torch.compile on the
mutation-heavy inference cache measured 4.5× *slower* (broadcast
materialization, functionalization copy tax) — parked, noted that
compile belongs to the mutation-free training path.

Round 3 (a9's call to push): the dual-pass "mask" is really *per-lane
prefix lengths*, which `flash_attn_with_kvcache` expresses via per-row
`cache_seqlens` — both lanes in-kernel, 43 µs. flash-attn 2.8 exposes
`fwd_kvcache` as an untraceable PyCapsule → wrapped as torch custom op
`wire::fa2_kvcache` with `mutates_args`, after which the slab compiles
fullgraph. FA2 × compile compose; landing 3 s per 100×512 at B=100
(~5.9 ms/step, at the estimated weight-traffic floor).

## 2026-08-20 — Canonicalization (D10) and the bf16 null method

Compiled-FA2 ratified as the *only* path — CUDA, half precision,
jobe-class hardware only; sdpa dual-pass, mask machinery, and backend
flags stripped (git history keeps them). The fp32 identity reference
retired with sdpa; rather than guessing a bf16 tolerance, G0 now
calibrates against a measured **null**: the plain HF forward vs
*itself* under a kernel-tiling change disagrees at mean |Δlogit|
5.45e-2, top-1 0.9648 — and the α=0 engine sits *inside* that null.
Method worth keeping: when a gate loses its exact reference, measure
the self-noise null before pinning tolerances. Same-day addendum:
stock sdpa retired from the fallback tier too (pass A and plain
forwards via HF's flash_attention_2 interface with a registered
mask-interface pair; models load as `attn_implementation=
"wire_attention"`), so baselines and engine share one attention
library end to end. Consequence accepted: the whole flipped model
object is half-precision-only.

## 2026-08-20 — Codex scaling passes (compile, CUDA graph, memory)

Three sessions, each gate-checked:

1. **Compile-everything + packed projections.** Regional
   fullgraph compilation of pass A, first slab step, recurrent
   mix+slab, readout; Q/K/V and gate/up packed once (7→4 GEMMs/layer),
   original Parameters as disjoint views with a state-dict clone hook.
   B=128,T=512: 3.52→2.84 s (1.24×).
2. **Higher-effort CUDA.** Manual steady-step CUDA graph with
   device-resident counters (~2%); **branch-decomposed cache** — both
   attention branches share one physical refreshed prefix `0..t-2`
   plus a one-slot side buffer, FA2 reads it via duplicate
   `cache_batch_idx`, a compiled two-key softmax + fp32 LSE merge adds
   each branch's distinct keys (1.35× on attention; 2.84→2.35 s,
   9.36→8.20 GiB). The two-key merge's changed bf16 association moved
   identity ppl drift 1.32e-3→2.59e-3 — caught by an adversarial gate,
   explicitly ratified, ppl threshold widened 2e-3→**3e-3**. Rejected
   after whole-path measurement: FA2-fused RoPE (isolated 3.2× at
   short prefixes, whole wire −0.5%), RMSNorm-linear weight folding
   (slower, larger drift, no TE wheel for the pinned stack), readout
   overlap on a second stream (GEMM contends, not hides).
3. **Memory/batch scaling.** Fixed B=64 pass-A chunks with padded
   final chunk; packed-projection views retire 0.857 GiB; compact RoPE
   gather `[t,t−1]` inside the graph (~1 GiB at B=512 → 0.001 GiB).
   B=128: 5.35 GiB peak. New default **B=512: 8.01 s, 32.7k tok/s,
   10.15 GiB**, five unique graphs, zero timed recompiles.

## 2026-08-20 — Tasks module (D11); untrained baseline grid

Serialization: id-space composition `[BOS] prompt | think-span →
answer`, answer never in the eval input; knobs pinned so every surface
form is one Gemma3 token (reachability at nodes=10 — "10"/"11" split);
forced-choice eval with full-vocab legality and gold logprob riding
along; `engine.answer_logits` readout added (same answer-position-only
shape as training supervision).

Burned once: Dynamo's default `recompile_limit=8` is a hard abort
under fullgraph, and the then-standalone grid legitimately visited ~30
distinct lengths — it died mid-grid at parity T=69. Raised globally to
256 (later retired — see the hindsight pass, which cut the grid to
eight prefill lengths and restored the default guard).

Untrained grid recorded (full-scope wire arms; 4 tasks ×
{none, wire, dots, dots+wire} + cot, k∈{1..32}, n=512): accuracy at
chance everywhere; legality 0.000 in every non-CoT cell; CoT toplines
split by content (parity fully in answer space at exactly ln ½;
s5 *below* chance — it continues the digit pattern; reachability's
BFS trace leaks the answer but barely helps untrained); the wire's ppl
gain shows through gold_lp in nearly every matched pair. Full readings
in findings.md — this is the money plot's zero line.

Infra lesson from the overnight run: it finished in ~8 minutes but the
in-session watcher's completion never surfaced; push overnight jobe
completions through something durable (a DONE marker file, a push
notification), not an idle session.

## 2026-08-21 — Train module (D12): think-scope BPTT

Forks surfaced and a9-ratified: **F1** think-first (the wire-alone 2×2
cell is inherently full-scope, joins later); **F2** k sampled per
batch, homogeneous within batch; **F3** lm_head over the whole
emission span (a9's call, superseding D9's answer-only line — with
sampled k the dot targets teach a stopping hazard, activating D4's
halting-by-sampling in v0; `CE(answer) + λ·mean(CE(emission))` keeps
the task gradient k-independent and the dot targets content-free, so
H2 stays clean); **F4** mixture as the goal, per-task as the
benchmark. Defaults: online fresh sampling with train seeds disjoint
from the eval seed, full-vocab CE.

Built the functional think-scope forward (frozen parallel prompt
prefill, one parallel bottom-slab call over the dot span, serial
two-pass slab with the gate MLP at each mix, K/V cat'd inside
checkpointed layer calls, differentiable `flash_attn_func` — the
inference kvcache op has no backward, D9 amended accordingly).

The gradient gate's design caught a real bug during drafting: an early
draft merged the second-pass refresh into the visible set *before* the
first pass ran (same-snapshot violation) — identically in both the
functional and reference paths, because the reference shared
bookkeeping. That is why the reference now derives visibility
independently by column index, deliberately dumb and O(k²). Gate
thresholds pinned from measurement, not guesses: the rerun null at the
gate shape measured bitwise zero, so 10×-null thresholds were useless;
absolute thresholds sit in the gap between measured kernel noise
(grad max-rel 1.67e-2) and semantic-bug scale (O(0.5+), measured by
breaking visibility on purpose).

Smoke (parity, dots+wire, 150 steps, B=64, k∈{1,2,4}, eager ~0.33
s/step): loss 29.4→1.7, emission CE 19.5→~0.01, eval legality
0.000→1.00 — the trained row fully claims the answer surface the
untrained baseline lacked; answer CE settles at ~ln 2
(calibrated-but-ignorant parity floor), accuracy still chance as
expected at this scale. Also this session: `load_model` moved to wire
(D10 loading convention), `single_token` publicized, Eq. 1 mix
cross-referenced between wire and train; a shared-utilities audit
concluded **no fourth module** — the duplications are load-bearing
(two layer drives with different shapes, reference independence for
the gate, three CE variants with different memory shapes).

Handed to Codex for the optimization pass.

## 2026-08-21 — Codex: training/task audit + CUDA pass

Full contract audit before optimization; eight correctness fixes
landed together: base's 340 Parameters frozen before any graph
exists; head flattened to a 2-D GEMM; D12's untrained null made
explicit as `dots+think-wire` with the historical full-scope arm
separately named `dots+full-wire`; CoT tokenized as one natural
space-bearing string truncated by actual token count (s5's trace is
15 tokens for eight states, not 8); Torch/CUDA seeding plus
optimizer/RNG-complete atomic resumable checkpoints; a perturbed-state
gradient gate (zero-init `out.weight` blocks the hidden MLP layers'
gradients — a second deterministic nonzero state activates them);
one BF16 live-row readout shared by both trained arms (the fp32
surface masters dropped — BF16 throughout, only Gemma's own RMSNorm
fp32 accumulator remains); project tests + Ruff.

Seven performance changes: `checkpoint=auto` (retain activations while
B·k ≤ 2048, recompute above); compiled one-shape 512-row CE slab;
max-k sweep evaluation (one causal run, every k read as a prefix —
2.92× over standalone executions, differences inside the established
bf16 tiling null, full-vocab top-1 100%); shared regional compilation
across prompt/span/serial math; refresh+first branches through one
packed projection and one differentiable FA2-varlen call; training
reuses the wire's packed projections as views; bounded deterministic
producer with pinned nonblocking transfer. Paired evidence at
B=64,k=4: **290.91→98.72 ms/step (2.95×), 6.38→3.54 GiB**; the
baseline had been accumulating 1.862 GiB of gradients for all 340
frozen base Parameters, the new path accumulates zero. A rejected
intermediate that persisted assembled prefixes hit 22.76 GiB; no such
cache remains.

## 2026-08-21 — Codex: throughput follow-up + compile hindsight

K-aware effective batches: training defaults to effective **B=512**;
k=8 runs as two retained B=256 microbatches (1.060 s, 482.8 ex/s,
13.08 GiB — 11.1% over one checkpointed B=512 step), k=16 keeps B=512
with four retained recurrent layers {7,13,19,25} (2.025 s, 252.8
ex/s, 18.01 GiB), k=32 recomputes every layer (4.024 s, 127.2 ex/s,
17.79 GiB; retaining even one layer OOMs; B=640 gains nothing).
Prompt-state pinned-host LRU: repeated evaluations restore frozen
prompt hidden/KV in one packed transfer (parity B=256 entry 0.432 GiB;
compiled k=32 sweep segment 435→295 ms, bitwise-equal hiddens).
Manual whole-step CUDA graphs rejected: ≤1% at throughput shapes for
15–20 GiB of private graph pools. Profiler attribution (42.9% GEMM,
30.8% FA2 backward) gave no case for external kernel packages.

Compile hindsight pass, region by region: the heavy wire regions keep
clear wins (prefill 111.5→63.3 ms, recurrent step 12.1→4.7 ms, fused
NLL 6.2→3.3 ms; steady graph stays canonical); the tiny boundaries
did not pay and reverted to eager — the logits-only head saved 0.023
ms/call against 1.01 s cold (a ~44,000-call break-even), and the
gate/mix + final-norm boundaries changed a whole step by <0.2% while
costing 4.44 s cold. Max-k evaluation cut the default task grid to
exactly eight prefill lengths, so the process-global
`recompile_limit=256` was retired — the sequence passes Dynamo's
default guard, with the real protection being largest-shape-first
preparation (T=137 before T=105, so cache growth can't recompile
cache-dependent slabs; cold prep 65.2→49.4 s at B=512, 15→12 graphs).
Dynamic prefill was rejected on principle: it collapsed four shapes
to one graph at +0.05% ppl, but changed several n=64 forced-choice
cells by one example — grid canon stays per-shape. `encode_dot_sweep`
became the one shared sweep constructor; still no fourth module.

## 2026-08-21 — Docs restructure

design.md and findings.md had drifted into ledgers — decisions and
measurements accreted in the order they happened, with superseded
layers marked by addenda. Restructured (a9's call): design.md is now
authoritative current-only (idea, architecture, tasks, training
recipe, gates, decision register with the stable D-numbers);
findings.md is current data + interpretation; this journal absorbs
the history and takes future entries as they happen; AGENTS.md and
README.md properly populated. Handoff gates re-run on jobe at
e80af1c before the restructure was committed — see findings for the
witnessed numbers.
