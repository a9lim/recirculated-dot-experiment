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

## 2026-08-21 — H2 launch: two warmup crashes, then D14

First real-run launch (parity dots+wire + dots control, sequential,
detached with a DONE marker) died twice in warmup, both times caught
in minutes because the launch was probed rather than trusted:

1. **Recompile guard, round two.** `_dual_layer_math` blew Dynamo's
   default 8-per-frame limit at k=32 — span-view strides, rope
   slices, and prefix lengths all carry k, so the dual layer
   legitimately compiles a shape family per k. The hindsight pass
   had retired the global `recompile_limit=256` after validating the
   task grid and a k=8 train smoke under the default guard; the
   full-k training warmup was the unexercised hole. Fix: raise the
   limit at **CLI scope** in train/tasks mains (imports still never
   touch global Dynamo config); the accidental-recompile guard
   remains the in-step unique-graph audit.
2. **Warm-set heuristic hole.** With the guard raised, warmup
   completed (36 graphs) but the audit tripped at the first sampled
   k=8 step: the structural warm set [max,1,2,4] never ran k=8's
   execution plan, whose B=256 microbatch gives the dynamic=False
   head-CE chunk a new [256,2] shape. Fix: warm **every** configured
   k, largest first — each k is shape-distinct somewhere, and
   compile is setup by canon.

Lesson recorded: an audit that fires is doing its job — both crashes
were the zero-compile-in-step contract catching real coverage holes,
not noise. Probe a launch with `--steps 4` before detaching it.

**D14 landed** (a9 ratified): free-running evaluation as a *derived*
readout on the max-k sweep. Content-free identical dots make a free
rollout's prefix equal the teacher-forced one, so halting needs no
generation loop: greedy (first non-`<t>` argmax; non-halt within
budget scores as wrong and is reported) plus the exact closed-form
sampled-halting marginal — P(halt at t) = (1−p_t(dot))·∏p_s(dot),
gold/legal emission masses, expected halt time — all from one run's
per-position logits, no Monte Carlo. Verified against a 200k-rollout
Monte Carlo simulation in the tests. This finally *measures* what
D4/D12 train (the stopping hazard), and instance-conditional halting
correlated with serial difficulty becomes a second wire-vs-no-wire
discriminator. The forced k-sweep stays the training monitor; the
running H2 jobs predate the readout, so their checkpoints get scored
post hoc.

## 2026-08-21 — First real runs: parity v0, an informative null

Both arms ran clean at defaults from `8020e51` (wire ~50 min, dots
~22 min, zero in-step compiles, exit 0). The monitor-while-working
setup earned its keep twice over; annoyances for the next runner:
detached stdout is block-buffered (evals surface ~1400 steps late —
use PYTHONUNBUFFERED=1) and atomic checkpoint *replacement* keeps
only final state (copy per eval point if trajectories matter).

Result: accuracy chance everywhere, both arms, all k — H2 unresolved
at 2000×512 and serial depth 32. But the post-hoc pass (sweep +
gold_lp + D14 free running, untrained null, and a transfer cell) made
it an informative null — full table in findings. The keeper: the
wire-trained surface is *wire-dependent* — identical gold_lp across
arms at k≤2, exactly where the wire is structurally invisible to the
readout (refreshed columns reach a readout only from t=3; a live
semantics cross-check), then collapse at k≥4 when the wire is removed
(−0.72 → −3.7..−9.1, legality 1→0, halting 1→never). Both trained
arms self-halt at k~4 with full legality against an untrained
halt-immediately-illegally null, so D4's halting-by-sampling is real
and measured. Everything *around* the computation trains; the
computation didn't, yet. Next fork (a9's call): parity difficulty
scaling vs longer runs.

## 2026-08-21 — Parity length sweep: len 4 cracks, sweep cut short

a9 ratified the difficulty-scaling fork; new runner flags (`--knobs`
for per-run task knob overrides, `--snapshot-every` for immutable
checkpoint copies, `f131a08`) made the sweep {4,8,16} × {wire,dots}
a six-line script. Probed with `--steps 4`, launched detached ~16:26.

parity4-wire replayed the len-32 script for three evals (floor
calibration, oscillating legality) and then left chance at its final
eval — the project's first off-chance accuracy. parity4-dots'
training evals flickered above chance without cohering. When the
control landed a9 called the stop; the runner and the just-started
parity8-wire (pre-snapshot) were killed by hand — sweep-STATUS
records the boundary. Lesson from the kill: `pkill -f` self-matches
an ssh command string that contains the pattern; kill by pid.

Post-hoc at n=512 turned the picture crisp (findings): the *frozen
untrained* model is already above chance on len-4 forced choice
(0.57–0.63 — label-logit margins with garbage outputs), so per-k
untrained deltas are the only honest reference at this length; only
wire-trained + wire-run beats that reference (0.752 at k=32, gold_lp
−0.62 above the ln 2 floor — both readouts, only cell); the entire
gain appeared in steps 1500→2000; and there is no length transfer —
a length-4 lookup, but one necessarily implemented as input-dependent
routing through the wire, since the surface has no other path to the
input. At 16 instances and B=512 the run was effectively full-batch
GD; the training-eval oscillation was optimizer dynamics, not noise.

## 2026-08-21 — D15: the recipe stops being defaults

Went over the training setup with a9, provenance by provenance: the
execution half was measured (D13 knees, calibrated gates), the
learning half never was — lr/λ/B/k-distribution were priors, `--steps
2000` a round number, and flat-after-warmup was the one empirically
grounded choice, inherited from the predecessor's curriculum lesson.
The parity4 data indicted two of them directly: flat 1e-3 as the
prime suspect for the bistable legality/late-phase-change dynamics,
and uniform k for training a halt-at-4 hazard while competence sits
at k=32 (plus 1/6 of steps spent at k=1, where no serial computation
exists).

D15 (a9's calls: cosine, whose stage lesson was always about resets,
never the shape; no k=1 in training; fat tail): within-stage cosine
to `--lr-floor` with `--cosine` period *independent of the run
length* — lr is a pure function of the within-run step (warmup →
one cosine period → flat at the floor), so a run can end mid-cosine
or coast, stages reset by being separate runs, and exact resume
carries no schedule state. Training k decouples from the eval sweep:
`--train-k` (default {2..32}) weighted P(k) ∝ k^γ (`--k-gamma`,
default 1 → E[k]≈22, ~52% of steps at k=32). k=1 stays in the eval
sweep as the structural-invariance probe. Deliberately *not* taken
now: λ-per-k. Note: the (task,k) schedule stream changed, so
resuming pre-D15 checkpoints under new defaults would silently alter
their schedule — fresh runs only across this boundary.

Addendum, same session: a9 asked whether k=2 should go too. It
should, and for a structural reason rather than a hunch — refreshed
columns reach a readout only from t=3, so at k≤2 no supervised logit
sees the wire and the step trains the row alone. Default `--train-k`
is {4,8,16,32}; k≤2 remain in the eval sweep as the invariance probe.

## 2026-08-21 — λ, and the lab gets a front door

λ discussion with a9, from the parity4 logs: the emission term is the
fast easy loss (collapses to its hazard-entropy floor within ~400
steps) and its residual gradient *oscillates by construction* —
homogeneous-k batches alternately push "dot" and "answer" at the same
position, and the tied row absorbs that every step, scaled by λ.
Plausibly a contributor to the bistable small-k legality. The task
plateau itself read as the gate's routing search from a symmetric
answer distribution, not as emission crowding out task gradient, so
λ is second-order to D15's schedule/k changes. Options weighed: lower
constant (cheap, mechanistically clean), anneal high→low (mostly
what cosine already buys), reverse-anneal low→high ("learn to think,
then learn to stop" — elegant, but interference risk through the
tied row and unreadable as an ablation). a9's call: **λ=0.125
constant by default**. Gradient gate re-witnessed under it (loss
12.875, grad 2.698e-2 — numbers move with λ, thresholds hold).

Also raised, not yet acted on: at length 4 B=512 presents 16
instances ~32× per step — full-batch GD — so B=128 would give ~2–2.5×
more optimizer steps per GPU-hour at the same gradient; a len-4
efficiency, not a recipe change, and it evaporates by len 16.

a9 wants to drive runs herself to save session budget, roping the
seat in when something interesting shows up. Hence `scripts/`
(operator tooling, outside the three-module package): `lab.sh`
(gate, probe, probed detached runs and queues with logs/queue.log +
queue-STATUS/DONE, score, status, watch, kill — including the
pkill-by-pattern and stdin-inheritance lessons from today) and
`score.py` (the post-hoc pass generalized, driven by each
checkpoint's saved args: nulls per knob set, home + cross-arm cells,
snapshot trajectories, `--transfer` knob sweeps). The planned parity4
D15 pair sits in `scripts/example-queue.txt`. Smoke-tested end to
end on jobe before handing over.

## 2026-08-21 — activation checkpointing is not a run axis

a9 caught that `--checkpoint {auto,always,never}` exposed an execution
detail as though it were an experimental knob. Removed the flag and the
alternate policy argument beneath it: `_execution_plan` now directly applies
the measured D13 automatic policy, so every warmup and training step uses the
same B*k/microbatch/layer knees. The checkpoint *saving* flags are unchanged;
this is only activation recomputation. Operator and current-state docs now
describe the planner as internal.

## 2026-08-22 — warmup moves inside the schedule horizon

The absolute `--warmup 100` made `--cosine N` reach the floor at
step `N+100`, a hidden extension that also made run-tail arithmetic easy to
misstate. a9 and the seat agreed on one proportional schedule: `--warmup` is
now a ratio (default 0.05) inside the total `--cosine` horizon. For the default
2000-step horizon that is 100 warmup steps, 1900 cosine-decay steps, and the
floor is reached at step 2000; a 3000-step run therefore has a 1000-step flat
tail. `--cosine 0` is flat at peak from the first step and implies zero warmup.
This changes the schedule encoded by pre-change checkpoints, so runs do not
resume across this boundary.

## 2026-08-22 — parity16 null; the gate had been dead all along

a9 ran `parity16-wire` overnight through the queue (10000 × 512, len
16, D15 at `ffa8cfa`) and asked what gave: accuracy at chance at every
k, every snapshot, every transfer length. The run itself was clean.
Three layers came out of the post-mortem, each with its own numbers in
findings.

1. *The task never moved*, same shape as the len-32 null: legality,
   calibration, wire-dependence, halting all trained; answer CE sat on
   ln 2. The late CE-by-k table (0.776/0.732/0.713/0.697) turned out
   to be ln 2 plus a halting leak that the loss weighting predicts
   exactly — which was the first clue that the "learned" hazard was
   arithmetic: `λ·mean` weighs each `<t>` target λ/k against 1 for the
   answer, P∝k cancels the 1/k, and the dot-4 readout learns
   p(answer) ≈ 0.91 at any γ. Greedy halt at 4.0 in every run was that.
2. *The gate was dead.* Recording α, β per dot position on the
   snapshots: exactly 0 or 1 per dimension, identical across instances,
   weights frozen from step ≤2000, pre-sigmoid logits at ±100. Same
   signature in parity-wire (len 32) and parity4-wire. A 200-step
   replica with snapshots every 10 put the death between steps 200 and
   500: AdamW's lr-sized per-entry steps on the zero-init 1152-fan-in
   output layer drift the logits coherently (lr·Σ|a| per step), warmup
   lifts lr into that, bf16 rounds σ to exactly 1 from z ≥ 6.25, and
   the gradient is gone. Every run the project has made trained
   "constant binary mask + row" — H3's gate has never been exercised.
3. *bf16 was eating the optimizer.* `gate.out.bias` and `norm.weight`
   never moved in any run (AdamW steps below half an ulp at magnitude
   ≥ 1), the row freezes in part at the 1e-4 floor, and the bf16 second
   moment cannot decay at all. And the forced-choice readout had become
   one bf16 ulp wide at the trained logit scale — eight distinct margins
   and 40% exact ties on parity16's final sweep, ties resolved to label
   index 0. Re-scoring parity4 through an fp32 readout kept its 12/16 at
   step 2000 (and moved @500 from 0.486 to 0.566); the len-4 claim
   survives precision but is thinner than findings had said — 16
   instances, accuracy quantized to 1/16, p ≈ 0.04 against a coin.

a9's calls: fp32 for everything trained or measured (surface, AdamW
state, gate arithmetic, CE, readouts — the base stays bf16 and the
identity/repro gates did not move a digit), λ default to 1, and scale
the gate's pre-sigmoid output. The seat chose 1/d for the scale (the
Adam output-layer multiplier: with it the logit drift per step is
O(lr) instead of O(lr·d)). On λ the seat deviated from the literal
ask, flagged: λ=1 under the existing mean leaves p(answer | dot 4) at
0.57, still a greedy halt at 4; the form that makes D15's hazard true
is the per-position sum, `CE(ans) + λ·Σ CE(emit)`, where λ=1 is the
plain token-level LM loss and the learned hazard equals the training
P(k). The sum makes per-step gradient magnitude grow with k (up to
33× the answer term at k=32); under Adam that mostly washes out, and
a per-step normalization would have broken the exactness again. The
untrained-model margin also turned out to be a near-linear function
of the bits (R² 0.94 at len 16) — the "above chance at len 4" null in
findings is that linear readout landing on 9–10 of 16 instances.

Checkpoint version 2; v1 surfaces refuse to load (bf16, unscaled gate
— their stored `out.W` means something else under 1/d). The gradient
gate re-witnessed at loss 67.2547 vs 67.2768, grads 3.64e-2 / 3.49e-2;
identity and repro unchanged to print resolution. a9 makes the
commits; jobe got the working tree by rsync for the gates. TF32 is
deliberately left off — inductor warns about it on every fp32 matmul,
and that warning is the point.

Lesson for the logbook: "BF16 throughout" read as a simplicity
virtue and was a silent trainability bug in three places at once;
the cheap probes (α per position, bias norms across checkpoints,
distinct-margin counts) would have caught each on day one and now
live as the first thing to run on a new surface.

## 2026-08-22 — the precision fix, and validating a live gate

Same day, after the parity16 post-mortem above. Landed the three-part
fix a9 signed off on, in one working tree (a9 commits it; jobe got the
tree by rsync since the runs precede the commit):

- **fp32 for everything trained or measured**, bf16 base unchanged: the
  surface row + gate are fp32 master weights, AdamW state fp32, the gate
  MLP arithmetic fp32 (mix casts once back to the residual dtype), the
  span CE fp32 (bf16 GEMM, logits upcast, fp32 log-softmax), and every
  forced-choice/free-running readout fp32 via a new `tasks.fp32_logits`
  chunked over the 262k vocab. Checkpoint version 2; v1 surfaces refuse
  to load (they are bf16 with an unscaled gate, a different meaning of
  `out.W`). The identity and repro gates did not move a printed digit —
  the base path is bit-for-bit what it was.
- **gate pre-sigmoid scaled by 1/d** (`GateMLP.out_scale`). The muP
  point for an Adam-trained zero-init output layer: the coherent drift's
  contribution to the logit is lr·Σ|a|·out_scale ≈ lr·d·rms·out_scale,
  so 1/d makes it d-independent instead of the ~1000× overdrive that
  killed the unscaled gate by step 500.
- **`λ·Σ CE(emit)` at λ=1** replacing `λ·mean`, default λ 0.125→1. The
  mean weighed each `<t>` target λ/k, and with P(k)∝k that cancelled the
  fat tail exactly, pinning greedy halt at k_min for any γ — the
  measured "halts at 4.0". The per-position sum makes the learned hazard
  equal the training P(k); a9's call, one deviation from the literal ask
  (λ=1 under the old mean would only have moved p(answer|dot4) from 0.91
  to 0.57, still a halt at k_min).

An OOM surfaced immediately: the fp32 [512, 262k] CE slabs held across a
k=32 span are ~0.5 GB each × 33 = >16 GB. Fixed by checkpointing each CE
chunk (recompute the head GEMM in backward); peak fell 16.45 → 12.46
GiB, and the bf16 slabs it had also been keeping (~9 GB) are gone too.

**Gate re-witness** (train gate, B=2 parity len4 k=3): loss 67.2547 =
rerun vs reference 67.2768 (rel 3.3e-4; the loss now sums three
emission positions), grads 3.64e-2 zero-init / 3.49e-2 perturbed (the
perturbation std scales with d to match the calibration), span-drive vs
HF 9.42e-2 / top-1 1.0000 inside the HF head's own rounding. 19 tests
green on jobe.

**Validation — a live gate, and the gate-lr knob.** Two 200-step
timelines (snapshots every 10, `/tmp/sat2` and `/tmp/sat3` on jobe),
probing α,β per dot position:

- *1/d, full lr:* cross-instance std of α rises 0.0000 → 0.0035 — the
  first time it has ever been nonzero (every prior run: exactly 0). The
  median pre-sigmoid holds at the 2.2 init while |out.W| climbs to 23.5;
  100% of dims stay unsaturated through step 140, a 0.4–2.6% tail by
  step 200. Alive, engaging, with a slow-growing saturation tail (no
  restoring force — weight decay 0).
- *1/d, gate lr ×0.1:* |out.W| only 2.55 at step 200, 100% unsaturated
  throughout, median pinned at 2.2 — pristine but input-dependence had
  not engaged yet (std still 0.0000 at 200; at full lr it appeared once
  |out.W| crossed ~8.6). Slower wall-clock, same eventual headroom.

That contrast is the whole point of the second knob. a9's call: give the
gate its own AdamW param group at a fraction of the row lr, exposed as
`--gate-ratio` (a9 preferred the ratio over an absolute `--gate-lr`, so
the schedule multiplier is explicit; default 0.1). The gate rides the
row's warmup/cosine shape scaled by the ratio, floor included; each
group carries its own (peak, floor) and the step loop drives them, so
resume stays a pure function of the within-run step (peak/floor
re-asserted from args after load_state_dict, since they are schedule
config, not checkpoint state).

**Why 1/d is the load-bearing one** (a9 asked whether it is necessary
given gate-ratio; discussion, not a new measurement): Adam's step is
scale-invariant, ≈lr·sign(grad) per entry regardless of gradient
magnitude, so lowering the gate lr by r delays the sigmoid freeze by 1/r
but slows W's per-step progress by r — the product, i.e. the
input-dependent structure the gate can build before its gradient dies,
is unchanged. 1/d instead pushes the freeze d× later while W keeps
moving at ~lr, giving ~d× more achievable structure. So gate-ratio only
paces wall-clock; the output scaling sets the ceiling. If one of the two
were dropped, it should be gate-ratio, not the scaling. Left as
argument — a9 declined the cheap falsification (unscaled × ratio sweep,
predicted to keep std ~0 at every ratio).

Nothing has been trained on a task under the corrected recipe yet;
parity4 (a9 will launch) is the first test of whether a live gate
changes the picture. The dead-gate parity results in findings are
therefore all "row + constant mask", and were trimmed to a current-state
summary there — their forensics (the α-mask tables, the
bf16-untrainability ledger: `out.bias`/`norm.weight` frozen because half
an ulp at magnitude ≥1 exceeds the Adam step, the non-decaying bf16
second moment, and the one-ulp-wide readout with 40% ties) are recorded
in the post-mortem entry above and here.

