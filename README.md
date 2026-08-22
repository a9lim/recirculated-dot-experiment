# Recirculated Dots

Can a frozen pretrained model be taught to think in opaque tokens?

This experiment combines **recirculation** (Mozer et al.,
[arXiv:2608.17981](https://arxiv.org/abs/2608.17981)) — a
deep-to-shallow activation wire that carries serial state between
positions — with **filler-token thinking** (Pfau et al.,
[arXiv:2404.15758](https://arxiv.org/abs/2404.15758)): the model
emits meaningless `<t>` tokens while the wire does the thinking.

```
normal CoT:      text  text  text  answer
dots:            text  <t>   <t>    answer          (no wire between dots)
recirculation:   text+s text+s      answer+s        (wire, no extra time steps)
this project:    text[+s] <t>+s <t>+s ... answer+s  (wire x dots)
```

Each `<t>` becomes one step of a recurrent computation whose body is
frozen transformer layers — a computation the filler-token
expressivity bound says the token channel alone cannot express, with
the dot count as an unrolled time axis you can scale at inference.

The trainable surface is deliberately minimal: **one embedding row**
(Gemma3's reserved `<unused0>`, tied with the LM head — training it
shapes both how a dot is read and when one is emitted, so halting is
ordinary sampling) plus **one small gating MLP** on the wire. The
base model (Gemma3-1B PT) stays frozen. That minimality is the
thesis: if it suffices, latent serial thought is something a
pretrained transformer can do *natively*, given only a wire and a
place to stand.

## Method in brief

- **Wire**: the paper's two-pass recirculation at layer pair {11,4},
  norm-matched mixing, verified against its published untrained
  perplexity signature (−9% PG19 / −5% C4 here; controls null).
- **Tasks**: a discriminating 2×2 — {dots, none} × {wire, none} —
  over serial tasks (S5 word problems, parity, graph reachability)
  with a parallelizable control (3SUM) and a legible-CoT topline.
  Forced-choice evaluation with single-token answers; the money plot
  is accuracy vs dot budget k.
- **Training**: BPTT through a functional (mutation-free) rebuild of
  the wire, answer supervision plus a content-free emission term;
  dot budget sampled per batch. Everything is gated: an identity
  check (engine ≡ plain forward at α=0), a gradient check against an
  independent naive reference, and recompile audits in every timed
  loop.

## Repository

| | |
|---|---|
| `recirculated_dot_experiment/wire.py` | two-pass engine (inference; FA2 custom ops, CUDA-graph steady step) |
| `recirculated_dot_experiment/tasks.py` | task serialization + forced-choice eval |
| `recirculated_dot_experiment/train.py` | think-scope BPTT, gradient gate, training CLI |
| `recirculated_dot_experiment/g0.py` | wire gates: identity + perplexity repro |
| `scripts/lab.sh`, `scripts/score.py` | operator front door: gates, probed detached runs/queues, post-hoc scoring |
| `docs/design.md` | authoritative design: architecture, recipe, decisions |
| `docs/findings.md` | current results and interpretation |
| `docs/journal.md` | chronological lab notebook |
| `docs/literature.md` | paper digests |

Task generators are shared with a sibling experiment via the parent
repo ([transformer-experiments](https://github.com/a9lim/transformer-experiments)).

**Status**: wire and training path built and gate-verified; first
learned signal at parity length 4 (wire-trained, wire-run beats the
untrained forced-choice baseline; no length transfer yet — see
findings); recipe D15 (cosine, fat-tailed k, λ=0.125) is the current
training configuration.

## Running

Everything model-facing requires CUDA, bf16, and flash-attn (a single
24 GB GPU suffices; developed on an RTX 4090 with torch 2.8,
flash-attn 2.8.3, transformers ~=5.15.1):

```bash
python -m recirculated_dot_experiment.g0 identity     # engine == HF forward at alpha=0
python -m recirculated_dot_experiment.g0 repro        # untrained perplexity signature
python -m recirculated_dot_experiment.tasks           # task grid, all conditions
python -m recirculated_dot_experiment.train gate      # BPTT gradient gate
python -m recirculated_dot_experiment.train run       # train the surface (--resume to continue)
```

`tasks.py` and the test suite import without a GPU. Day-to-day
operation goes through `scripts/lab.sh` (`gate`, `probe`, `train TAG
…`, `queue FILE`, `score TAG…`, `status`, `watch`, `kill`); see its
header and `scripts/example-queue.txt`.

## License

CC BY-SA 4.0.
