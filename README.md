# Recirculated Dots

Can a frozen pretrained model be taught to think in opaque tokens?
This experiment combines recirculation (Mozer et al.,
[arXiv:2608.17981](https://arxiv.org/abs/2608.17981)) — a deep-to-shallow
activation wire that carries serial state between positions — with
filler-token thinking (Pfau et al.,
[arXiv:2404.15758](https://arxiv.org/abs/2404.15758)): the model emits
meaningless `<t>` tokens while the wire does the thinking. Each dot
becomes one step of a recurrent computation the filler-token bound says
the token channel alone cannot express. The trainable surface is one
embedding row and a small gating MLP; the base model (Gemma3 1B) stays
frozen.

Decisions ledger: [docs/design.md](docs/design.md). Paper digests:
[docs/literature.md](docs/literature.md). Results:
[docs/findings.md](docs/findings.md). Task generators are shared with
the sibling chain-of-dots experiment via the parent repo
([transformer-experiments](https://github.com/a9lim/transformer-experiments)).

CUDA entry points on Jobe:

```bash
python -m recirculated_dot_experiment.g0 identity
python -m recirculated_dot_experiment.train gate
python -m recirculated_dot_experiment.train run
python -m recirculated_dot_experiment.train run --resume
python -m recirculated_dot_experiment.tasks
```

Training defaults to an effective BF16 B=512 with k-aware equal-shape
microbatching, selective checkpointing, overlapped input preparation,
and compile/cache warmup outside timing. Atomic resumable state lives at
`data/train/surface.pt`. Task scope labels distinguish
`dots+think-wire` (the D12 null) from historical `dots+full-wire`.

## License

CC BY-SA 4.0.
