"""Task serialization + forced-choice evaluation for the 2x2 (D3, D4, D8).

Instances come from the shared generators in
transformer_experiments.dot_tasks (import, don't re-implement); this
module owns everything tokenizer-facing. Sequences are composed in id
space and never re-tokenized:

    [BOS] prompt-ids | think-span  ->  answer     (answer never in ids)

with the think span one of: k copies of <t> := <unused0> (D8), the
legible render_cot trace, or empty. Answers are the pinned single
tokens directly after the last think/prompt token (D8: space-free,
the prompt ends with its own cue token).

Task knobs are pinned so every wire-facing condition has ONE token
length per (task, k): all surface forms (states, node ids, bits, vec
digits) stay in 0..9, each exactly one Gemma3 token — reachability
runs nodes=10 because "10"/"11" split. That keeps the compiled wire at
a single execution shape per condition and makes a CoT trace exactly
one token per serial state (budget-matched against k dots). Ragged
sequences (CoT on reachability) only ever meet the plain forward,
which buckets by length instead.

Evaluation is forced-choice at the final position: argmax over the
task's label tokens (chance = 1/|labels|). Soft-everywhere: full-vocab
legality (does the unrestricted argmax land in the label set) and the
gold answer's full-vocab logprob ride along. Wire conditions go
through RecirculationEngine.answer_logits with duplicate-row batch
padding (discarded); no-wire conditions use the plain flipped-model
forward with logits_to_keep=1.

Importable on the Mac: the wire's flash-attn hard import is reached
only from the CLI, which is jobe-only like everything model-facing.

Run: python -m recirculated_dot_experiment.tasks [--tasks ..] [--k ..]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from transformer_experiments.dot_tasks import (
    Instance,
    parity,
    reachability,
    render,
    render_cot,
    s5_chain,
    sample,
    threesum,
)

DOT = "<unused0>"  # D8: reserved single token, row tied with the LM head

TASKS = {
    "s5_chain": s5_chain,
    "parity": parity,
    "reachability": reachability,
    "threesum": threesum,
}

# Pinned so every rendered surface form is a single token (see module
# docstring); the serial-depth knobs themselves are the shared defaults.
KNOBS: dict[str, dict] = {
    "s5_chain": {"length": 8},
    "parity": {"length": 32},
    "reachability": {"nodes": 10, "edges": 18},
    "threesum": {},
}

# condition -> (think span, wire on). The D3 2x2 plus the CoT topline;
# k applies to the dots conditions only.
CONDITIONS = {
    "none": ("none", False),
    "wire": ("none", True),
    "dots": ("dots", False),
    "dots+wire": ("dots", True),
    "cot": ("cot", False),
}


@dataclass(frozen=True)
class Encoded:
    ids: tuple[int, ...]  # [BOS] prompt + think span; answer NOT included
    answer: int  # gold answer token id
    label: int  # instance label == index into label_ids
    label_ids: tuple[int, ...]  # forced-choice candidates, label-aligned
    think: tuple[int, int]  # [start, end) of the think span within ids


def single_token(tok, text: str) -> int:
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"{text!r} must be one token, got {ids}")
    return ids[0]


def label_space(inst: Instance) -> tuple[str, ...]:
    if inst.task == "s5_chain":
        return tuple(str(i) for i in range(inst.data["n"]))
    if inst.task == "parity":
        return ("0", "1")
    return ("no", "yes")  # reachability, threesum: label 0/1


def encode(tok, inst: Instance, think: str = "none", k: int = 0) -> Encoded:
    prompt, answer = render(inst)
    ids = tok(prompt).input_ids  # BOS via the tokenizer's default specials
    start = len(ids)
    if think == "dots":
        ids = ids + [single_token(tok, DOT)] * k
    elif think == "cot":
        trace = render_cot(inst)
        if trace is None:
            raise ValueError(f"{inst.task} has no CoT trace")
        for state in trace.split():
            ids = ids + tok(state, add_special_tokens=False).input_ids
    elif think != "none":
        raise ValueError(f"unknown think mode {think!r}")
    label_ids = tuple(single_token(tok, s) for s in label_space(inst))
    answer_id = single_token(tok, answer)
    if answer_id != label_ids[inst.label]:
        raise ValueError(f"answer {answer!r} disagrees with label {inst.label}")
    return Encoded(tuple(ids), answer_id, int(inst.label), label_ids, (start, len(ids)))


def _score(logits: torch.Tensor, group: list[Encoded], sums: dict) -> None:
    device = logits.device
    labels = torch.tensor([e.label for e in group], device=device)
    label_ids = torch.tensor(group[0].label_ids, device=device)
    answers = torch.tensor([e.answer for e in group], device=device)
    logits = logits[: len(group)].float()
    sums["acc"] += (logits[:, label_ids].argmax(-1) == labels).sum().item()
    sums["legal"] += torch.isin(logits.argmax(-1), label_ids).sum().item()
    gold = logits.gather(-1, answers.unsqueeze(-1)).squeeze(-1)
    sums["gold_lp"] += (gold - logits.logsumexp(-1)).sum().item()
    sums["n"] += len(group)


@torch.inference_mode()
def evaluate(
    model, encoded: list[Encoded], engine=None, batch: int = 512
) -> dict[str, float]:
    """Forced-choice accuracy (+ full-vocab legality and gold logprob).

    engine=None is the wire-off arm: the plain flipped-model forward
    (identical FA2 kernels, D10 addendum). With an engine, rows must
    share one length — guaranteed by the pinned knobs — and the batch
    is padded with duplicate rows to keep one compiled shape.
    """
    device = next(model.parameters()).device
    sums = {"acc": 0, "legal": 0, "gold_lp": 0.0, "n": 0}
    if engine is not None:
        lengths = {len(e.ids) for e in encoded}
        if len(lengths) != 1:
            raise ValueError(f"wire conditions need one length, got {sorted(lengths)}")
        for i in range(0, len(encoded), batch):
            group = encoded[i : i + batch]
            rows = group + [group[0]] * (batch - len(group))
            ids = torch.tensor([e.ids for e in rows], device=device)
            _score(engine.answer_logits(ids), group, sums)
    else:
        by_len: dict[int, list[Encoded]] = {}
        for e in encoded:
            by_len.setdefault(len(e.ids), []).append(e)
        for rows in by_len.values():
            for i in range(0, len(rows), batch):
                group = rows[i : i + batch]
                ids = torch.tensor([e.ids for e in group], device=device)
                _score(model(ids, logits_to_keep=1).logits[:, -1], group, sums)
    n = sums.pop("n")
    return {key: value / n for key, value in sums.items()} | {"n": n}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", default="s5_chain,parity,reachability,threesum")
    p.add_argument("--conditions", default="none,wire,dots,dots+wire,cot")
    p.add_argument("--k", default="1,2,4,8,16,32", help="dot budgets")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="google/gemma-3-1b-pt")
    p.add_argument("--source", type=int, default=11)
    p.add_argument("--dest", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--ramp", type=int, default=10)
    args = p.parse_args()

    from .wire import RecirculationEngine, WireConfig, load_model

    tok, model = load_model(args.model, "cuda")
    engine = RecirculationEngine(
        model,
        WireConfig(args.source, args.dest, alpha=args.alpha, ramp_steps=args.ramp),
    )
    ks = [int(s) for s in args.k.split(",")]

    # One paired instance set per task (identical across conditions), the
    # whole grid encoded up front so the engine can warm at max length
    # once — later, shorter shapes reuse the capacity and CUDA graph.
    grid = []
    for task in args.tasks.split(","):
        instances = sample(TASKS[task], args.n, args.seed, **KNOBS[task])
        for cond in args.conditions.split(","):
            think, wired = CONDITIONS[cond]
            if think == "cot" and render_cot(instances[0]) is None:
                continue
            for k in ks if think == "dots" else [0]:
                rows = [encode(tok, inst, think, k) for inst in instances]
                grid.append((task, cond, k, rows, wired))
    warm_len = max(len(rows[0].ids) for _, _, _, rows, wired in grid if wired)
    warm = torch.full((args.batch, warm_len), tok.bos_token_id, device="cuda")
    engine.answer_logits(warm)

    current = None
    for task, cond, k, rows, wired in grid:
        if task != current:
            chance = 1 / len(rows[0].label_ids)
            print(f"\n{task} (n={args.n}, chance {chance:.2f})")
            current = task
        r = evaluate(model, rows, engine if wired else None, args.batch)
        kk = f"{k:3d}" if CONDITIONS[cond][0] == "dots" else "  -"
        print(
            f"  {cond:9s} k={kk}  acc {r['acc']:.3f}  "
            f"legal {r['legal']:.3f}  gold_lp {r['gold_lp']:7.3f}"
        )


if __name__ == "__main__":
    main()
