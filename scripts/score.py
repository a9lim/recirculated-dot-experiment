"""Post-hoc scorer for trained surfaces — operator tool, not a module.

    python scripts/score.py TAG [TAG ...] [--n 512] [--seed 0]
        [--k 1,2,4,8,16,32] [--no-trajectory] [--transfer KNOB=V1,V2,...]

Each TAG names data/train/TAG.pt; the checkpoint's own args supply the
condition (home arm), tasks, knobs, and wire pair. Per task it prints:

- the untrained null in both arms (once per knob set — the honest
  reference line: at short parity lengths the frozen model is already
  off chance in forced choice);
- the final surface in its home arm and in the other arm (cross-arm
  transfer: wire-trained run wire-free, dots-trained given the wire);
- the snapshot trajectory data/train/TAG.pt.STEP in the home arm;
- with --transfer, the final surface (home arm) and the null under
  each knob value, e.g. parity length transfer (lookup vs algorithm).

Rows: acc/legal/gold_lp per k on the max-k forced sweep, then the D14
free-running line (greedy halt and the closed-form soft marginal).
CUDA + flash-attn only, like everything model-facing (D10).
"""

import argparse
import glob

import torch

torch._dynamo.config.recompile_limit = 256

from recirculated_dot_experiment import tasks
from recirculated_dot_experiment.train import (
    CHECKPOINT_VERSION,
    DotsAdapter,
    PromptStateCache,
    Surface,
    ThinkAdapter,
)
from recirculated_dot_experiment.wire import load_model


def parse_knobs(task: str, spec: str | None) -> dict:
    knobs = dict(tasks.KNOBS[task])
    for pair in (spec or "").split(","):
        if not pair:
            continue
        key, _, value = pair.partition("=")
        if key not in knobs:
            raise ValueError(f"{task} has no knob {key!r}")
        knobs[key] = int(value)
    return knobs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tags", nargs="+")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", default="1,2,4,8,16,32")
    p.add_argument("--no-trajectory", action="store_true")
    p.add_argument("--transfer", default=None, help="KNOB=V1,V2,... overrides")
    p.add_argument("--model", default="google/gemma-3-1b-pt")
    args = p.parse_args()
    ks = [int(s) for s in args.k.split(",")]

    tok, model = load_model(args.model, "cuda")
    dot_id = tasks.single_token(tok, tasks.DOT)
    # Built before any trained surface syncs its row into the model.
    untrained = Surface(model, dot_id).cuda()
    cache = PromptStateCache(model)
    rows_memo: dict[tuple, list] = {}
    null_done: set[tuple] = set()

    def rows_for(task, knobs):
        key = (task, tuple(sorted(knobs.items())))
        if key not in rows_memo:
            instances = tasks.sample(tasks.TASKS[task], args.n, args.seed, **knobs)
            rows_memo[key] = tasks.encode_dot_sweep(tok, instances, ks)
        return rows_memo[key]

    def load_surface(path):
        state = torch.load(path, map_location="cuda", weights_only=False)
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"{path}: checkpoint version {state.get('version')}, scorer wants "
                f"{CHECKPOINT_VERSION} (pre-precision-boundary surfaces are bf16 "
                "with an unscaled gate output; score them at their own commit)"
            )
        surface = Surface(model, dot_id).cuda()
        surface.load_state_dict(state["surface"], strict=True)
        return surface, state

    def score(label, surface, arm, rows, source, dest, batch=512):
        surface.sync_into(model)
        runner = (
            ThinkAdapter(model, surface, source, dest, cache)
            if arm == "think"
            else DotsAdapter(model, surface, cache)
        )
        sweep = tasks.evaluate_dot_sweep(model, rows, runner, min(batch, args.n))
        free = tasks.evaluate_free_running(
            model, rows[max(ks)], runner, min(batch, args.n)
        )
        cells = "  ".join(
            f"k{k}:{r['acc']:.3f}/{r['legal']:.2f}/{r['gold_lp']:.2f}"
            for k, r in sweep.items()
        )
        print(f"{label:26s} {cells}")
        print(
            f"{'':26s} free greedy halt {free['halt']:.3f} acc {free['acc']:.3f} "
            f"legal {free['legal']:.3f} k~{free['k_halt']:.1f} | soft halt "
            f"{free['p_halt']:.3f} gold {free['p_gold']:.3f} legal "
            f"{free['p_legal']:.3f} k~{free['k_soft']:.2f}",
            flush=True,
        )

    def null(task, knobs, source, dest):
        key = (task, tuple(sorted(knobs.items())))
        if key in null_done:
            return
        null_done.add(key)
        rows = rows_for(task, knobs)
        print(f"\n-- untrained null: {task} {knobs} (n={args.n}, seed {args.seed}) --")
        score("untrained/think", untrained, "think", rows, source, dest)
        score("untrained/dots", untrained, "dots", rows, source, dest)

    print("cells: acc/legal/gold_lp per k (forced sweep); free = D14 readout")
    for tag in args.tags:
        path = f"data/train/{tag}.pt"
        final, state = load_surface(path)
        run = state["args"]
        home = "think" if run["condition"] == "dots+wire" else "dots"
        other = "dots" if home == "think" else "think"
        source, dest = run["source"], run["dest"]
        knob_spec = run.get("knobs")
        print(
            f"\n==== {tag}: step {state['step']}, {run['condition']}, "
            f"tasks {run['tasks']}, knobs {knob_spec or '-'}, lr {run['lr']} "
            f"cosine {run.get('cosine', 0)} lam {run['lam']} "
            f"train-k {run.get('train_k', run['k'])} gamma {run.get('k_gamma', 0)} ===="
        )
        for task in run["tasks"].split(","):
            knobs = parse_knobs(task, knob_spec)
            rows = rows_for(task, knobs)
            null(task, knobs, source, dest)
            print(f"\n-- {tag} final, {task} --")
            score(f"{tag}/{home}", final, home, rows, source, dest)
            score(f"{tag}/{other} (transfer)", final, other, rows, source, dest)
            if not args.no_trajectory:
                snaps = sorted(
                    (int(f.rsplit(".", 1)[1]), f)
                    for f in glob.glob(f"{path}.*")
                    if f.rsplit(".", 1)[1].isdigit()
                )
                if snaps:
                    print(f"\n-- {tag} trajectory ({home} arm), {task} --")
                    for step, snap in snaps:
                        if step == state["step"]:
                            score(
                                f"{tag}@{step}/{home}", final, home, rows, source, dest
                            )
                        else:
                            surface, _ = load_surface(snap)
                            score(
                                f"{tag}@{step}/{home}",
                                surface,
                                home,
                                rows,
                                source,
                                dest,
                            )
            if args.transfer:
                key, _, values = args.transfer.partition("=")
                for value in values.split(","):
                    over = parse_knobs(
                        task, f"{knob_spec + ',' if knob_spec else ''}{key}={value}"
                    )
                    null(task, over, source, dest)
                    print(f"\n-- {tag} final under {key}={value}, {task} --")
                    score(
                        f"{tag}/{home} @{key}={value}",
                        final,
                        home,
                        rows_for(task, over),
                        source,
                        dest,
                    )


if __name__ == "__main__":
    main()
