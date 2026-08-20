"""Gate G0 (design.md): implementation-correctness checks for the wire.

  identity  alpha=0 two-pass run must match the plain HF forward
            (second passes recompute identical activations, so the KV
            overwrite is a no-op). Runs in bf16 on the canonical
            compiled-FA2 path (D10); thresholds are pinned from the
            measured serial-vs-parallel bf16 noise floor — machinery
            bugs produce O(1) divergence, not noise. The retired fp32
            sdpa reference passed at max |dlogit| 1.2e-4 on 2026-08-20.
  repro     untrained perplexity reduction on text windows, paper pair
            {11,4} at alpha=0.15 — expect order 10% on PG19, ~4% on C4.
            --pairs adds controls (e.g. adjacent bad pair 8,7 -> ~none).

Reproduction reports compile+capture as an explicit untimed warmup.
Timed batches use one padded execution shape (duplicate rows are discarded
before scoring) and fail if Dynamo records any new graph compilation.

Run: python -m recirculated_dot_experiment.g0 identity|repro [flags]
"""

from __future__ import annotations

import argparse
import time

import torch

from .wire import RecirculationEngine, WireConfig

# emozilla/pg19 is the parquet mirror; deepmind/pg19 is script-based, which
# datasets >= 5 refuses to load.
DATASETS = {
    "pg19": ("emozilla/pg19", None, "test", "text"),
    "c4": ("allenai/c4", "en", "validation", "text"),
}


def load_model(name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    # wire_attention (registered at wire import): FA2 everywhere, so the
    # baselines run the same kernels before and after engine construction
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="wire_attention"
    )
    model.to(device).eval()
    return tok, model


def collect_windows(dataset: str, tok, n_windows: int, win_len: int, per_doc: int):
    from datasets import load_dataset

    path, config, split, field = DATASETS[dataset]
    stream = load_dataset(path, config, split=split, streaming=True)
    windows = []
    for doc in stream:
        ids = tok(doc[field]).input_ids
        for j in range(per_doc):
            chunk = ids[j * win_len : (j + 1) * win_len]
            if len(chunk) < win_len:
                break
            windows.append(chunk)
            if len(windows) >= n_windows:
                return windows
    return windows


def batched(windows, batch, device, *, fixed_shape: bool = True):
    """Yield fixed-shape batches plus their real row count.

    Padding the final batch with a duplicate row keeps one compiled shape;
    callers discard the duplicate NLLs, so evaluation is unchanged.
    """
    if not windows:
        raise ValueError("evaluation needs at least one window")
    batch = min(batch, len(windows))
    for i in range(0, len(windows), batch):
        rows = windows[i : i + batch]
        valid = len(rows)
        if fixed_shape and valid < batch:
            rows = rows + [rows[-1]] * (batch - valid)
        yield torch.tensor(rows, device=device), valid


def nll_values(
    logits: torch.Tensor, ids: torch.Tensor, chunk: int = 64
) -> torch.Tensor:
    """Per-token NLL, chunked to bound the full-vocabulary fp32 buffer."""
    pieces = []
    for i in range(0, ids.shape[1] - 1, chunk):
        logprobs = torch.log_softmax(logits[:, i : i + chunk].float(), dim=-1)
        targets = ids[:, i + 1 : i + 1 + chunk]
        pieces.append(-logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1))
    return (
        torch.cat(pieces, dim=1)
        if pieces
        else torch.empty(ids.shape[0], 0, device=ids.device)
    )


def nll_sum(
    logits: torch.Tensor, ids: torch.Tensor, chunk: int = 64
) -> tuple[float, int]:
    nll = nll_values(logits, ids, chunk)
    return nll.sum().item(), nll.numel()


@torch.no_grad()
def perplexity(nll_fn, windows, batch, device, *, fixed_shape: bool = True) -> float:
    total, count = 0.0, 0
    for ids, valid in batched(windows, batch, device, fixed_shape=fixed_shape):
        nll = nll_fn(ids)[:valid]
        total, count = total + nll.sum().item(), count + nll.numel()
    return float(torch.exp(torch.tensor(total / count)))


def compiled_graph_count() -> int:
    """Pinned-torch audit counter for accidental timed-path recompiles."""
    return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["identity", "repro"])
    p.add_argument("--model", default="google/gemma-3-1b-pt")
    p.add_argument("--source", type=int, default=11)
    p.add_argument("--dest", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--ramp", type=int, default=10)
    p.add_argument("--datasets", default="pg19,c4")
    p.add_argument("--windows", type=int, default=100)
    p.add_argument("--window-len", type=int, default=512)
    p.add_argument("--per-doc", type=int, default=2)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument(
        "--pairs",
        default=None,
        help="extra source,dest pairs to run as controls, e.g. '8,7;12,4'",
    )
    args = p.parse_args()

    device = "cuda"  # canonical wire is CUDA-only (D10)
    if args.mode == "identity":
        args.windows, args.per_doc = 4, 1
        args.window_len = min(args.window_len, 128)
    tok, model = load_model(args.model, device)
    print(f"{args.model} on {device} (bfloat16)")

    if args.mode == "identity":
        windows = collect_windows("pg19", tok, args.windows, args.window_len, 1)
        ids = torch.tensor(windows, device=device)
        base = model(ids).logits.float()
        engine = RecirculationEngine(
            model, WireConfig(args.source, args.dest, alpha=0.0, ramp_steps=0)
        )
        ours = engine.teacher_forced_logits(ids).float()
        diff = (ours - base).abs()
        top1 = (ours.argmax(-1) == base.argmax(-1)).float().mean().item()
        base_ppl = float(
            torch.exp(torch.tensor(nll_sum(base, ids)[0] / nll_sum(base, ids)[1]))
        )
        ours_ppl = float(
            torch.exp(torch.tensor(nll_sum(ours, ids)[0] / nll_sum(ours, ids)[1]))
        )
        print(
            f"max|dlogit| {diff.max():.2e}  mean|dlogit| {diff.mean():.2e}  top1 {top1:.4f}"
        )
        print(f"ppl plain {base_ppl:.4f}  engine(alpha=0) {ours_ppl:.4f}")
        # Thresholds are calibrated against both the measured bf16 null and
        # the accepted specialized two-key softmax: the plain HF forward vs
        # itself under a kernel-tiling change gives mean|dlogit| 5.5e-2 and
        # top1 0.965; the specialized engine gives 5.86e-2 / 0.9746 with
        # ppl rel 2.59e-3. Machinery bugs (e.g. a wrong KV slot) sit about
        # 20x higher in mean error and scramble top1. The 3e-3 ppl boundary
        # is deliberately narrow around the ratified kernel's measured
        # drift; the retired fp32 reference proved the algorithm exact.
        ok = (
            diff.mean().item() < 0.15
            and abs(ours_ppl - base_ppl) / base_ppl < 3e-3
            and top1 > 0.95
        )
        print("IDENTITY", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)

    pairs = [(args.source, args.dest)]
    if args.pairs:
        pairs += [tuple(map(int, q.split(","))) for q in args.pairs.split(";")]

    evaluations = []
    for name in args.datasets.split(","):
        windows = collect_windows(
            name, tok, args.windows, args.window_len, args.per_doc
        )
        print(f"\n{name}: {len(windows)} windows x {args.window_len}")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # baseline materializes full [B, T, 262k] logits — cap its batch
        base = perplexity(
            lambda ids: nll_values(model(ids).logits, ids),
            windows,
            min(args.batch, 16),
            device,
            fixed_shape=False,
        )
        torch.cuda.synchronize()
        print(
            f"  baseline           ppl {base:8.3f}          "
            f"({time.perf_counter() - t0:.2f}s)"
        )
        evaluations.append((name, windows, base))

    for s, d in pairs:
        engine = RecirculationEngine(
            model, WireConfig(s, d, alpha=args.alpha, ramp_steps=args.ramp)
        )
        prepared_shape = None
        print(f"\nrecirc {{{s:2d},{d:2d}}} a={args.alpha}:")
        for name, windows, base in evaluations:
            warm_ids, _ = next(batched(windows, args.batch, device))
            shape = tuple(warm_ids.shape)
            if shape != prepared_shape:
                before = compiled_graph_count()
                torch.cuda.synchronize()
                warm_t0 = time.perf_counter()
                engine.teacher_forced(warm_ids)
                torch.cuda.synchronize()
                warm_seconds = time.perf_counter() - warm_t0
                after = compiled_graph_count()
                print(
                    f"  compile+capture B={shape[0]} T={shape[1]}  "
                    f"{warm_seconds:.2f}s outside timing  "
                    f"({after - before} graphs)"
                )
                prepared_shape = shape

            compiled_before = compiled_graph_count()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ppl = perplexity(
                lambda ids, engine=engine: engine.teacher_forced(ids)[0],
                windows,
                args.batch,
                device,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            compiled_after = compiled_graph_count()
            if compiled_after != compiled_before:
                raise RuntimeError(
                    "timed evaluation triggered a recompile "
                    f"({compiled_before} -> {compiled_after})"
                )
            drop = 100.0 * (base - ppl) / base
            print(
                f"  {name:8s} ppl {ppl:8.3f}  {drop:+6.2f}%  "
                f"({seconds:.2f}s, 0 recompiles)"
            )
        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
