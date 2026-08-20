"""Gate G0 (design.md): implementation-correctness checks for the wire.

  identity  alpha=0 two-pass run must match the plain HF forward
            (second passes recompute identical activations, so the KV
            overwrite is a no-op; run in fp32, tolerance covers serial-
            vs-parallel op-order differences only).
  repro     untrained perplexity reduction on text windows, paper pair
            {11,4} at alpha=0.15 — expect order 10% on PG19, ~4% on C4.
            --pairs adds controls (e.g. adjacent bad pair 8,7 -> ~none).

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


def load_model(name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation="sdpa"
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


def batched(windows, batch, device):
    for i in range(0, len(windows), batch):
        yield torch.tensor(windows[i : i + batch], device=device)


def nll_sum(logits: torch.Tensor, ids: torch.Tensor, chunk: int = 64) -> tuple[float, int]:
    # chunked along positions: fp32 log_softmax over a 262k vocab at full
    # length would double the logits' multi-GB footprint
    total = 0.0
    for i in range(0, ids.shape[1] - 1, chunk):
        logprobs = torch.log_softmax(logits[:, i : i + chunk].float(), dim=-1)
        targets = ids[:, i + 1 : i + 1 + chunk]
        total += -logprobs.gather(-1, targets.unsqueeze(-1)).sum().item()
    return total, ids[:, 1:].numel()


@torch.no_grad()
def perplexity(nll_fn, windows, batch, device) -> float:
    total, count = 0.0, 0
    for ids in batched(windows, batch, device):
        s, n = nll_fn(ids)
        total, count = total + s, count + n
    return float(torch.exp(torch.tensor(total / count)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["identity", "repro"])
    p.add_argument("--model", default="google/gemma-3-1b-pt")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--source", type=int, default=11)
    p.add_argument("--dest", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--ramp", type=int, default=10)
    p.add_argument("--datasets", default="pg19,c4")
    p.add_argument("--windows", type=int, default=100)
    p.add_argument("--window-len", type=int, default=512)
    p.add_argument("--per-doc", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument(
        "--pairs",
        default=None,
        help="extra source,dest pairs to run as controls, e.g. '8,7;12,4'",
    )
    p.add_argument(
        "--compile",
        nargs="?",
        const="default",
        default=None,
        help="torch.compile the slab step ('default' or 'reduce-overhead')",
    )
    args = p.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if args.mode == "identity":
        args.dtype, args.windows, args.per_doc = "float32", 4, 1
        args.window_len = min(args.window_len, 128)
    dtype = getattr(torch, args.dtype)
    tok, model = load_model(args.model, device, dtype)
    print(f"{args.model} on {device} ({args.dtype})")

    if args.mode == "identity":
        windows = collect_windows("pg19", tok, args.windows, args.window_len, 1)
        ids = torch.tensor(windows, device=device)
        base = model(ids).logits.float()
        engine = RecirculationEngine(
            model,
            WireConfig(args.source, args.dest, alpha=0.0, ramp_steps=0),
            compile_mode=args.compile,
        )
        ours = engine.teacher_forced_logits(ids).float()
        diff = (ours - base).abs()
        base_ppl = float(torch.exp(torch.tensor(nll_sum(base, ids)[0] / nll_sum(base, ids)[1])))
        ours_ppl = float(torch.exp(torch.tensor(nll_sum(ours, ids)[0] / nll_sum(ours, ids)[1])))
        print(f"max|dlogit| {diff.max():.2e}  mean|dlogit| {diff.mean():.2e}")
        print(f"ppl plain {base_ppl:.4f}  engine(alpha=0) {ours_ppl:.4f}")
        ok = diff.max().item() < 1e-3
        print("IDENTITY", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)

    pairs = [(args.source, args.dest)]
    if args.pairs:
        pairs += [tuple(map(int, q.split(","))) for q in args.pairs.split(";")]
    for name in args.datasets.split(","):
        windows = collect_windows(name, tok, args.windows, args.window_len, args.per_doc)
        print(f"\n{name}: {len(windows)} windows x {args.window_len}")
        t0 = time.time()
        # baseline materializes full [B, T, 262k] logits — cap its batch
        base = perplexity(
            lambda ids: nll_sum(model(ids).logits, ids), windows, min(args.batch, 16), device
        )
        print(f"  baseline           ppl {base:8.3f}          ({time.time() - t0:.0f}s)")
        for s, d in pairs:
            engine = RecirculationEngine(
                model,
                WireConfig(s, d, alpha=args.alpha, ramp_steps=args.ramp),
                compile_mode=args.compile,
            )
            def engine_nll(ids, engine=engine):
                nll, _ = engine.teacher_forced(ids)
                return nll.sum().item(), nll.numel()

            t0 = time.time()
            ppl = perplexity(engine_nll, windows, args.batch, device)
            if device == "cuda":
                torch.cuda.empty_cache()
            drop = 100.0 * (base - ppl) / base
            print(
                f"  recirc {{{s:2d},{d:2d}}} a={args.alpha}  ppl {ppl:8.3f}  "
                f"{drop:+6.2f}%  ({time.time() - t0:.0f}s)"
            )


if __name__ == "__main__":
    main()
