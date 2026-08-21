"""Training path: think-scope BPTT through the wire (design.md D12).

Trainable surface (D2): the `<t>` embedding row (fp32 master, tied —
it is both how a dot is read and the dot's output logit) plus the
paper's alpha,beta gate MLP. Base frozen throughout.

Think scope (D5, ratified think-first): the prompt is prefilled once,
in parallel, frozen and detached — its per-layer KV is a constant.
The dot span is the only serial region. Two exact structural facts
carry over from the wire: dots' bottom layers (0..dest) never see
refreshed state and the dot embedding is input-known, so the whole
span's bottom slab runs as ONE parallel causal call; the serial slab
(dest+1..top) then runs the two-pass recirculation per dot, with the
gate MLP supplying alpha,beta at each mix.

Functional cache (D9): per slab layer, the detached prompt KV, a
tuple of settled one-token refreshed KVs, and the previous column's
first-pass frontier KV. Nothing in the graph is ever overwritten;
dual views are cat'd inside checkpointed layer calls (use_reentrant=
False) so no assembled K/V survives the forward. Attention is
flash_attn_func — differentiable, unlike the inference kvcache op
(D9 amendment: the training path shares kernels, not custom ops).

Supervision (D12, a9's call): lm_head over the whole emission span —
the last prompt position and every dot target `<t>`, the last dot
targets the answer. With per-batch sampled k this teaches a stopping
hazard, activating D4's halting-by-sampling. Loss is
CE(answer) + lambda*mean(CE(emission)) so the task gradient is
k-independent; dot targets carry no task content, so credit for the
computation still flows only through the answer (H2 stays clean).
The head is chunked and checkpointed; logits for the `<t>` column are
recomputed from the live row (tied), everything else from the frozen
head.

Gradient gate (D9, mandatory before any run): the functional path vs
a naive reference — per-column sequential bottom slab, separate
unbatched dual-pass calls, no checkpointing — compared on loss and on
every surface gradient, against the measured nondeterminism null
(flash-attn backward uses atomics; even the same path twice is not
bitwise). Plus an HF cross-check: the parallel span drive with the
row synced into the model must match the plain forward's answer
logits within kernel noise.

Run: python -m recirculated_dot_experiment.train gate|run [flags]
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch
from flash_attn import flash_attn_func
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

from . import tasks


class GateMLP(nn.Module):
    """Paper recipe (D2): LN on concat(source, dest), two hidden GELU
    layers at d_model, sigmoid vector output. Zero-init last layer with
    logit biases so training starts exactly at alpha=0.1, beta=0.9."""

    def __init__(self, d: int, alpha0: float = 0.1, beta0: float = 0.9):
        super().__init__()
        self.norm = nn.LayerNorm(2 * d)
        self.h1 = nn.Linear(2 * d, d)
        self.h2 = nn.Linear(d, d)
        self.out = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias[:d] = math.log(alpha0 / (1 - alpha0))
            self.out.bias[d:] = math.log(beta0 / (1 - beta0))

    def forward(self, h_s: Tensor, h_d: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([h_s, h_d], dim=-1).float()
        x = self.out(F.gelu(self.h2(F.gelu(self.h1(self.norm(x))))))
        alpha, beta = torch.sigmoid(x).chunk(2, dim=-1)
        return alpha, beta


class Surface(nn.Module):
    """The entire trainable state: one row + one gate. fp32 masters."""

    def __init__(self, model, dot_id: int):
        super().__init__()
        d = model.config.hidden_size
        self.dot_id = dot_id
        self.row = nn.Parameter(
            model.model.embed_tokens.weight[dot_id].detach().float().clone()
        )
        self.gate = GateMLP(d)

    def sync_into(self, model) -> None:
        """Write the trained row into the (tied) model embedding, so plain
        HF forwards — the dots-alone eval arm — see the trained surface."""
        w = model.model.embed_tokens.weight
        with torch.no_grad():
            w[self.dot_id] = self.row.to(w.dtype)


def _mix(cfg_eps: float, h_s: Tensor, h_d: Tensor, alpha, beta) -> Tensor:
    ratio = h_d.norm(dim=-1, keepdim=True) / h_s.norm(dim=-1, keepdim=True).clamp_min(
        cfg_eps
    )
    return (beta * h_d.float() + alpha * ratio.float() * h_s.float()).to(h_d.dtype)


def _rope(model, positions: Tensor, dtype) -> dict[str, tuple[Tensor, Tensor]]:
    marker = torch.empty(
        1, 1, model.config.hidden_size, dtype=dtype, device=positions.device
    )
    return {
        lt: model.model.rotary_emb(marker, positions.unsqueeze(0), lt)
        for lt in set(model.config.layer_types)
    }


def _layer_call(layer, causal: bool, x: Tensor, cos: Tensor, sin: Tensor, *kv):
    """One frozen Gemma3 decoder layer over new positions x, attending
    cat(kv-pieces) + itself. kv holds the K pieces then the V pieces
    (flash layout [B, S, kv_heads, dim]); the cat happens in here so
    checkpointing stores only references to the persistent pieces.
    Returns (out, k_new, v_new)."""
    attn = layer.self_attn
    residual = x
    h = layer.input_layernorm(x)
    B, S, _ = h.shape
    q = attn.q_proj(h).view(B, S, -1, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(h).view(B, S, -1, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(h).view(B, S, -1, attn.head_dim)
    q, k = attn.q_norm(q), attn.k_norm(k)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    q, k_new = q.transpose(1, 2), k.transpose(1, 2)
    half = len(kv) // 2
    k_all = torch.cat(kv[:half] + (k_new,), dim=1) if half else k_new
    v_all = torch.cat(kv[half:] + (v,), dim=1) if half else v
    a = flash_attn_func(q, k_all, v_all, softmax_scale=attn.scaling, causal=causal)
    h = residual + layer.post_attention_layernorm(attn.o_proj(a.reshape(B, S, -1)))
    residual = h
    h = layer.pre_feedforward_layernorm(h)
    h = layer.mlp(h)
    return residual + layer.post_feedforward_layernorm(h), k_new, v


@torch.no_grad()
def _prompt_prefill(model, prompt_ids: Tensor):
    """Frozen parallel prompt pass: last top hidden + per-layer KV
    (flash layout, detached). no_grad, not inference_mode — the outputs
    become constants inside a later autograd graph."""
    out = model.model(input_ids=prompt_ids, use_cache=True)
    kv = [
        (k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous())
        for k, v, _ in out.past_key_values  # (keys, values, sliding-window)
    ]
    return out.last_hidden_state[:, -1:], kv


def _pe(rope, layer_types, i, sl=None):
    cos, sin = rope[layer_types[i]]
    return (cos, sin) if sl is None else (cos[:, sl], sin[:, sl])


def _span_parallel(model, x, span_rope, prompt_kv, lo, hi, ckpt):
    """Drive layers lo..hi-1 over the whole span in parallel (causal,
    prompt as prefix). Exact for any layer range that never sees
    refreshed state: 0..dest under the wire, the full stack without it."""
    layer_types = model.config.layer_types
    for i in range(lo, hi):
        args = (model.model.layers[i], True, x, *_pe(span_rope, layer_types, i))
        kv = prompt_kv[i]
        if ckpt:
            x, _, _ = checkpoint(_layer_call, *args, *kv, use_reentrant=False)
        else:
            x, _, _ = _layer_call(*args, *kv)
    return x


def _slab_column(model, x, pe_sl, rope, kv_lists, lo, hi, source, ckpt):
    """First or second pass of one column through the serial slab.
    kv_lists[i] is that layer's visible past (tuple of K pieces + V
    pieces); returns (top hidden, h_source, per-layer new KV)."""
    layer_types = model.config.layer_types
    h_s, new_kv = None, []
    for i in range(lo, hi):
        args = (model.model.layers[i], False, x, *_pe(rope, layer_types, i, pe_sl))
        if ckpt:
            x, k, v = checkpoint(_layer_call, *args, *kv_lists[i], use_reentrant=False)
        else:
            x, k, v = _layer_call(*args, *kv_lists[i])
        new_kv.append((k, v))
        if i == source:
            h_s = x
    return x, h_s, new_kv


def think_outputs(
    model, surface: Surface, ids: Tensor, span_start: int, source: int, dest: int,
    ckpt: bool = True,
) -> Tensor:
    """Think-scope wire forward. Returns supervised hiddens [B, k+1, d]
    (post final-norm): last prompt position, then every dot top."""
    B, T = ids.shape
    P, k = span_start, T - span_start
    dtype = model.model.embed_tokens.weight.dtype
    scale = float(model.config.hidden_size) ** 0.5
    h_init, prompt_kv = _prompt_prefill(model, ids[:, :P])

    positions = torch.arange(P, T, device=ids.device)
    span_rope = _rope(model, positions, dtype)
    x = (surface.row.to(dtype) * scale).expand(B, k, -1)
    h_dest = _span_parallel(model, x, span_rope, prompt_kv, 0, dest + 1, ckpt)

    n_layers = model.config.num_hidden_layers
    slab = range(dest + 1, n_layers)
    # Functional cache: per slab layer, K/V piece tuples of the shared
    # visible past (prompt + refreshed columns 0..t-2) plus the previous
    # column's first-pass frontier. Same-snapshot rule: the second pass
    # of t-1 and the first pass of t read the SAME visible past — the
    # refresh joins it only after both have run (the wire's snapshot
    # boundary), and the first pass sees column t-1 only through the
    # frontier, at first-pass fidelity.
    vis_k = {i: (prompt_kv[i][0],) for i in slab}
    vis_v = {i: (prompt_kv[i][1],) for i in slab}
    frontier: dict[int, tuple[Tensor, Tensor]] = {}
    tops, h_s_prev = [], None
    for t in range(k):
        refresh = None
        if t > 0:
            alpha, beta = surface.gate(h_s_prev, h_dest[:, t - 1 : t])
            x2 = _mix(1e-6, h_s_prev, h_dest[:, t - 1 : t], alpha, beta)
            kv2 = {i: vis_k[i] + vis_v[i] for i in slab}
            _, _, refresh = _slab_column(
                model, x2, slice(t - 1, t), span_rope, kv2,
                dest + 1, n_layers, source, ckpt,
            )
        kv1 = {
            i: vis_k[i] + ((frontier[i][0],) if t else ())
            + vis_v[i] + ((frontier[i][1],) if t else ())
            for i in slab
        }
        x1, h_s, new_kv = _slab_column(
            model, h_dest[:, t : t + 1], slice(t, t + 1), span_rope, kv1,
            dest + 1, n_layers, source, ckpt,
        )
        if refresh is not None:
            for j, i in enumerate(slab):
                vis_k[i] = vis_k[i] + (refresh[j][0],)
                vis_v[i] = vis_v[i] + (refresh[j][1],)
        frontier = {i: new_kv[j] for j, i in enumerate(slab)}
        tops.append(x1)
        h_s_prev = h_s
    return torch.cat([h_init, model.model.norm(torch.cat(tops, dim=1))], dim=1)


def parallel_outputs(model, surface: Surface, ids: Tensor, span_start: int) -> Tensor:
    """Dots-alone forward: the same span drive through ALL layers, no
    wire. Returns supervised hiddens [B, k+1, d], post final-norm."""
    B, T = ids.shape
    P, k = span_start, T - span_start
    dtype = model.model.embed_tokens.weight.dtype
    scale = float(model.config.hidden_size) ** 0.5
    h_init, prompt_kv = _prompt_prefill(model, ids[:, :P])
    positions = torch.arange(P, T, device=ids.device)
    span_rope = _rope(model, positions, dtype)
    x = (surface.row.to(dtype) * scale).expand(B, k, -1)
    x = _span_parallel(
        model, x, span_rope, prompt_kv, 0, model.config.num_hidden_layers, True
    )
    return torch.cat([h_init, model.model.norm(x)], dim=1)


def _head_ce_chunk(W, row, dot_id, softcap, h, targets):
    logits = F.linear(h, W).float()
    dot = (h.float() @ row.float()).unsqueeze(-1)
    logits = logits.scatter(
        -1, torch.full_like(targets, dot_id).unsqueeze(-1), dot
    )
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).view_as(targets)


def head_ce(model, surface: Surface, hiddens: Tensor, targets: Tensor) -> Tensor:
    """Chunked, checkpointed CE over supervised positions. The `<t>`
    column comes from the live tied row; the rest from the frozen head."""
    W = model.lm_head.weight
    softcap = getattr(model.config, "final_logit_softcapping", None)
    B, S, _ = hiddens.shape
    chunk = max(1, 1024 // B)
    parts = []
    for i in range(0, S, chunk):
        parts.append(
            checkpoint(
                _head_ce_chunk, W, surface.row, surface.dot_id, softcap,
                hiddens[:, i : i + chunk], targets[:, i : i + chunk],
                use_reentrant=False,
            )
        )
    return torch.cat(parts, dim=1)


def span_loss(
    model, surface: Surface, hiddens: Tensor, answers: Tensor, lam: float
) -> tuple[Tensor, Tensor, Tensor]:
    """L = CE(answer) + lam * mean(CE(emission)). Emission targets are
    all `<t>` (initiation + continuation); k-independent task gradient."""
    B, S, _ = hiddens.shape
    targets = torch.full((B, S), surface.dot_id, device=hiddens.device)
    targets[:, -1] = answers
    ce = head_ce(model, surface, hiddens, targets)
    ce_ans, ce_emit = ce[:, -1].mean(), ce[:, :-1].mean()
    return ce_ans + lam * ce_emit, ce_ans, ce_emit


def answer_logits_from(model, surface: Surface, hiddens: Tensor) -> Tensor:
    lg = F.linear(hiddens[:, -1], model.lm_head.weight).float()
    lg[:, surface.dot_id] = hiddens[:, -1].float() @ surface.row.float()
    softcap = getattr(model.config, "final_logit_softcapping", None)
    return torch.tanh(lg / softcap) * softcap if softcap is not None else lg


class ThinkAdapter:
    """tasks.evaluate-compatible answer_logits for the think-scope arm."""

    def __init__(self, model, surface: Surface, source: int, dest: int):
        self.model, self.surface = model, surface
        self.source, self.dest = source, dest

    @torch.no_grad()
    def answer_logits(self, ids: Tensor) -> Tensor:
        span = (ids[0] == self.surface.dot_id).nonzero()
        start = int(span[0, 0]) if len(span) else ids.shape[1]
        if start == ids.shape[1]:
            raise ValueError("think-scope eval needs a dot span")
        h = think_outputs(
            self.model, self.surface, ids, start, self.source, self.dest, ckpt=False
        )
        return answer_logits_from(self.model, self.surface, h)


def reference_outputs(
    model, surface: Surface, ids: Tensor, span_start: int, source: int, dest: int
) -> Tensor:
    """Naive reference for the gradient gate: per-column sequential
    bottom slab (no parallel prefill), the same visibility rules
    re-derived, no checkpointing. Deliberately dumb and O(k^2)."""
    B, T = ids.shape
    P, k = span_start, T - span_start
    dtype = model.model.embed_tokens.weight.dtype
    scale = float(model.config.hidden_size) ** 0.5
    n_layers = model.config.num_hidden_layers
    h_init, prompt_kv = _prompt_prefill(model, ids[:, :P])
    positions = torch.arange(P, T, device=ids.device)
    rope = _rope(model, positions, dtype)
    e = (surface.row.to(dtype) * scale).expand(B, 1, -1)

    layer_types = model.config.layer_types
    slab = range(dest + 1, n_layers)
    bottom: dict[tuple[int, int], tuple] = {}  # (column, layer) -> (k, v)
    ref: dict[tuple[int, int], tuple] = {}  # refreshed (second-pass) KV
    first: dict[tuple[int, int], tuple] = {}  # first-pass KV
    h_dest_cols, h_s_cols, tops = [], [], []
    for t in range(k):
        # bottom slab, one column at a time (no parallel prefill)
        x = e
        for i in range(dest + 1):
            x, kn, vn = _layer_call(
                model.model.layers[i], False, x,
                *_pe(rope, layer_types, i, slice(t, t + 1)),
                prompt_kv[i][0], *(bottom[(c, i)][0] for c in range(t)),
                prompt_kv[i][1], *(bottom[(c, i)][1] for c in range(t)),
            )
            bottom[(t, i)] = (kn, vn)
        h_dest_cols.append(x)
        # second pass of column t-1: sees prompt + refreshed 0..t-2 + own
        if t > 0:
            alpha, beta = surface.gate(h_s_cols[t - 1], h_dest_cols[t - 1])
            x2 = _mix(1e-6, h_s_cols[t - 1], h_dest_cols[t - 1], alpha, beta)
            for i in slab:
                x2, kn, vn = _layer_call(
                    model.model.layers[i], False, x2,
                    *_pe(rope, layer_types, i, slice(t - 1, t)),
                    prompt_kv[i][0], *(ref[(c, i)][0] for c in range(t - 1)),
                    prompt_kv[i][1], *(ref[(c, i)][1] for c in range(t - 1)),
                )
                ref[(t - 1, i)] = (kn, vn)
        # first pass of column t: sees prompt + refreshed 0..t-2 +
        # column t-1 at FIRST-pass fidelity + own (same snapshot as the
        # second pass — visibility derived by column index, not order)
        x1, h_s = h_dest_cols[t], None
        for i in slab:
            side = (first[(t - 1, i)],) if t > 0 else ()
            x1, kn, vn = _layer_call(
                model.model.layers[i], False, x1,
                *_pe(rope, layer_types, i, slice(t, t + 1)),
                prompt_kv[i][0], *(ref[(c, i)][0] for c in range(t - 1)),
                *(s[0] for s in side),
                prompt_kv[i][1], *(ref[(c, i)][1] for c in range(t - 1)),
                *(s[1] for s in side),
            )
            first[(t, i)] = (kn, vn)
            if i == source:
                h_s = x1
        tops.append(x1)
        h_s_cols.append(h_s)
    return torch.cat([h_init, model.model.norm(torch.cat(tops, dim=1))], dim=1)


def _grads(loss: Tensor, surface: Surface) -> dict[str, Tensor]:
    names, params = zip(*surface.named_parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=False)
    return dict(zip(names, [g.detach().clone() for g in grads]))


def _grad_diff(a: dict[str, Tensor], b: dict[str, Tensor]) -> float:
    worst = 0.0
    for name in a:
        denom = b[name].abs().max().clamp_min(1e-12)
        worst = max(worst, float((a[name] - b[name]).abs().max() / denom))
    return worst


def gate_mode(args) -> None:
    """The D9 gradient gate + an HF cross-check of the span drive."""
    from .g0 import load_model

    tok, model = load_model(args.model, "cuda")
    dot_id = tasks._single(tok, tasks.DOT)
    surface = Surface(model, dot_id).cuda()
    inst = tasks.sample(tasks.TASKS["parity"], 2, 7, length=4)
    rows = [tasks.encode(tok, i, "dots", 3) for i in inst]
    ids = torch.tensor([e.ids for e in rows], device="cuda")
    answers = torch.tensor([e.answer for e in rows], device="cuda")
    start = rows[0].think[0]

    def functional():
        h = think_outputs(model, surface, ids, start, args.source, args.dest)
        return span_loss(model, surface, h, answers, args.lam)[0]

    def reference():
        h = reference_outputs(model, surface, ids, start, args.source, args.dest)
        return span_loss(model, surface, h, answers, args.lam)[0]

    l1, g1 = (lambda l: (float(l.detach()), _grads(l, surface)))(functional())
    l1b, g1b = (lambda l: (float(l.detach()), _grads(l, surface)))(functional())
    l2, g2 = (lambda l: (float(l.detach()), _grads(l, surface)))(reference())
    null = _grad_diff(g1, g1b)
    diff = _grad_diff(g1, g2)
    print(f"loss functional {l1:.6f} / rerun {l1b:.6f} / reference {l2:.6f}")
    print(f"grad rel-diff: null (same path twice) {null:.3e}, vs reference {diff:.3e}")

    surface.sync_into(model)
    with torch.no_grad():
        h = parallel_outputs(model, surface, ids, start)
        ours = answer_logits_from(model, surface, h)
        hf = model(ids, logits_to_keep=1).logits[:, -1].float()
    mean = float((ours - hf).abs().mean())
    top1 = float((ours.argmax(-1) == hf.argmax(-1)).float().mean())
    print(f"span-drive vs HF forward: mean|dlogit| {mean:.3e}, top1 {top1:.4f}")

    # Measured 2026-08-21 (B=2, parity len 4, k=3): rerun null 0 (bitwise
    # deterministic at this shape), loss rel-diff vs reference 6.9e-4,
    # grad max-rel 1.7e-2 — pure bf16 kernel-order noise (parallel vs
    # sequential bottom, different cat layouts). Semantic bugs (a wrong
    # visibility set) sit O(0.5+); thresholds live in the gap.
    ok = (
        abs(l1 - l2) / max(abs(l2), 1e-9) < 5e-3
        and diff < 0.1
        and mean < 0.15
        and top1 > 0.95
    )
    print("GRADIENT GATE", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


def evaluate_sweep(model, surface, tok, task, k_set, n, condition, source, dest, batch):
    surface.sync_into(model)
    adapter = (
        ThinkAdapter(model, surface, source, dest) if condition == "dots+wire" else None
    )
    accs = {}
    for k in k_set:
        inst = tasks.sample(tasks.TASKS[task], n, 0, **tasks.KNOBS[task])
        rows = [tasks.encode(tok, i, "dots", k) for i in inst]
        r = tasks.evaluate(model, rows, engine=adapter, batch=min(batch, n))
        accs[k] = (r["acc"], r["legal"])
    return accs


def run_mode(args) -> None:
    from .g0 import load_model

    tok, model = load_model(args.model, "cuda")
    dot_id = tasks._single(tok, tasks.DOT)
    surface = Surface(model, dot_id).cuda()
    opt = torch.optim.AdamW(surface.parameters(), lr=args.lr, weight_decay=0.0)
    task_list = args.tasks.split(",")
    k_set = [int(s) for s in args.k.split(",")]
    rng = random.Random(args.seed)
    print(f"condition {args.condition}, tasks {task_list}, k {k_set}, B {args.batch}")

    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        task = rng.choice(task_list)
        k = rng.choice(k_set)
        inst = tasks.sample(
            tasks.TASKS[task], args.batch, 1_000_000 + step, **tasks.KNOBS[task]
        )
        rows = [tasks.encode(tok, i, "dots", k) for i in inst]
        ids = torch.tensor([e.ids for e in rows], device="cuda")
        answers = torch.tensor([e.answer for e in rows], device="cuda")
        start = rows[0].think[0]
        if args.condition == "dots+wire":
            h = think_outputs(model, surface, ids, start, args.source, args.dest)
        else:  # dots
            h = parallel_outputs(model, surface, ids, start)
        loss, ce_ans, ce_emit = span_loss(model, surface, h, answers, args.lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for group in opt.param_groups:  # linear warmup, then flat (D6)
            group["lr"] = args.lr * min(step / max(args.warmup, 1), 1.0)
        opt.step()
        if step % args.log_every == 0:
            dt = time.perf_counter() - t0
            print(
                f"step {step:5d}  {task:12s} k={k:2d}  "
                f"loss {float(loss.detach()):.4f}  "
                f"answer {float(ce_ans.detach()):.4f}  "
                f"emit {float(ce_emit.detach()):.4f}  ({dt:.0f}s)"
            )
        if args.eval_every and step % args.eval_every == 0:
            for task_name in task_list:
                accs = evaluate_sweep(
                    model, surface, tok, task_name, k_set, args.eval_n,
                    args.condition, args.source, args.dest, args.batch,
                )
                row = "  ".join(f"k{k}:{a:.3f}/{l:.2f}" for k, (a, l) in accs.items())
                print(f"  eval {task_name:12s} acc/legal  {row}")
            torch.save(
                {"surface": surface.state_dict(), "args": vars(args)}, args.out
            )
    torch.save({"surface": surface.state_dict(), "args": vars(args)}, args.out)
    print(f"saved {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["gate", "run"])
    p.add_argument("--model", default="google/gemma-3-1b-pt")
    p.add_argument("--source", type=int, default=11)
    p.add_argument("--dest", type=int, default=4)
    p.add_argument("--condition", choices=["dots+wire", "dots"], default="dots+wire")
    p.add_argument("--tasks", default="parity")
    p.add_argument("--k", default="1,2,4,8,16,32")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--out", default="surface.pt")
    args = p.parse_args()
    if args.mode == "gate":
        gate_mode(args)
    else:
        run_mode(args)


if __name__ == "__main__":
    main()
