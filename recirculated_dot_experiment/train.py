"""Training path: think-scope BPTT through the wire (design.md D12).

Trainable surface (D2): the `<t>` embedding row (bf16, tied —
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
the refresh and first-pass branches share one packed projection and
one differentiable FA2-varlen call. KV views are assembled inside
checkpointed layer calls, so no prefix copy survives the forward.
Prompt, span, serial layers, and the frozen projections use the same
packed, regionally compiled tensor math.

Supervision (D12, a9's call): lm_head over the whole emission span —
the last prompt position and every dot target `<t>`, the last dot
targets the answer. With per-batch sampled k this teaches a stopping
hazard, activating D4's halting-by-sampling. Loss is
CE(answer) + lambda*mean(CE(emission)) so the task gradient is
k-independent; dot targets carry no task content, so credit for the
computation still flows only through the answer (H2 stays clean).
The head flattens each BF16 chunk to a 2-D GEMM and compiles CE with
one padded shape; logits for the `<t>` column are recomputed from the
live row (tied), everything else from the frozen head.

Gradient gate (D9, mandatory before any run): the functional path vs
a naive reference — per-column sequential bottom slab, separate
unbatched dual-pass calls, no checkpointing — compared on loss and on
every surface gradient, against the measured nondeterminism null
(flash-attn backward uses atomics; even the same path twice is not
bitwise). A deterministic nonzero output-weight gate activates the
otherwise-blocked hidden MLP gradients. Plus an HF cross-check: the parallel span drive with the
row synced into the model must match the plain forward's answer
logits within kernel noise.

Run mode freezes the base before graph construction, adapts activation
checkpointing to B*k, overlaps deterministic CPU sampling/tokenization
through pinned memory, warms every graph outside timing, and writes
atomic optimizer/RNG-complete checkpoints that resume by step.

Run: python -m recirculated_dot_experiment.train gate|run [flags]
"""

from __future__ import annotations

import argparse
import gc
import math
import random
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

from . import tasks
from .wire import pack_model_projections


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
        x = torch.cat([h_s, h_d], dim=-1)
        x = self.out(F.gelu(self.h2(F.gelu(self.h1(self.norm(x))))))
        alpha, beta = torch.sigmoid(x).chunk(2, dim=-1)
        return alpha, beta


def _gate_mix_math(
    cfg_eps: float,
    h_s: Tensor,
    h_d: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    h1_weight: Tensor,
    h1_bias: Tensor,
    h2_weight: Tensor,
    h2_bias: Tensor,
    out_weight: Tensor,
    out_bias: Tensor,
) -> Tensor:
    """Tensor-only adaptive gate plus norm-matched residual mix."""
    x = torch.cat((h_s, h_d), dim=-1)
    x = F.layer_norm(x, (norm_weight.shape[0],), norm_weight, norm_bias)
    x = F.gelu(F.linear(x, h1_weight, h1_bias))
    x = F.gelu(F.linear(x, h2_weight, h2_bias))
    alpha, beta = torch.sigmoid(F.linear(x, out_weight, out_bias)).chunk(2, dim=-1)
    ratio = h_d.norm(dim=-1, keepdim=True) / h_s.norm(dim=-1, keepdim=True).clamp_min(
        cfg_eps
    )
    return beta * h_d + alpha * ratio * h_s


def _gate_mix(gate: GateMLP, h_s: Tensor, h_d: Tensor, eps: float = 1e-6) -> Tensor:
    """Adaptive-alpha surface used by the functional train path."""
    return _gate_mix_math(
        eps,
        h_s,
        h_d,
        gate.norm.weight,
        gate.norm.bias,
        gate.h1.weight,
        gate.h1.bias,
        gate.h2.weight,
        gate.h2.bias,
        gate.out.weight,
        gate.out.bias,
    )


class Surface(nn.Module):
    """The entire trainable state: one tied row plus one bf16 gate."""

    def __init__(self, model, dot_id: int):
        super().__init__()
        d = model.config.hidden_size
        self.dot_id = dot_id
        weight = model.model.embed_tokens.weight
        if weight.dtype != torch.bfloat16:
            raise TypeError(f"training requires BF16 model weights, got {weight.dtype}")
        self.row = nn.Parameter(weight[dot_id].detach().clone())
        self.gate = GateMLP(d).to(device=weight.device, dtype=weight.dtype)

    def sync_into(self, model) -> None:
        """Write the trained row into the (tied) model embedding, so plain
        HF forwards — the dots-alone eval arm — see the trained surface."""
        w = model.model.embed_tokens.weight
        with torch.no_grad():
            w[self.dot_id] = self.row.to(w.dtype)


def _mix(cfg_eps: float, h_s: Tensor, h_d: Tensor, alpha, beta) -> Tensor:
    # Eq. 1 (norm-matched mix). Semantic twin of RecirculationEngine.mix
    # in wire.py — keep them in sync. All persistent arithmetic is bf16.
    ratio = h_d.norm(dim=-1, keepdim=True) / h_s.norm(dim=-1, keepdim=True).clamp_min(
        cfg_eps
    )
    return beta * h_d + alpha * ratio * h_s


def _rope(model, positions: Tensor, dtype) -> dict[str, tuple[Tensor, Tensor]]:
    marker = torch.empty(
        1, 1, model.config.hidden_size, dtype=dtype, device=positions.device
    )
    return {
        lt: model.model.rotary_emb(marker, positions.unsqueeze(0), lt)
        for lt in set(model.config.layer_types)
    }


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Gemma3 RMSNorm exactly; its mandated reduction accumulator is fp32."""
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return (y * (1.0 + weight.float())).to(dtype)


def _single_layer_math(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    k_past: Tensor,
    v_past: Tensor,
    qkv_weight: Tensor,
    o_weight: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    input_norm: Tensor,
    post_attn_norm: Tensor,
    pre_ff_norm: Tensor,
    post_ff_norm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    eps: float,
    q_eps: float,
    k_eps: float,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    causal: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Tensor-only packed Gemma layer, shared by all layers and positions."""
    residual = x
    h = _rms_norm(x, input_norm, eps)
    batch, span, _ = h.shape
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    qkv = F.linear(h, qkv_weight)
    q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
    q = q.view(batch, span, q_heads, head_dim).transpose(1, 2)
    k = k.view(batch, span, kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch, span, kv_heads, head_dim)
    q = _rms_norm(q, q_norm, q_eps)
    k = _rms_norm(k, k_norm, k_eps)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    q, k_new = q.transpose(1, 2), k.transpose(1, 2)
    k_all = torch.cat((k_past, k_new), dim=1)
    v_all = torch.cat((v_past, v), dim=1)
    attended = flash_attn_func(
        q, k_all, v_all, dropout_p=0.0, softmax_scale=scale, causal=causal
    )
    h = F.linear(attended.reshape(batch, span, -1), o_weight)
    h = residual + _rms_norm(h, post_attn_norm, eps)
    residual = h
    h = _rms_norm(h, pre_ff_norm, eps)
    gate, up = F.linear(h, gate_up_weight).chunk(2, dim=-1)
    h = F.linear(F.gelu(gate, approximate="tanh") * up, down_weight)
    return residual + _rms_norm(h, post_ff_norm, eps), k_new, v


def _dual_layer_math(
    refresh: Tensor,
    first: Tensor,
    cos_refresh: Tensor,
    sin_refresh: Tensor,
    cos_first: Tensor,
    sin_first: Tensor,
    shared_k: Tensor,
    shared_v: Tensor,
    frontier_k: Tensor,
    frontier_v: Tensor,
    qkv_weight: Tensor,
    o_weight: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    input_norm: Tensor,
    post_attn_norm: Tensor,
    pre_ff_norm: Tensor,
    post_ff_norm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    eps: float,
    q_eps: float,
    k_eps: float,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run refresh and first-pass branches in one packed FA2 varlen call."""
    batch = refresh.shape[0]
    x = torch.cat((refresh, first), dim=0)
    residual = x
    h = _rms_norm(x, input_norm, eps)
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    qkv = F.linear(h, qkv_weight)
    q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
    q = q.view(2 * batch, 1, q_heads, head_dim).transpose(1, 2)
    k = k.view(2 * batch, 1, kv_heads, head_dim).transpose(1, 2)
    v = v.view(2 * batch, 1, kv_heads, head_dim)
    q = _rms_norm(q, q_norm, q_eps)
    k = _rms_norm(k, k_norm, k_eps)
    cos = torch.cat(
        (cos_refresh.expand(batch, -1, -1), cos_first.expand(batch, -1, -1)),
        dim=0,
    )
    sin = torch.cat(
        (sin_refresh.expand(batch, -1, -1), sin_first.expand(batch, -1, -1)),
        dim=0,
    )
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    q = q.transpose(1, 2).squeeze(1)
    k_new = k.transpose(1, 2)
    k_refresh, k_first = k_new[:batch], k_new[batch:]
    v_refresh, v_first = v[:batch], v[batch:]

    refresh_k = torch.cat((shared_k, k_refresh), dim=1)
    refresh_v = torch.cat((shared_v, v_refresh), dim=1)
    first_k = torch.cat((shared_k, frontier_k, k_first), dim=1)
    first_v = torch.cat((shared_v, frontier_v, v_first), dim=1)
    refresh_len, first_len = refresh_k.shape[1], first_k.shape[1]
    cu_q = torch.arange(2 * batch + 1, device=x.device, dtype=torch.int32)
    cu_refresh = (
        torch.arange(batch + 1, device=x.device, dtype=torch.int32) * refresh_len
    )
    cu_first = cu_refresh[-1] + (
        torch.arange(1, batch + 1, device=x.device, dtype=torch.int32) * first_len
    )
    cu_k = torch.cat((cu_refresh, cu_first))
    packed_k = torch.cat((refresh_k.flatten(0, 1), first_k.flatten(0, 1)))
    packed_v = torch.cat((refresh_v.flatten(0, 1), first_v.flatten(0, 1)))
    attended = flash_attn_varlen_func(
        q,
        packed_k,
        packed_v,
        cu_q,
        cu_k,
        1,
        first_len,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
    ).unsqueeze(1)
    h = F.linear(attended.reshape(2 * batch, 1, -1), o_weight)
    h = residual + _rms_norm(h, post_attn_norm, eps)
    residual = h
    h = _rms_norm(h, pre_ff_norm, eps)
    gate, up = F.linear(h, gate_up_weight).chunk(2, dim=-1)
    h = F.linear(F.gelu(gate, approximate="tanh") * up, down_weight)
    h = residual + _rms_norm(h, post_ff_norm, eps)
    return h[:batch], h[batch:], k_refresh, v_refresh, k_first, v_first


_single_layer_c = torch.compile(_single_layer_math, fullgraph=True, dynamic=True)
_dual_layer_c = torch.compile(_dual_layer_math, fullgraph=True, dynamic=True)


def _finish_outputs_math(
    prompt_hidden: Tensor,
    dot_hidden: Tensor,
    norm_weight: Tensor,
    eps: float,
) -> Tensor:
    return torch.cat(
        (prompt_hidden, _rms_norm(dot_hidden, norm_weight, eps)),
        dim=1,
    )


def _dual_layer_from_pieces(
    refresh: Tensor,
    first: Tensor,
    cos_refresh: Tensor,
    sin_refresh: Tensor,
    cos_first: Tensor,
    sin_first: Tensor,
    frontier_k: Tensor,
    frontier_v: Tensor,
    *args,
):
    """Assemble functional KV inside checkpoint, so no prefix copy survives."""
    layer_args = args[:17]
    pieces = args[17:]
    half = len(pieces) // 2
    shared_k = torch.cat(pieces[:half], dim=1)
    shared_v = torch.cat(pieces[half:], dim=1)
    return _dual_layer_c(
        refresh,
        first,
        cos_refresh,
        sin_refresh,
        cos_first,
        sin_first,
        shared_k,
        shared_v,
        frontier_k,
        frontier_v,
        *layer_args,
    )


def _layer_args(model, packed, i: int) -> tuple:
    layer = model.model.layers[i]
    attn = layer.self_attn
    qkv_weights, qkv_biases, gate_up_weights = packed
    if qkv_biases[i] is not None:
        raise ValueError("the compiled training core requires bias-free QKV")
    return (
        qkv_weights[i],
        attn.o_proj.weight,
        gate_up_weights[i],
        layer.mlp.down_proj.weight,
        layer.input_layernorm.weight,
        layer.post_attention_layernorm.weight,
        layer.pre_feedforward_layernorm.weight,
        layer.post_feedforward_layernorm.weight,
        attn.q_norm.weight,
        attn.k_norm.weight,
        layer.input_layernorm.eps,
        attn.q_norm.eps,
        attn.k_norm.eps,
        model.config.num_attention_heads,
        model.config.num_key_value_heads,
        attn.head_dim,
        attn.scaling,
    )


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
def _prompt_prefill(model, packed, prompt_ids: Tensor):
    """Frozen packed/compiled prompt pass: top hidden plus per-layer KV."""
    batch, length = prompt_ids.shape
    dtype = model.model.embed_tokens.weight.dtype
    x = model.model.embed_tokens(prompt_ids)
    positions = torch.arange(length, device=prompt_ids.device)
    rope = _rope(model, positions, dtype)
    empty = torch.empty(
        batch,
        0,
        model.config.num_key_value_heads,
        model.config.head_dim,
        device=prompt_ids.device,
        dtype=dtype,
    )
    kv = []
    for i in range(model.config.num_hidden_layers):
        x, k, v = _single_layer_c(
            x,
            *_pe(rope, model.config.layer_types, i),
            empty,
            empty,
            *_layer_args(model, packed, i),
            True,
        )
        kv.append((k, v))
    return _rms_norm(x[:, -1:], model.model.norm.weight, model.model.norm.eps), kv


@dataclass
class _PromptCacheEntry:
    packed: Tensor
    shapes: tuple[torch.Size, ...]
    ready: torch.cuda.Event
    nbytes: int


class PromptStateCache:
    """Pinned-host LRU for frozen prompt hiddens and per-layer KV.

    The prompt excludes `<t>`, so its state is invariant while the surface
    trains. A dedicated stream overlaps the one-time D2H fill with the serial
    dot computation; later evaluations copy the compact BF16 state back.
    """

    def __init__(self, model, max_bytes: int = 8 * 2**30):
        self.model = model
        self.max_bytes = max_bytes
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.entries: OrderedDict[tuple, _PromptCacheEntry] = OrderedDict()
        self.stream = torch.cuda.Stream(device=next(model.parameters()).device)

    @staticmethod
    def _flatten(state) -> tuple[Tensor, ...]:
        hidden, kv = state
        return (hidden, *(tensor for pair in kv for tensor in pair))

    @staticmethod
    def _unflatten(tensors: tuple[Tensor, ...]):
        return tensors[0], list(zip(tensors[1::2], tensors[2::2]))

    @classmethod
    def _views(cls, packed: Tensor, shapes: tuple[torch.Size, ...]):
        tensors = []
        offset = 0
        for shape in shapes:
            size = math.prod(shape)
            tensors.append(packed.narrow(0, offset, size).view(shape))
            offset += size
        return cls._unflatten(tuple(tensors))

    def _key(self, ids: Tensor, start: int) -> tuple:
        prompt = ids[:, :start]
        # Evaluation input construction starts on CPU, but the adapter API is
        # device-facing. This small ID copy is negligible beside prompt KV.
        return prompt.shape, tuple(prompt.reshape(-1).tolist())

    @torch.no_grad()
    def get(self, ids: Tensor, start: int):
        key = self._key(ids, start)
        entry = self.entries.pop(key, None)
        current = torch.cuda.current_stream(ids.device)
        if entry is not None:
            self.hits += 1
            self.entries[key] = entry
            current.wait_event(entry.ready)
            packed = entry.packed.to(ids.device, non_blocking=True)
            return self._views(packed, entry.shapes)

        self.misses += 1
        state = _prompt_prefill(
            self.model,
            pack_model_projections(self.model),
            ids[:, :start],
        )
        source = self._flatten(state)
        shapes = tuple(tensor.shape for tensor in source)
        sizes = tuple(tensor.numel() for tensor in source)
        host = torch.empty(
            sum(sizes),
            dtype=source[0].dtype,
            device="cpu",
            pin_memory=True,
        )
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(current)
            offset = 0
            for size, tensor in zip(sizes, source):
                host.narrow(0, offset, size).copy_(
                    tensor.reshape(-1), non_blocking=True
                )
                tensor.record_stream(self.stream)
                offset += size
            ready = self.stream.record_event()
        nbytes = host.numel() * host.element_size()
        self.entries[key] = _PromptCacheEntry(host, shapes, ready, nbytes)
        self.bytes += nbytes
        while self.bytes > self.max_bytes and len(self.entries) > 1:
            _, evicted = self.entries.popitem(last=False)
            evicted.ready.synchronize()
            self.bytes -= evicted.nbytes
        return state


def _pe(rope, layer_types, i, sl=None):
    cos, sin = rope[layer_types[i]]
    return (cos, sin) if sl is None else (cos[:, sl], sin[:, sl])


def _span_parallel(model, packed, x, span_rope, prompt_kv, lo, hi, ckpt):
    """Drive layers lo..hi-1 over the whole span in parallel (causal,
    prompt as prefix). Exact for any layer range that never sees
    refreshed state: 0..dest under the wire, the full stack without it."""
    layer_types = model.config.layer_types
    for i in range(lo, hi):
        args = (
            x,
            *_pe(span_rope, layer_types, i),
            *prompt_kv[i],
            *_layer_args(model, packed, i),
            True,
        )
        if _checkpoint_layer(ckpt, i):
            x, _, _ = checkpoint(
                _single_layer_c,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            x, _, _ = _single_layer_c(*args)
    return x


def _slab_first(model, packed, x, pe_sl, rope, shared, lo, hi, source, ckpt):
    """The unique first column, before a refresh/first pair exists."""
    layer_types = model.config.layer_types
    h_s, new_kv = None, []
    for i in range(lo, hi):
        args = (
            x,
            *_pe(rope, layer_types, i, pe_sl),
            *shared[i],
            *_layer_args(model, packed, i),
            False,
        )
        if _checkpoint_layer(ckpt, i):
            x, k, v = checkpoint(
                _single_layer_c,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            x, k, v = _single_layer_c(*args)
        new_kv.append((k, v))
        if i == source:
            h_s = x
    return x, h_s, new_kv


def _slab_pair(
    model,
    packed,
    refresh,
    first,
    refresh_sl,
    first_sl,
    rope,
    shared,
    frontier,
    lo,
    hi,
    source,
    ckpt,
):
    """One differentiably batched refresh/first pair through the slab."""
    layer_types = model.config.layer_types
    h_s, refresh_kv, first_kv = None, [], []
    for i in range(lo, hi):
        cos_refresh, sin_refresh = _pe(rope, layer_types, i, refresh_sl)
        cos_first, sin_first = _pe(rope, layer_types, i, first_sl)
        args = (
            refresh,
            first,
            cos_refresh,
            sin_refresh,
            cos_first,
            sin_first,
            *frontier[i],
            *_layer_args(model, packed, i),
            *shared[i][0],
            *shared[i][1],
        )
        if _checkpoint_layer(ckpt, i):
            out = checkpoint(
                _dual_layer_from_pieces,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            out = _dual_layer_from_pieces(*args)
        refresh, first, kr, vr, kf, vf = out
        refresh_kv.append((kr, vr))
        first_kv.append((kf, vf))
        if i == source:
            h_s = first
    return refresh, first, h_s, refresh_kv, first_kv


def think_outputs(
    model,
    surface: Surface,
    ids: Tensor,
    span_start: int,
    source: int,
    dest: int,
    ckpt: bool = True,
    prompt_state=None,
) -> Tensor:
    """Think-scope wire forward. Returns supervised hiddens [B, k+1, d]
    (post final-norm): last prompt position, then every dot top."""
    B, T = ids.shape
    P, k = span_start, T - span_start
    dtype = model.model.embed_tokens.weight.dtype
    packed = pack_model_projections(model)
    scale = float(model.config.hidden_size) ** 0.5
    h_init, prompt_kv = (
        _prompt_prefill(model, packed, ids[:, :P])
        if prompt_state is None
        else prompt_state
    )

    positions = torch.arange(P, T, device=ids.device)
    span_rope = _rope(model, positions, dtype)
    x = (surface.row.to(dtype) * scale).expand(B, k, -1)
    h_dest = _span_parallel(model, packed, x, span_rope, prompt_kv, 0, dest + 1, ckpt)

    n_layers = model.config.num_hidden_layers
    slab = range(dest + 1, n_layers)
    # Functional cache: per slab layer, shared visible K/V (prompt plus
    # refreshed columns 0..t-2) and the previous first-pass frontier.
    # Same-snapshot rule: the second pass
    # of t-1 and the first pass of t read the SAME visible past — the
    # refresh joins it only after both have run (the wire's snapshot
    # boundary), and the first pass sees column t-1 only through the
    # frontier, at first-pass fidelity.
    shared = {i: ((prompt_kv[i][0],), (prompt_kv[i][1],)) for i in slab}
    frontier: dict[int, tuple[Tensor, Tensor]] = {}
    tops, h_s_prev = [], None
    x1, h_s_prev, new_kv = _slab_first(
        model,
        packed,
        h_dest[:, :1],
        slice(0, 1),
        span_rope,
        {i: (shared[i][0][0], shared[i][1][0]) for i in slab},
        dest + 1,
        n_layers,
        source,
        ckpt,
    )
    frontier = {i: new_kv[j] for j, i in enumerate(slab)}
    tops.append(x1)
    for t in range(1, k):
        refresh = _gate_mix(surface.gate, h_s_prev, h_dest[:, t - 1 : t])
        _, x1, h_s, refresh_kv, first_kv = _slab_pair(
            model,
            packed,
            refresh,
            h_dest[:, t : t + 1],
            slice(t - 1, t),
            slice(t, t + 1),
            span_rope,
            shared,
            frontier,
            dest + 1,
            n_layers,
            source,
            ckpt,
        )
        shared = {
            i: (
                (*shared[i][0], refresh_kv[j][0]),
                (*shared[i][1], refresh_kv[j][1]),
            )
            for j, i in enumerate(slab)
        }
        frontier = {i: first_kv[j] for j, i in enumerate(slab)}
        tops.append(x1)
        h_s_prev = h_s
    return _finish_outputs_math(
        h_init,
        torch.cat(tops, dim=1),
        model.model.norm.weight,
        model.model.norm.eps,
    )


def parallel_outputs(
    model,
    surface: Surface,
    ids: Tensor,
    span_start: int,
    ckpt: bool = True,
    prompt_state=None,
) -> Tensor:
    """Dots-alone forward: the same span drive through ALL layers, no
    wire. Returns supervised hiddens [B, k+1, d], post final-norm."""
    B, T = ids.shape
    P, k = span_start, T - span_start
    dtype = model.model.embed_tokens.weight.dtype
    packed = pack_model_projections(model)
    scale = float(model.config.hidden_size) ** 0.5
    h_init, prompt_kv = (
        _prompt_prefill(model, packed, ids[:, :P])
        if prompt_state is None
        else prompt_state
    )
    positions = torch.arange(P, T, device=ids.device)
    span_rope = _rope(model, positions, dtype)
    x = (surface.row.to(dtype) * scale).expand(B, k, -1)
    x = _span_parallel(
        model,
        packed,
        x,
        span_rope,
        prompt_kv,
        0,
        model.config.num_hidden_layers,
        ckpt,
    )
    return _finish_outputs_math(
        h_init,
        x,
        model.model.norm.weight,
        model.model.norm.eps,
    )


def _head_ce_chunk(W, row, dot_id, softcap, h, targets):
    batch, span, width = h.shape
    flat = h.reshape(batch * span, width)
    logits = F.linear(flat, W)
    dot = F.linear(flat, row.unsqueeze(0))
    dot_column = torch.full(
        (batch * span, 1), dot_id, dtype=torch.long, device=h.device
    )
    logits = logits.scatter(-1, dot_column, dot)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return F.cross_entropy(logits, targets.flatten(), reduction="none").view_as(targets)


_head_ce_chunk_c = torch.compile(_head_ce_chunk, fullgraph=True, dynamic=False)


def head_ce(model, surface: Surface, hiddens: Tensor, targets: Tensor) -> Tensor:
    """One-shape compiled CE chunks with a live tied `<t>` column."""
    W = model.lm_head.weight
    softcap = getattr(model.config, "final_logit_softcapping", None)
    B, S, _ = hiddens.shape
    # A 512-row full-vocab slab leaves headroom at the B=256,k=32 gate;
    # one compiled shape is reused and the final slab is scratch-padded.
    chunk = max(1, 512 // B)
    padded = ((S + chunk - 1) // chunk) * chunk
    if padded != S:
        hiddens = torch.cat(
            (hiddens, hiddens[:, -1:].expand(-1, padded - S, -1)), dim=1
        )
        targets = torch.cat(
            (
                targets,
                torch.full(
                    (B, padded - S),
                    surface.dot_id,
                    device=targets.device,
                    dtype=targets.dtype,
                ),
            ),
            dim=1,
        )
    parts = []
    for i in range(0, padded, chunk):
        parts.append(
            _head_ce_chunk_c(
                W,
                surface.row,
                surface.dot_id,
                softcap,
                hiddens[:, i : i + chunk],
                targets[:, i : i + chunk],
            )
        )
    return torch.cat(parts, dim=1)[:, :S]


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
    hidden = hiddens[:, -1]
    logits = F.linear(hidden, model.lm_head.weight)
    dot = F.linear(hidden, surface.row.unsqueeze(0))
    column = torch.full(
        (hidden.shape[0], 1), surface.dot_id, dtype=torch.long, device=hidden.device
    )
    logits = logits.scatter(-1, column, dot)
    softcap = getattr(model.config, "final_logit_softcapping", None)
    return torch.tanh(logits / softcap) * softcap if softcap is not None else logits


class ThinkAdapter:
    """tasks.evaluate-compatible answer_logits for the think-scope arm."""

    def __init__(
        self,
        model,
        surface: Surface,
        source: int,
        dest: int,
        prompt_cache: PromptStateCache | None = None,
    ):
        self.model, self.surface = model, surface
        self.source, self.dest = source, dest
        self.prompt_cache = prompt_cache or PromptStateCache(model)

    @torch.no_grad()
    def answer_hiddens(self, ids: Tensor, ks: list[int]) -> Tensor:
        span = (ids[0] == self.surface.dot_id).nonzero()
        start = int(span[0, 0]) if len(span) else ids.shape[1]
        if start == ids.shape[1]:
            raise ValueError("think-scope eval needs a dot span")
        prompt_state = self.prompt_cache.get(ids, start)
        h = think_outputs(
            self.model,
            self.surface,
            ids,
            start,
            self.source,
            self.dest,
            ckpt=False,
            prompt_state=prompt_state,
        )
        positions = torch.tensor(ks, device=ids.device, dtype=torch.long)
        return h.index_select(1, positions)

    @torch.no_grad()
    def logits_from_hidden(self, hidden: Tensor) -> Tensor:
        return answer_logits_from(self.model, self.surface, hidden.unsqueeze(1))

    @torch.no_grad()
    def answer_logits(self, ids: Tensor) -> Tensor:
        span = (ids[0] == self.surface.dot_id).nonzero()
        start = int(span[0, 0]) if len(span) else ids.shape[1]
        k = ids.shape[1] - start
        hidden = self.answer_hiddens(ids, [k])[:, 0]
        return self.logits_from_hidden(hidden)


class DotsAdapter:
    """The matched no-wire span runner with the identical BF16 readout."""

    def __init__(
        self,
        model,
        surface: Surface,
        prompt_cache: PromptStateCache | None = None,
    ):
        self.model, self.surface = model, surface
        self.prompt_cache = prompt_cache or PromptStateCache(model)

    @torch.no_grad()
    def answer_hiddens(self, ids: Tensor, ks: list[int]) -> Tensor:
        span = (ids[0] == self.surface.dot_id).nonzero()
        start = int(span[0, 0]) if len(span) else ids.shape[1]
        if start == ids.shape[1]:
            raise ValueError("dots eval needs a dot span")
        prompt_state = self.prompt_cache.get(ids, start)
        h = parallel_outputs(
            self.model,
            self.surface,
            ids,
            start,
            ckpt=False,
            prompt_state=prompt_state,
        )
        positions = torch.tensor(ks, device=ids.device, dtype=torch.long)
        return h.index_select(1, positions)

    @torch.no_grad()
    def logits_from_hidden(self, hidden: Tensor) -> Tensor:
        return answer_logits_from(self.model, self.surface, hidden.unsqueeze(1))

    @torch.no_grad()
    def answer_logits(self, ids: Tensor) -> Tensor:
        span = (ids[0] == self.surface.dot_id).nonzero()
        start = int(span[0, 0]) if len(span) else ids.shape[1]
        k = ids.shape[1] - start
        hidden = self.answer_hiddens(ids, [k])[:, 0]
        return self.logits_from_hidden(hidden)


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
    packed = pack_model_projections(model)
    h_init, prompt_kv = _prompt_prefill(model, packed, ids[:, :P])
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
                model.model.layers[i],
                False,
                x,
                *_pe(rope, layer_types, i, slice(t, t + 1)),
                prompt_kv[i][0],
                *(bottom[(c, i)][0] for c in range(t)),
                prompt_kv[i][1],
                *(bottom[(c, i)][1] for c in range(t)),
            )
            bottom[(t, i)] = (kn, vn)
        h_dest_cols.append(x)
        # second pass of column t-1: sees prompt + refreshed 0..t-2 + own
        if t > 0:
            alpha, beta = surface.gate(h_s_cols[t - 1], h_dest_cols[t - 1])
            x2 = _mix(1e-6, h_s_cols[t - 1], h_dest_cols[t - 1], alpha, beta)
            for i in slab:
                x2, kn, vn = _layer_call(
                    model.model.layers[i],
                    False,
                    x2,
                    *_pe(rope, layer_types, i, slice(t - 1, t)),
                    prompt_kv[i][0],
                    *(ref[(c, i)][0] for c in range(t - 1)),
                    prompt_kv[i][1],
                    *(ref[(c, i)][1] for c in range(t - 1)),
                )
                ref[(t - 1, i)] = (kn, vn)
        # first pass of column t: sees prompt + refreshed 0..t-2 +
        # column t-1 at FIRST-pass fidelity + own (same snapshot as the
        # second pass — visibility derived by column index, not order)
        x1, h_s = h_dest_cols[t], None
        for i in slab:
            side = (first[(t - 1, i)],) if t > 0 else ()
            x1, kn, vn = _layer_call(
                model.model.layers[i],
                False,
                x1,
                *_pe(rope, layer_types, i, slice(t, t + 1)),
                prompt_kv[i][0],
                *(ref[(c, i)][0] for c in range(t - 1)),
                *(s[0] for s in side),
                prompt_kv[i][1],
                *(ref[(c, i)][1] for c in range(t - 1)),
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
    for name, value in a.items():
        denom = b[name].abs().max().clamp_min(1e-12)
        worst = max(worst, float((value - b[name]).abs().max() / denom))
    return worst


def _loss_and_grads(loss_fn, surface: Surface) -> tuple[float, dict[str, Tensor]]:
    loss = loss_fn()
    return float(loss.detach()), _grads(loss, surface)


CheckpointSpec = bool | frozenset[int]


def _checkpoint_layer(spec: CheckpointSpec, layer: int) -> bool:
    return spec if isinstance(spec, bool) else layer in spec


@dataclass(frozen=True)
class ExecutionPlan:
    """One optimizer batch expressed as equal-shape CUDA microbatches."""

    microbatch: int
    checkpoint_layers: frozenset[int]


def _largest_divisor_at_most(value: int, limit: int) -> int:
    """Keep every microbatch shape identical so it reuses one compilation."""
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    raise AssertionError("one always divides a positive batch")


def _execution_plan(
    batch: int,
    k: int,
    n_layers: int,
) -> ExecutionPlan:
    """Select physical batch and checkpoint layers from measured 4090 knees.

    ``batch`` remains the optimizer/effective batch. Short spans retain
    activations in equal B*k<=2048 microbatches; long spans run the full
    batch with recomputation because the larger GEMMs recover more throughput.
    """
    if batch < 1 or k < 1:
        raise ValueError(f"batch and k must be positive, got B={batch}, k={k}")
    layers = frozenset(range(n_layers))
    if batch * k <= 2048:
        return ExecutionPlan(batch, frozenset())
    if k <= 8:
        microbatch = _largest_divisor_at_most(batch, max(1, 2048 // k))
        return ExecutionPlan(microbatch, frozenset())
    if batch == 512 and k == 16 and n_layers == 26:
        # Four evenly spaced recurrent layers fit at 18.00 GiB and avoid
        # their 15 repeated recomputations (1.9% measured throughput gain).
        return ExecutionPlan(batch, layers - {7, 13, 19, 25})
    return ExecutionPlan(batch, layers)


def gate_mode(args) -> None:
    """The D9 gradient gate + an HF cross-check of the span drive."""
    from .wire import load_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tok, model = load_model(args.model, "cuda")
    dot_id = tasks.single_token(tok, tasks.DOT)
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

    l1, g1 = _loss_and_grads(functional, surface)
    l1b, g1b = _loss_and_grads(functional, surface)
    l2, g2 = _loss_and_grads(reference, surface)
    null = _grad_diff(g1, g1b)
    diff = _grad_diff(g1, g2)
    print(f"loss functional {l1:.6f} / rerun {l1b:.6f} / reference {l2:.6f}")
    print(f"grad rel-diff: null (same path twice) {null:.3e}, vs reference {diff:.3e}")

    # The paper initialization deliberately zeros gate.out.weight, which
    # blocks gradients into norm/h1/h2. A second deterministic perturbation
    # activates the complete MLP so the gate checks every trainable tensor.
    with torch.no_grad():
        generator = torch.Generator(device=surface.row.device).manual_seed(1729)
        surface.gate.out.weight.normal_(
            std=1e-3,
            generator=generator,
        )
    lp1, gp1 = _loss_and_grads(functional, surface)
    lp2, gp2 = _loss_and_grads(reference, surface)
    perturbed_diff = _grad_diff(gp1, gp2)
    print(
        f"perturbed-gate loss functional {lp1:.6f} / reference {lp2:.6f}; "
        f"grad rel-diff {perturbed_diff:.3e}"
    )

    surface.sync_into(model)
    with torch.no_grad():
        h = parallel_outputs(model, surface, ids, start)
        ours = answer_logits_from(model, surface, h)
        hf = model(ids, logits_to_keep=1).logits[:, -1]
    mean = float((ours - hf).abs().mean())
    top1 = float((ours.argmax(-1) == hf.argmax(-1)).to(torch.bfloat16).mean())
    print(f"span-drive vs HF forward: mean|dlogit| {mean:.3e}, top1 {top1:.4f}")

    # Measured 2026-08-21 (B=2, parity len 4, k=3): rerun null 0 (bitwise
    # deterministic at this shape), loss rel-diff vs reference 6.9e-4,
    # grad max-rel 1.7e-2 — pure bf16 kernel-order noise (parallel vs
    # sequential bottom, different cat layouts). Semantic bugs (a wrong
    # visibility set) sit O(0.5+); thresholds live in the gap.
    ok = (
        abs(l1 - l2) / max(abs(l2), 1e-9) < 5e-3
        and diff < 0.1
        and abs(lp1 - lp2) / max(abs(lp2), 1e-9) < 5e-3
        and perturbed_diff < 0.1
        and mean < 0.15
        and top1 > 0.95
    )
    print("GRADIENT GATE", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


def evaluate_sweep(
    model,
    surface,
    tok,
    task,
    k_set,
    n,
    condition,
    source,
    dest,
    batch,
    *,
    runner=None,
    rows=None,
):
    surface.sync_into(model)
    if runner is None:
        runner = (
            ThinkAdapter(model, surface, source, dest)
            if condition == "dots+wire"
            else DotsAdapter(model, surface)
        )
    if rows is None:
        instances = tasks.sample(tasks.TASKS[task], n, 0, **tasks.KNOBS[task])
        rows = tasks.encode_dot_sweep(tok, instances, k_set)
    results = tasks.evaluate_dot_sweep(model, rows, runner, min(batch, n))
    return {k: (result["acc"], result["legal"]) for k, result in results.items()}


def _lr_at(step: int, args) -> float:
    """Learning rate as a pure function of the within-run step (D15).

    Linear warmup to the peak, one cosine period down to the floor, then
    flat at the floor. The period is independent of --steps: a run may
    end mid-cosine or coast on the floor, and curriculum stages reset
    schedules by being separate runs (D6). --cosine 0 is the flat recipe.
    """
    if step < args.warmup:
        return args.lr * step / max(args.warmup, 1)
    if not args.cosine:
        return args.lr
    t = min((step - args.warmup) / args.cosine, 1.0)
    return args.lr_floor + (args.lr - args.lr_floor) * 0.5 * (1 + math.cos(math.pi * t))


def _step_choice(
    seed: int,
    step: int,
    task_list: list[str],
    k_set: list[int],
    k_weights: list[float] | None = None,
):
    """A random-looking schedule addressable by step for exact resume."""
    rng = random.Random((seed << 32) ^ step)
    return rng.choice(task_list), rng.choices(k_set, weights=k_weights)[0]


def _make_cpu_batch(tok, args, task_list, k_set, step: int, k_weights=None):
    task, k = _step_choice(args.seed, step, task_list, k_set, k_weights)
    sample_seed = 1_000_000 + args.seed * 10_000_000 + step
    instances = tasks.sample(
        tasks.TASKS[task], args.batch, sample_seed, **tasks.KNOBS[task]
    )
    rows = [tasks.encode(tok, instance, "dots", k) for instance in instances]
    ids = torch.tensor([row.ids for row in rows], pin_memory=True)
    answers = torch.tensor([row.answer for row in rows], pin_memory=True)
    return task, k, ids, answers, rows[0].think[0]


class _BatchProducer:
    """Bounded one-worker tokenizer/generator pipeline."""

    def __init__(self, tok, args, task_list, k_set, start: int, k_weights=None):
        self.tok, self.args = tok, args
        self.task_list, self.k_set = task_list, k_set
        self.k_weights = k_weights
        self.depth = max(args.prefetch, 1)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dot-batches")
        self.futures: dict[int, Future] = {}
        for step in range(start, min(args.steps + 1, start + self.depth)):
            self._submit(step)

    def _submit(self, step: int) -> None:
        if step <= self.args.steps and step not in self.futures:
            self.futures[step] = self.pool.submit(
                _make_cpu_batch,
                self.tok,
                self.args,
                self.task_list,
                self.k_set,
                step,
                self.k_weights,
            )

    def get(self, step: int):
        self._submit(step)
        result = self.futures.pop(step).result()
        self._submit(step + self.depth)
        return result

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)


def _save_checkpoint(path: str, surface, opt, args, step: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "version": 1,
        "step": step,
        "surface": surface.state_dict(),
        "optimizer": opt.state_dict(),
        "args": vars(args),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    torch.save(payload, temporary)
    temporary.replace(target)


def _load_checkpoint(path: str, surface, opt) -> int:
    checkpoint_state = torch.load(path, map_location="cuda", weights_only=False)
    if checkpoint_state.get("version") != 1:
        raise ValueError(f"unsupported checkpoint version in {path}")
    wrong_dtype = {
        name: tensor.dtype
        for name, tensor in checkpoint_state["surface"].items()
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16
    }
    if wrong_dtype:
        raise TypeError(f"checkpoint surface is not wholly BF16: {wrong_dtype}")
    surface.load_state_dict(checkpoint_state["surface"], strict=True)
    opt.load_state_dict(checkpoint_state["optimizer"])
    random.setstate(checkpoint_state["python_rng"])
    torch.set_rng_state(checkpoint_state["torch_rng"].cpu())
    torch.cuda.set_rng_state_all(
        [state.cpu() for state in checkpoint_state["cuda_rng"]]
    )
    return int(checkpoint_state["step"])


def _forward_loss(model, surface, ids, answers, start, args, ckpt):
    if args.condition == "dots+wire":
        h = think_outputs(model, surface, ids, start, args.source, args.dest, ckpt=ckpt)
    else:
        h = parallel_outputs(model, surface, ids, start, ckpt=ckpt)
    return span_loss(model, surface, h, answers, args.lam)


def _run_microbatches(
    model,
    surface,
    ids_cpu,
    answers_cpu,
    span_start,
    args,
    plan: ExecutionPlan,
    *,
    accumulate: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run one effective batch through equal compiled CUDA shapes.

    Gradients and reported losses are weighted by row count, making a
    microbatched step identical to the effective-batch mean objective.
    """
    total = ids_cpu.shape[0]
    if total % plan.microbatch:
        raise ValueError(
            f"effective batch {total} is not divisible by B={plan.microbatch}"
        )
    params = tuple(surface.parameters())
    metrics = None
    for offset in range(0, total, plan.microbatch):
        stop = offset + plan.microbatch
        ids = ids_cpu[offset:stop].to("cuda", non_blocking=True)
        answers = answers_cpu[offset:stop].to("cuda", non_blocking=True)
        values = _forward_loss(
            model,
            surface,
            ids,
            answers,
            span_start,
            args,
            plan.checkpoint_layers,
        )
        fraction = plan.microbatch / total
        scaled_loss = values[0] * fraction
        if accumulate:
            scaled_loss.backward()
        else:
            torch.autograd.grad(scaled_loss, params, allow_unused=True)
        weighted = tuple(value.detach() * fraction for value in values)
        metrics = (
            weighted
            if metrics is None
            else tuple(old + new for old, new in zip(metrics, weighted))
        )
    assert metrics is not None
    return metrics


def run_mode(args) -> None:
    from .wire import load_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tok, model = load_model(args.model, "cuda")
    dot_id = tasks.single_token(tok, tasks.DOT)
    surface = Surface(model, dot_id).cuda()
    opt = torch.optim.AdamW(surface.parameters(), lr=args.lr, weight_decay=0.0)
    task_list = args.tasks.split(",")
    k_set = [int(s) for s in args.k.split(",")]
    train_k = [int(s) for s in args.train_k.split(",")]
    k_weights = [k**args.k_gamma for k in train_k]
    eval_runner = None
    eval_rows = {}
    if args.eval_every:
        prompt_cache = PromptStateCache(model)
        eval_runner = (
            ThinkAdapter(model, surface, args.source, args.dest, prompt_cache)
            if args.condition == "dots+wire"
            else DotsAdapter(model, surface, prompt_cache)
        )
        for task_name in task_list:
            instances = tasks.sample(
                tasks.TASKS[task_name],
                args.eval_n,
                0,
                **tasks.KNOBS[task_name],
            )
            eval_rows[task_name] = tasks.encode_dot_sweep(tok, instances, k_set)
    start_step = 1
    if args.resume is not None:
        resume_path = args.out if args.resume == "auto" else args.resume
        start_step = _load_checkpoint(resume_path, surface, opt) + 1
        print(f"resumed {resume_path} at step {start_step - 1}")
    print(
        f"condition {args.condition}, tasks {task_list}, eval k {k_set}, "
        f"train k {train_k} (P~k^{args.k_gamma:g}), B {args.batch}, "
        f"lr {args.lr:g} cosine {args.cosine} floor {args.lr_floor:g}"
    )

    producer = _BatchProducer(tok, args, task_list, train_k, start_step, k_weights)
    completed_step = start_step - 1
    try:
        # Warm every training k before the experiment clock: each k is
        # shape-distinct somewhere (span length, the execution plan's
        # microbatch, the dynamic=False head-CE chunk — a [max,1,2,4]
        # heuristic let k=8's B=256 chunk escape into a timed step), plus
        # every prompt family. Eval-only ks (e.g. k=1) warm through the
        # eval sweep below. Largest k first (cache-backing canon). No
        # optimizer update occurs; the timed loop rejects escaped compiles.
        warm_args = argparse.Namespace(**vars(args))
        warm_args.seed = args.seed + 97_531
        warm_args.batch = args.batch
        structural = sorted(set(train_k), reverse=True)
        warm_cases = [(task_list[0], k) for k in structural]
        warm_cases.extend((task, structural[-1]) for task in task_list)
        warm_cases = list(dict.fromkeys(warm_cases))
        compile_start = time.perf_counter()
        for warm_index, (warm_task, warm_k) in enumerate(warm_cases):
            _, _, ids_cpu, answers_cpu, span_start = _make_cpu_batch(
                tok, warm_args, [warm_task], [warm_k], warm_index
            )
            warm_plan = _execution_plan(
                args.batch,
                warm_k,
                model.config.num_hidden_layers,
            )
            warm_loss = _run_microbatches(
                model,
                surface,
                ids_cpu,
                answers_cpu,
                span_start,
                args,
                warm_plan,
                accumulate=False,
            )[0]
        if args.eval_every:
            for task_name in task_list:
                # Miss once to fill the pinned-host cache, then hit once to
                # compile its packed-view stride/layout. Both are untimed.
                for _ in range(2):
                    evaluate_sweep(
                        model,
                        surface,
                        tok,
                        task_name,
                        k_set,
                        args.eval_n,
                        args.condition,
                        args.source,
                        args.dest,
                        args.batch,
                        runner=eval_runner,
                        rows=eval_rows[task_name],
                    )
        torch.cuda.synchronize()
        compiled_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        print(
            f"compile/warmup {time.perf_counter() - compile_start:.1f}s "
            f"({compiled_graphs} graphs, not timed)"
        )
        del warm_loss, ids_cpu, answers_cpu
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        for step in range(start_step, args.steps + 1):
            task, k, ids_cpu, answers_cpu, span_start = producer.get(step)
            opt.zero_grad(set_to_none=True)
            plan = _execution_plan(args.batch, k, model.config.num_hidden_layers)
            loss, ce_ans, ce_emit = _run_microbatches(
                model,
                surface,
                ids_cpu,
                answers_cpu,
                span_start,
                args,
                plan,
                accumulate=True,
            )
            current_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            if current_graphs != compiled_graphs:
                raise RuntimeError(
                    "a compile escaped warmup: "
                    f"{compiled_graphs} -> {current_graphs} unique graphs"
                )
            if step == start_step and any(
                parameter.grad is not None for parameter in model.parameters()
            ):
                raise RuntimeError("frozen base model accumulated a gradient")
            lr_now = _lr_at(step, args)
            for group in opt.param_groups:
                group["lr"] = lr_now
            opt.step()
            completed_step = step
            if step % args.log_every == 0:
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(
                    f"step {step:5d}  {task:12s} k={k:2d} "
                    f"B={plan.microbatch}x{args.batch // plan.microbatch}  "
                    f"loss {float(loss.detach()):.4f}  "
                    f"answer {float(ce_ans.detach()):.4f}  "
                    f"emit {float(ce_emit.detach()):.4f}  "
                    f"({dt:.0f}s, peak {peak:.2f} GiB)"
                )
            if args.eval_every and step % args.eval_every == 0:
                for task_name in task_list:
                    accs = evaluate_sweep(
                        model,
                        surface,
                        tok,
                        task_name,
                        k_set,
                        args.eval_n,
                        args.condition,
                        args.source,
                        args.dest,
                        args.batch,
                        runner=eval_runner,
                        rows=eval_rows[task_name],
                    )
                    row = "  ".join(
                        f"k{k}:{accuracy:.3f}/{legal:.2f}"
                        for k, (accuracy, legal) in accs.items()
                    )
                    print(f"  eval {task_name:12s} acc/legal  {row}")
            if args.save_every and step % args.save_every == 0:
                _save_checkpoint(args.out, surface, opt, args, step)
            if args.snapshot_every and step % args.snapshot_every == 0:
                _save_checkpoint(f"{args.out}.{step}", surface, opt, args, step)
    except KeyboardInterrupt:
        _save_checkpoint(args.out, surface, opt, args, completed_step)
        print(f"interrupted; checkpointed completed step {completed_step}")
        raise
    finally:
        producer.close()
    _save_checkpoint(args.out, surface, opt, args, args.steps)
    print(f"saved {args.out}")


def main() -> None:
    # The training surface legitimately compiles a shape family per k
    # (span strides, rope slices, and prefix lengths all carry k), and
    # Dynamo's default 8-per-frame guard hard-aborts under fullgraph
    # mid-warmup (seen at k=32 on _dual_layer_math). Raised at CLI scope
    # only — imports never touch global config — and the guard against
    # *accidental* recompiles remains the in-step unique-graph audit.
    torch._dynamo.config.recompile_limit = 256
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["gate", "run"])
    p.add_argument("--model", default="google/gemma-3-1b-pt")
    p.add_argument("--source", type=int, default=11)
    p.add_argument("--dest", type=int, default=4)
    p.add_argument("--condition", choices=["dots+wire", "dots"], default="dots+wire")
    p.add_argument("--tasks", default="parity")
    p.add_argument("--k", default="1,2,4,8,16,32", help="eval sweep k set")
    p.add_argument(
        "--train-k",
        default="4,8,16,32",
        help="training k set; k<=2 is eval-only by default (D15): refreshed "
        "columns reach no supervised logit before t=3, so those steps train "
        "the row alone",
    )
    p.add_argument(
        "--k-gamma",
        type=float,
        default=1.0,
        help="training k weight exponent, P(k) ~ k**gamma; 0 = uniform",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=512,
        help="effective optimizer batch; auto mode chooses the CUDA microbatch",
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--cosine",
        type=int,
        default=2000,
        help="cosine period in steps after warmup, then flat at the floor; "
        "independent of --steps; 0 = flat at --lr (D15)",
    )
    p.add_argument("--lr-floor", type=float, default=1e-4)
    p.add_argument(
        "--lam",
        type=float,
        default=0.125,
        help="emission-span CE weight (D15): the hazard calibrates at any "
        "weight, and a small one damps the per-batch dot/answer pressure on "
        "the tied row",
    )
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="also keep an immutable checkpoint copy at OUT.STEP every N steps",
    )
    p.add_argument(
        "--knobs",
        default=None,
        help="task knob overrides for this run, e.g. 'length=8'",
    )
    p.add_argument("--out", default="data/train/surface.pt")
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="resume from PATH, or from --out when passed without PATH",
    )
    args = p.parse_args()
    if args.knobs:
        # CLI-scope override of the pinned task knobs (like the recompile
        # guard above): batches and eval rows in this process see one
        # consistent knob set, and the checkpoint's args record it.
        for pair in args.knobs.split(","):
            key, _, value = pair.partition("=")
            for task in args.tasks.split(","):
                if key not in tasks.KNOBS[task]:
                    raise ValueError(f"{task} has no knob {key!r}")
                tasks.KNOBS[task][key] = int(value)
    if args.mode == "gate":
        gate_mode(args)
    else:
        run_mode(args)


if __name__ == "__main__":
    main()
