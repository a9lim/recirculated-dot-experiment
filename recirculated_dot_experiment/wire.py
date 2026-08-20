"""Two-pass recirculation engine for Gemma3 (design.md D1).

Drives the HF decoder layers manually — no fork of layer internals, no
hooks. Semantics (verified by the G0 identity gate and an adversarial
Codex review with an fp32 sequential reference, 2026-08-20): per
wall-clock step t, the first pass of column t (its logits are the
readout) reads column t-1 at first-pass fidelity and everything older
refreshed; the second pass of column t-1 (layers dest+1..top, with the
layer-dest output replaced by the alpha-mix of Eq. 1) then overwrites
t-1's KV entries for those layers. Later columns see the refreshed state
through ordinary attention — that is the wire.

The implementation exploits two exact structural facts:

1. Refresh only ever touches layers dest+1..top, so layers 0..dest of
   every column form a plain causal transformer — computed for the whole
   sequence in ONE parallel prefill (pass A). Only the top slab is
   serial.
2. First pass of column t and second pass of column t-1 share a cache
   snapshot (the paper runs them concurrently), so they run as ONE
   batched [2B] call over the serial slab.

DualCache keeps two persistent lanes per layer in one [2B] buffer —
lane 0 the first-pass view (refreshed history + newest first-pass
column), lane 1 the refresh view (refreshed history + its own
recompute). All slot writes are tensor-index ops so the compiled slab
specializes on shapes only, never on the step index. Visibility comes
from a running additive mask, opened progressively from outside the
graph: at step t, lane 0 gains slot t and lane 1 gains slot t-1 (lane
1's slot t stays masked — its content after the batched write is moot).
Lane 0's refresh of slot t-1 is committed via `commit(i)` after that
layer's attention has consumed its view. In eager mode the cache
returns the visible prefix; in compiled mode it returns the full
buffer and the mask alone carries visibility (stale/garbage slots are
fully masked).

`compile_mode` compiles the 21-layer slab step with
torch.compile(fullgraph=True, dynamic=False) — one graph per (B, T,
dtype) combination, first call pays the compile. The t=0 single-pass
step always runs eager.

NLL / logits are computed chunked at the end from collected top hiddens
— per-step fp32 softmax over Gemma3's 262k vocab would dominate runtime
and memory.

Requirements the constructor enforces: sdpa attention (pass A relies on
sdpa synthesizing causality for maskless q_len > 1; eager attention
would attend bidirectionally there) and sequences within one sliding
window, where sliding and full attention coincide.
`final_logit_softcapping` is applied when the config carries it (None
on Gemma3-1B).

Inference-only in v0 (`inference_mode`): the in-place lane buffers break
autograd. The BPTT training path gets a functional token-chunked cache
(design.md D9) instead.

Layer indexing: `source`/`dest` are 0-based indices into model.layers,
and the tapped value is the hidden state *after* that layer. G0
empirically confirmed the paper's Gemma3-1B pair {11, 4} is 0-based
(-8.8% ppl on PG19 vs -1.4% for the {10, 3} reading).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

try:
    from flash_attn import flash_attn_with_kvcache as _fa2_kvcache
except ImportError:  # e.g. the Mac; fa2 backend then refuses at construction
    _fa2_kvcache = None

if _fa2_kvcache is not None:
    # flash_attn 2.8 exposes fwd_kvcache as a raw C extension (PyCapsule),
    # which dynamo cannot trace; wrapping it as a torch custom op with the
    # cache mutations declared makes it an opaque-but-graphable extern.
    @torch.library.custom_op("wire::fa2_kvcache", mutates_args=("k_cache", "v_cache"))
    def _fa2_op(
        q: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        k_new: Tensor,
        v_new: Tensor,
        cache_seqlens: Tensor,
        scale: float,
    ) -> Tensor:
        return _fa2_kvcache(
            q, k_cache, v_cache, k=k_new, v=v_new, cache_seqlens=cache_seqlens, softmax_scale=scale
        )

    @_fa2_op.register_fake
    def _(q, k_cache, v_cache, k_new, v_new, cache_seqlens, scale):
        return torch.empty_like(q)


def _wire_sdpa(module, query, key, value, attention_mask, **kwargs):
    """The wire's attention: three dispatch tiers.

    1. `wire_cache` kwarg present (fa2 backend, dual/single step): the
       FlashAttention kvcache kernel — per-lane cache_seqlens express the
       dual-pass visibility (both lanes are prefix reads), the slot write
       happens in-kernel, GQA is native, and no mask tensor exists at all
       (measured 43 vs 119 us against masked sdpa at kv=512, 200 rows).
       bf16/fp16 only.
    2. mask present (sdpa backend, dual/single step): sdpa with stride-0
       expand views for GQA — HF's stock path would repeat_kv (a copy +
       4x traffic; 464 us), and enable_gqa=True with a dense mask falls
       back to the math backend (3354 us). Expand views hit the fused
       kernel at 1-head traffic (119 us, bitwise-identical).
    3. maskless (pass A): stock sdpa (is_causal synthesis for q_len > 1).
    """
    cache = kwargs.get("wire_cache")
    if cache is not None:
        out = _fa2_op(
            query.transpose(1, 2),
            *cache.views(module.layer_idx),
            key.transpose(1, 2),
            value.transpose(1, 2),
            cache.seqlens,
            kwargs.get("scaling"),
        )
        return out, None
    if attention_mask is None:
        return sdpa_attention_forward(module, query, key, value, None, **kwargs)
    if module.num_key_value_groups > 1 and key.shape[1] == 1:
        key = key.expand(-1, query.shape[1], -1, -1)
        value = value.expand(-1, query.shape[1], -1, -1)
    elif module.num_key_value_groups > 1:  # kv_heads > 1: strides can't merge
        from transformers.integrations.sdpa_attention import repeat_kv

        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    out = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, scale=kwargs.get("scaling")
    )
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("wire_sdpa", _wire_sdpa)


@dataclass
class WireConfig:
    source: int = 11
    dest: int = 4
    alpha: float = 0.15
    ramp_steps: int = 10  # alpha_t = min(t/ramp, 1) * alpha; 0 disables
    eps: float = 1e-6

    def alpha_at(self, t: int) -> float:
        if self.ramp_steps <= 0:
            return self.alpha
        return min(t / self.ramp_steps, 1.0) * self.alpha


class DualCache:
    """Two-lane KV store for the serial slab with same-snapshot dual-pass
    views. `update` matches the transformers Cache calling convention:
    attention attends over exactly what it returns, restricted by the
    engine's running mask. Lane buffers are allocated lazily per layer,
    so only slab layers ever hold memory."""

    def __init__(
        self,
        batch: int,
        kv_heads: int,
        max_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.B = batch
        self.shape = (2 * batch, kv_heads, max_len, head_dim)
        self.dtype = dtype
        self.device = device
        self.k: dict[int, Tensor] = {}
        self.v: dict[int, Tensor] = {}
        self.pending: dict[int, tuple[Tensor, Tensor]] = {}
        self.t = 0
        self.dual = False  # False: plain single pass at slot t (t == 0)
        self.static = False  # True: full-length returns (compiled mode)
        self.t_idx: Tensor | None = None  # slot t as a [1] index tensor
        self.tm1_idx: Tensor | None = None  # slot t-1

    def _lane(self, store: dict[int, Tensor], layer_idx: int) -> Tensor:
        if layer_idx not in store:
            store[layer_idx] = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        return store[layer_idx]

    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int, *a, **kw):
        B = self.B
        k, v = self._lane(self.k, layer_idx), self._lane(self.v, layer_idx)
        if not self.dual:
            k[:, :, self.t : self.t + 1] = torch.cat([key_states] * 2)
            v[:, :, self.t : self.t + 1] = torch.cat([value_states] * 2)
            return k[:B, :, : self.t + 1], v[:B, :, : self.t + 1]
        # slot t: k1 -> lane 0; k2 lands in lane 1's slot t, which is
        # masked until step t+1 overwrites it (content moot)
        k.index_copy_(2, self.t_idx, key_states)
        v.index_copy_(2, self.t_idx, value_states)
        k2, v2 = key_states[B:], value_states[B:]
        k[B:].index_copy_(2, self.tm1_idx, k2)  # lane 1 attends its own recompute
        v[B:].index_copy_(2, self.tm1_idx, v2)
        self.pending[layer_idx] = (k2, v2)
        if self.static:
            return k, v
        return k[:, :, : self.t + 1], v[:, :, : self.t + 1]

    def commit(self, layer_idx: int) -> None:
        """Refresh lane 0's slot t-1 — call after this layer's attention
        has consumed its first-pass view (same-snapshot semantics)."""
        k2, v2 = self.pending.pop(layer_idx)
        self.k[layer_idx][: self.B].index_copy_(2, self.tm1_idx, k2)
        self.v[layer_idx][: self.B].index_copy_(2, self.tm1_idx, v2)


class DualCacheFA2:
    """Stacked FA2-layout ([rows, seq, kv_heads, head_dim]) KV lanes for
    the serial slab. The kvcache kernel does the slot writes in-kernel
    from cache_seqlens (lane 0 appends first-pass at slot t; lane 1
    overwrites its recompute at t-1), so `update` just forwards the
    new-token KV to the attention interface. Lane 0's refresh commits as
    ONE fused lane1->lane0 copy across all layers at step end — next
    reader of any lane is step t+1, so end-of-step commit preserves the
    same-snapshot semantics exactly."""

    dual = False  # engine-side per-layer commit is never used on this cache

    def __init__(
        self,
        n_slab: int,
        first_layer: int,
        batch: int,
        kv_heads: int,
        max_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.B = batch
        self.first = first_layer
        shape = (n_slab, 2 * batch, max_len, kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.seq_base = torch.cat(
            [
                torch.zeros(batch, dtype=torch.int32, device=device),
                torch.full((batch,), -1, dtype=torch.int32, device=device),
            ]
        )
        self.seqlens = self.seq_base[: batch].clone()
        self.rows = batch

    def step(self, t: int) -> None:
        if t == 0:
            self.rows = self.B
            self.seqlens = torch.zeros(self.B, dtype=torch.int32, device=self.k.device)
        else:
            self.rows = 2 * self.B
            self.seqlens = self.seq_base + t

    def views(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        j = layer_idx - self.first
        return self.k[j, : self.rows], self.v[j, : self.rows]

    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int, *a, **kw):
        return key_states, value_states  # the fa2 kernel does the cache write

    def commit(self, t: int) -> None:
        self.k[:, : self.B, t - 1] = self.k[:, self.B :, t - 1]
        self.v[:, : self.B, t - 1] = self.v[:, self.B :, t - 1]


class RecirculationEngine:
    """Wraps a Gemma3ForCausalLM (weights untouched, model in eval mode).

    compile_mode: None (eager), "default", or "reduce-overhead" — passed
    to torch.compile for the slab step.
    attn_backend: "sdpa" (masked, works in any dtype — the fp32 identity
    gate requires it), "fa2" (FlashAttention kvcache; bf16/fp16 only), or
    "auto" (fa2 when available and the model is half precision).
    """

    def __init__(
        self,
        model,
        cfg: WireConfig,
        compile_mode: str | None = None,
        attn_backend: str = "auto",
    ):
        mc = model.config
        if not 0 <= cfg.dest < cfg.source < mc.num_hidden_layers:
            raise ValueError(f"need 0 <= dest < source < {mc.num_hidden_layers}")
        # pass A relies on sdpa synthesizing causality for maskless q_len > 1;
        # eager attention would attend bidirectionally there (Codex repro:
        # 0.21 max logit error on a tiny model)
        if mc._attn_implementation not in ("sdpa", "wire_sdpa"):
            raise ValueError(f"engine requires sdpa attention, got {mc._attn_implementation!r}")
        mc._attn_implementation = "wire_sdpa"  # native-GQA sdpa (see _wire_sdpa)
        self.model = model
        self.cfg = cfg
        inner = model.model
        self.embed = inner.embed_tokens
        self.layers = inner.layers
        self.final_norm = inner.norm
        self.rotary = inner.rotary_emb
        self.lm_head = model.lm_head
        self.layer_types = mc.layer_types
        self.n_layers = mc.num_hidden_layers
        self.kv_heads = mc.num_key_value_heads
        self.head_dim = mc.head_dim
        self.window = mc.sliding_window
        self.softcap = getattr(mc, "final_logit_softcapping", None)  # None on Gemma3-1B
        dtype = inner.embed_tokens.weight.dtype
        if attn_backend == "auto":
            attn_backend = (
                "fa2"
                if _fa2_kvcache is not None and dtype in (torch.bfloat16, torch.float16)
                else "sdpa"
            )
        if attn_backend == "fa2":
            if _fa2_kvcache is None:
                raise ValueError("fa2 backend requires the flash-attn package")
            if dtype not in (torch.bfloat16, torch.float16):
                raise ValueError(f"fa2 backend is half-precision only, model is {dtype}")
        elif attn_backend != "sdpa":
            raise ValueError(f"unknown attn_backend {attn_backend!r}")
        self.attn_backend = attn_backend
        self._cache: DualCache | DualCacheFA2 | None = None
        self._mask: Tensor | None = None
        self._slab_c = None
        if compile_mode is not None:
            self._slab_c = torch.compile(self._slab, fullgraph=True, dynamic=False, mode=compile_mode)

    def mix(self, h_s: Tensor, h_d: Tensor, alpha, beta) -> Tensor:
        ratio = h_d.norm(dim=-1, keepdim=True) / h_s.norm(dim=-1, keepdim=True).clamp_min(
            self.cfg.eps
        )
        return beta * h_d + alpha * ratio * h_s

    def _slab(self, x: Tensor, mask: Tensor | None, cos_f, sin_f, cos_s, sin_s):
        """One step over layers dest+1..top; the single shared body for
        eager and compiled paths."""
        cache = self._cache
        B = cache.B
        wire = cache if isinstance(cache, DualCacheFA2) else None
        h_s = x[:B]
        for i in range(self.cfg.dest + 1, self.n_layers):
            pe = (cos_f, sin_f) if self.layer_types[i] == "full_attention" else (cos_s, sin_s)
            x = self.layers[i](
                x,
                position_embeddings=pe,
                attention_mask=mask,
                past_key_values=cache,
                wire_cache=wire,
            )
            if i == self.cfg.source:
                h_s = x[:B]
            if cache.dual:  # sdpa path: per-layer commit; fa2 commits at step end
                cache.commit(i)
        return x, h_s

    def _ensure_buffers(self, B: int, T: int, dtype, device):
        c = self._cache
        if self.attn_backend == "fa2":
            if (
                not isinstance(c, DualCacheFA2)
                or c.B != B
                or c.k.shape[2] < T
                or c.k.dtype != dtype
            ):
                self._cache = DualCacheFA2(
                    self.n_layers - self.cfg.dest - 1,
                    self.cfg.dest + 1,
                    B,
                    self.kv_heads,
                    T,
                    self.head_dim,
                    dtype,
                    device,
                )
            return self._cache, None
        if not isinstance(c, DualCache) or c.B != B or c.shape[2] < T or c.dtype != dtype:
            self._cache = DualCache(B, self.kv_heads, T, self.head_dim, dtype, device)
            self._mask = torch.empty(2 * B, 1, 1, T, dtype=dtype, device=device)
        elif self._mask.shape[-1] < T:
            self._mask = torch.empty(2 * B, 1, 1, T, dtype=dtype, device=device)
        return self._cache, self._mask[..., :T] if self._mask.shape[-1] != T else self._mask

    @torch.inference_mode()
    def teacher_forced(
        self, input_ids: Tensor, alpha_fn=None, return_logits: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        """Full-scope recirculation; returns (nll [B, T-1], logits | None).

        alpha_fn(t, h_s, h_d) -> (alpha, beta), scalars or [B, 1, D]
        tensors; None uses the config constants with the ramp (convex,
        beta = 1 - alpha_t). This is the hook the adaptive gate MLP
        replaces later.
        """
        B, T = input_ids.shape
        if not 1 <= T <= self.window:
            raise ValueError(f"wire v0 requires 1 <= T <= sliding_window ({self.window})")
        device = input_ids.device
        dtype = self.embed.weight.dtype
        cfg = self.cfg
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        h = self.embed(input_ids)
        rope = {lt: self.rotary(h, pos_ids, lt) for lt in set(self.layer_types)}

        # Pass A: layers 0..dest for all columns in parallel (exact: the
        # refresh never touches these layers, so nothing here ever sees
        # recirculated state). No cache — slab KVs are never read later.
        for i in range(cfg.dest + 1):
            h = self.layers[i](
                h,
                position_embeddings=rope[self.layer_types[i]],
                attention_mask=None,
                past_key_values=None,
            )
        h_dest = h  # [B, T, D]: first-pass layer-dest output of every column

        # Serial slab: layers dest+1..top, batched dual pass. Slicing
        # rope_dual at [t-1 : t] yields (position t for the first half,
        # position t-1 for the second).
        cache, mask_buf = self._ensure_buffers(B, T, dtype, device)
        fa2 = self.attn_backend == "fa2"
        compiled = self._slab_c is not None and T > 1
        if not fa2:
            cache.static = compiled
            neg = torch.finfo(dtype).min
            mask_buf.fill_(neg)
            idxs = torch.arange(T, device=device)
        tops = torch.empty(B, T, h.shape[-1], dtype=dtype, device=device)
        rope_dual = (
            {
                lt: (
                    torch.cat([c[:, 1:].expand(B, -1, -1), c[:, : T - 1].expand(B, -1, -1)]),
                    torch.cat([s[:, 1:].expand(B, -1, -1), s[:, : T - 1].expand(B, -1, -1)]),
                )
                for lt, (c, s) in rope.items()
            }
            if T > 1
            else {}
        )
        h_s_prev: Tensor | None = None
        for t in range(T):
            if fa2:
                cache.step(t)
            if t == 0:
                if not fa2:
                    cache.t, cache.dual = 0, False
                cf, sf = rope["full_attention"]
                cs, ss = rope["sliding_attention"]
                x, h_s = self._slab(
                    h_dest[:, :1], None, cf[:, :1], sf[:, :1], cs[:, :1], ss[:, :1]
                )
                if not fa2:
                    mask_buf[:B, ..., 0] = 0.0
            else:
                if alpha_fn is None:
                    a = cfg.alpha_at(t - 1)
                    ab = (a, 1.0 - a)
                else:
                    ab = alpha_fn(t - 1, h_s_prev, h_dest[:, t - 1 : t])
                x = torch.cat([h_dest[:, t : t + 1], self.mix(h_s_prev, h_dest[:, t - 1 : t], *ab)])
                if fa2:
                    mask = None
                else:
                    cache.t, cache.dual = t, True
                    cache.t_idx, cache.tm1_idx = idxs[t : t + 1], idxs[t - 1 : t]
                    mask_buf[:B, ..., t] = 0.0  # lane 0 gains its own slot t
                    mask_buf[B:, ..., t - 1] = 0.0  # lane 1 gains its recomputed t-1
                    mask = mask_buf if compiled else mask_buf[..., : t + 1]
                cf, sf = rope_dual["full_attention"]
                cs, ss = rope_dual["sliding_attention"]
                pe4 = (cf[:, t - 1 : t], sf[:, t - 1 : t], cs[:, t - 1 : t], ss[:, t - 1 : t])
                if compiled:
                    x, h_s = self._slab_c(x, mask, *pe4)
                else:
                    x, h_s = self._slab(x, mask, *pe4)
                if fa2:
                    cache.commit(t)
            tops[:, t : t + 1] = x[:B]
            h_s_prev = h_s

        # Deferred readout: chunked over positions to bound the fp32
        # softmax footprint at Gemma3's 262k vocab (~1.5 GB per chunk).
        nlls, logits, chunk = [], [], max(1, 1024 // B)
        for i in range(0, T, chunk):
            lg = self.lm_head(self.final_norm(tops[:, i : i + chunk]))
            if self.softcap is not None:
                lg = torch.tanh(lg / self.softcap) * self.softcap
            if return_logits:
                logits.append(lg)
            hi = min(i + chunk, T - 1)
            if i < hi:
                logprobs = torch.log_softmax(lg[:, : hi - i].float(), dim=-1)
                targets = input_ids[:, i + 1 : hi + 1]
                nlls.append(-logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1))
        nll = torch.cat(nlls, dim=1) if nlls else torch.empty(B, 0, device=device)
        return nll, (torch.cat(logits, dim=1) if return_logits else None)

    def teacher_forced_logits(self, input_ids: Tensor, alpha_fn=None) -> Tensor:
        """First-pass logits [B, T, V]; small T only (full-vocab memory)."""
        return self.teacher_forced(input_ids, alpha_fn, return_logits=True)[1]
