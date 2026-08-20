"""Two-pass recirculation engine for Gemma3 (design.md D1, D10).

Canonical single path (ratified 2026-08-20): CUDA, half precision,
FlashAttention kvcache for the dual pass, torch.compile over the serial
slab. The sdpa dual-pass implementation, its mask machinery, and the
backend/compile flags were stripped at canonicalization — the fastest
path is the only path, so training and eval share one set of numerics.
History and measurements live in docs/findings.md; the stripped
variants remain in git history.

Semantics (verified across three review rounds, including an
adversarial Codex review with an fp32 sequential reference): per
wall-clock step t, the first pass of column t (its logits are the
readout) reads column t-1 at first-pass fidelity and everything older
refreshed; the second pass of column t-1 (layers dest+1..top, with the
layer-dest output replaced by the alpha-mix of Eq. 1) then overwrites
t-1's KV entries for those layers. Later columns see the refreshed
state through ordinary attention — that is the wire.

Structure the implementation exploits, all exact:

1. Refresh only ever touches layers dest+1..top, so layers 0..dest of
   every column form a plain causal transformer — computed for the
   whole sequence in ONE parallel prefill (pass A, stock sdpa). Only
   the top slab is serial.
2. First pass of column t and second pass of column t-1 share a cache
   snapshot (the paper runs them concurrently), so they run as ONE
   batched [2B] call over the serial slab.
3. Dual-pass visibility is per-lane prefix lengths, which the
   FlashAttention kvcache kernel expresses natively via per-row
   cache_seqlens — lane 0 appends the first-pass KV at slot t, lane 1
   overwrites its own recompute at t-1, both in-kernel, GQA native, no
   mask tensor anywhere. Lane 0's refresh commits as one fused
   lane1->lane0 copy at step end (the next reader is step t+1, so
   end-of-step commit preserves the same-snapshot semantics exactly).
4. The slab step is mutation-free from inductor's perspective (cache
   writes hide inside the custom op; the commit lives outside the
   graph), so it compiles fullgraph. flash-attn's raw PyCapsule entry
   is wrapped as a torch custom op with the cache mutations declared.

NLL / logits are computed chunked at the end from collected top
hiddens — per-step fp32 softmax over Gemma3's 262k vocab would
dominate runtime and memory.

Attention is FA2 everywhere (D10 addendum, ratified 2026-08-20): wire
steps use the kvcache custom op; pass A and plain HF forwards on the
flipped model defer to HF's flash_attention_2 interface with the
paired mask-interface registration — stock sdpa is fully retired, and
the whole flipped model object is half-precision-only as a
consequence. Constraints enforced: sequences within one sliding
window (where sliding and full attention coincide), CUDA + bf16/fp16
at call time (the fp32 identity gate retired at canonicalization —
G0 gates in bf16 with thresholds calibrated against the measured
self-noise null). `final_logit_softcapping` is applied when the
config carries it (None on Gemma3-1B).

Inference-only (`inference_mode`): the in-place lane buffers break
autograd. The BPTT training path gets a functional token-chunked cache
(design.md D9) and reuses the same custom op.

Layer indexing: `source`/`dest` are 0-based indices into model.layers,
and the tapped value is the hidden state *after* that layer. G0
empirically confirmed the paper's Gemma3-1B pair {11, 4} is 0-based.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb
from transformers import AttentionInterface, AttentionMaskInterface

# Hard requirement (D10): no degraded mode — the module refuses to import
# without flash-attn, by design.
from flash_attn import flash_attn_func as _fa2_prefill
from flash_attn import flash_attn_with_kvcache as _fa2_kvcache

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


@torch.library.custom_op("wire::fa2_prefill", mutates_args=())
def _fa2_prefill_op(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Traceable, maskless causal FA2 prefill for the <=window wire."""
    return _fa2_prefill(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)


@_fa2_prefill_op.register_fake
def _(q, k, v, scale):
    return torch.empty_like(q)

from transformers.integrations.flash_attention import flash_attention_forward
from transformers.masking_utils import flash_attention_mask

def _wire_attention(module, query, key, value, attention_mask, **kwargs):
    """FA2 everywhere (D10 addendum). Dual-pass steps (wire_cache set)
    go through the kvcache custom op; everything else — pass A's
    maskless causal prefill, plain HF forwards on the flipped model —
    defers to HF's flash_attention_2 interface, which handles causal,
    sliding-window, and padding/unpadding. The paired mask-interface
    registration makes create_causal_mask build FA2-format inputs
    (None when fully causal, 2D for padding) instead of dense 4D."""
    cache = kwargs.get("wire_cache")
    if cache is None:
        # HF's delegate resolves the flash package by reading
        # module.config._attn_implementation; scope the name to the
        # stock one for exactly this call (outside it, the config must
        # keep saying wire_attention so mask building stays FA2-format
        # and dispatch keeps hitting this function).
        mc = module.config
        mc._attn_implementation = "flash_attention_2"
        try:
            return flash_attention_forward(module, query, key, value, attention_mask, **kwargs)
        finally:
            mc._attn_implementation = "wire_attention"
    out = _fa2_op(
        query.transpose(1, 2),
        *cache.views(module.layer_idx),
        key.transpose(1, 2),
        value.transpose(1, 2),
        cache.seqlens,
        kwargs.get("scaling"),
    )
    return out, None

AttentionInterface.register("wire_attention", _wire_attention)
AttentionMaskInterface.register("wire_attention", flash_attention_mask)


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
    """Stacked FA2-layout ([rows, seq, kv_heads, head_dim]) KV lanes for
    the serial slab. The kvcache kernel does the slot writes in-kernel
    from cache_seqlens (lane 0 appends first-pass at slot t; lane 1
    overwrites its recompute at t-1), so `update` — the transformers
    Cache calling convention — just forwards the new-token KV through to
    the attention interface."""

    def __init__(
        self,
        n_slab: int,
        first_layer: int,
        batch: int,
        kv_heads: int,
        max_len: int,
        head_dim: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.B = batch
        self.first = first_layer
        shape = (n_slab, 2 * batch, max_len, kv_heads, head_dim)
        # Every visible slot is written before it is read. Empty allocation
        # avoids zeroing several GiB when a new shape bucket is created.
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty(shape, dtype=dtype, device=device)
        self.seq_base = torch.cat(
            [
                torch.zeros(batch, dtype=torch.int32, device=device),
                torch.full((batch,), -1, dtype=torch.int32, device=device),
            ]
        )
        self.seq_table = self.seq_base.unsqueeze(0) + torch.arange(
            max_len, dtype=torch.int32, device=device
        ).unsqueeze(1)
        self.seqlens = self.seq_table[0, :batch]
        self.rows = batch
        self.tops = torch.empty(batch, max_len, hidden_size, dtype=dtype, device=device)
        self.rope_dual: dict[str, tuple[Tensor, Tensor]] = {}

    def bind_rope(self, rope: dict[str, tuple[Tensor, Tensor]]) -> None:
        """Materialize stable dual-lane RoPE buffers once per shape bucket."""
        max_len = self.k.shape[2]
        self.rope_dual = {
            lt: (
                torch.cat(
                    [
                        c[:, 1:max_len].expand(self.B, -1, -1),
                        c[:, : max_len - 1].expand(self.B, -1, -1),
                    ]
                ),
                torch.cat(
                    [
                        s[:, 1:max_len].expand(self.B, -1, -1),
                        s[:, : max_len - 1].expand(self.B, -1, -1),
                    ]
                ),
            )
            for lt, (c, s) in rope.items()
        }

    def step(self, t: int) -> None:
        if t == 0:
            self.rows = self.B
            self.seqlens = self.seq_table[0, : self.B]
        else:
            self.rows = 2 * self.B
            self.seqlens = self.seq_table[t]

    def views(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        j = layer_idx - self.first
        return self.k[j, : self.rows], self.v[j, : self.rows]

    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int, *a, **kw):
        return key_states, value_states  # the kvcache kernel does the write

    def commit(self, t: int) -> None:
        """Refresh lane 0's slot t-1 across all layers in one fused copy —
        called at step end, after every layer's first-pass attention has
        consumed its view (same-snapshot semantics)."""
        self.k[:, : self.B, t - 1] = self.k[:, self.B :, t - 1]
        self.v[:, : self.B, t - 1] = self.v[:, self.B :, t - 1]


class RecirculationEngine:
    """Wraps a Gemma3ForCausalLM (weights untouched, model in eval mode)."""

    def __init__(self, model, cfg: WireConfig):
        mc = model.config
        if not 0 <= cfg.dest < cfg.source < mc.num_hidden_layers:
            raise ValueError(f"need 0 <= dest < source < {mc.num_hidden_layers}")
        # FA2 everywhere: the flip is safe regardless of the load-time
        # implementation because every forward after it — pass A, wire
        # steps, plain model(...) — dispatches through _wire_attention,
        # whose causality is explicit (module.is_causal / cache_seqlens),
        # and the paired mask interface builds FA2-format mask inputs.
        mc._attn_implementation = "wire_attention"
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
        self.q_heads = mc.num_attention_heads
        self.head_dim = mc.head_dim
        self.hidden_size = mc.hidden_size
        self.window = mc.sliding_window
        self.softcap = getattr(mc, "final_logit_softcapping", None)  # None on Gemma3-1B
        self._cache: DualCache | None = None
        self._rope: dict[str, tuple[Tensor, Tensor]] | None = None
        self._build_packed_weights()

        # Compile every tensor-heavy region. The outer position loop remains
        # the small state machine that advances views into the mutable cache.
        compile_kw = {"fullgraph": True, "dynamic": False}
        self._prefill_c = torch.compile(self._prefill, **compile_kw)
        self._slab_first_c = torch.compile(self._slab, **compile_kw)
        self._step_c = torch.compile(self._step, **compile_kw)
        self._nll_chunk_c = torch.compile(self._nll_chunk, **compile_kw)
        self._logits_chunk_c = torch.compile(self._logits_chunk, **compile_kw)
        self._nll_from_logits_c = torch.compile(self._nll_from_logits, **compile_kw)

    def _build_packed_weights(self) -> None:
        """Pack frozen projection families once; original model weights stay untouched."""
        qkv_weights, qkv_biases, gate_up_weights = [], [], []
        for layer in self.layers:
            attn = layer.self_attn
            qkv_weights.append(
                torch.cat(
                    [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0
                ).contiguous()
            )
            biases = (attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias)
            qkv_biases.append(
                torch.cat(biases, dim=0).contiguous() if biases[0] is not None else None
            )
            gate_up_weights.append(
                torch.cat([layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], dim=0).contiguous()
            )
        self._qkv_weights = tuple(qkv_weights)
        self._qkv_biases = tuple(qkv_biases)
        self._gate_up_weights = tuple(gate_up_weights)

    def _ensure_rope(self, dtype: torch.dtype, device: torch.device) -> dict[str, tuple[Tensor, Tensor]]:
        rope = self._rope
        if rope is None or next(iter(rope.values()))[0].dtype != dtype or next(iter(rope.values()))[
            0
        ].device != device:
            marker = torch.empty(1, 1, self.hidden_size, dtype=dtype, device=device)
            pos_ids = torch.arange(self.window, device=device).unsqueeze(0)
            self._rope = {
                lt: self.rotary(marker, pos_ids, lt) for lt in set(self.layer_types)
            }
        return self._rope

    def mix(self, h_s: Tensor, h_d: Tensor, alpha, beta) -> Tensor:
        ratio = h_d.norm(dim=-1, keepdim=True) / h_s.norm(dim=-1, keepdim=True).clamp_min(
            self.cfg.eps
        )
        return beta * h_d + alpha * ratio * h_s

    def _layer(self, i: int, x: Tensor, pe, cache: DualCache | None) -> Tensor:
        """Gemma3 decoder layer with packed QKV and gate/up projections."""
        layer = self.layers[i]
        attn = layer.self_attn
        residual = x
        h = layer.input_layernorm(x)

        q_size = self.q_heads * self.head_dim
        kv_size = self.kv_heads * self.head_dim
        qkv = F.linear(h, self._qkv_weights[i], self._qkv_biases[i])
        q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
        input_shape = h.shape[:-1]
        q = q.view(*input_shape, self.q_heads, self.head_dim).transpose(1, 2)
        k = k.view(*input_shape, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(*input_shape, self.kv_heads, self.head_dim).transpose(1, 2)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, *pe)

        if cache is None:
            a = _fa2_prefill_op(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn.scaling
            )
        else:
            a = _fa2_op(
                q.transpose(1, 2),
                *cache.views(i),
                k.transpose(1, 2),
                v.transpose(1, 2),
                cache.seqlens,
                attn.scaling,
            )
        a = a.reshape(*input_shape, -1).contiguous()
        h = residual + layer.post_attention_layernorm(attn.o_proj(a))

        residual = h
        h = layer.pre_feedforward_layernorm(h)
        gate, up = F.linear(h, self._gate_up_weights[i]).chunk(2, dim=-1)
        h = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
        return residual + layer.post_feedforward_layernorm(h)

    def _prefill(self, input_ids: Tensor, cos_f, sin_f, cos_s, sin_s) -> Tensor:
        h = self.embed(input_ids)
        for i in range(self.cfg.dest + 1):
            pe = (cos_f, sin_f) if self.layer_types[i] == "full_attention" else (cos_s, sin_s)
            h = self._layer(i, h, pe, None)
        return h

    def _slab(self, x: Tensor, cos_f, sin_f, cos_s, sin_s):
        """One step over layers dest+1..top (the t=0 step runs this body
        eagerly; every later step runs the compiled version)."""
        cache = self._cache
        h_s = x[: cache.B]
        for i in range(self.cfg.dest + 1, self.n_layers):
            pe = (cos_f, sin_f) if self.layer_types[i] == "full_attention" else (cos_s, sin_s)
            x = self._layer(i, x, pe, cache)
            if i == self.cfg.source:
                h_s = x[: cache.B]
        return x, h_s

    def _step(self, h_t: Tensor, h_prev: Tensor, h_s_prev: Tensor, alpha, beta, *rope):
        x = torch.cat([h_t, self.mix(h_s_prev, h_prev, alpha, beta)])
        return self._slab(x, *rope)

    def _logits_chunk(self, h: Tensor) -> Tensor:
        lg = self.lm_head(self.final_norm(h))
        if self.softcap is not None:
            lg = torch.tanh(lg / self.softcap) * self.softcap
        return lg

    @staticmethod
    def _nll_from_logits(lg: Tensor, targets: Tensor) -> Tensor:
        logprobs = torch.log_softmax(lg.float(), dim=-1)
        return -logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    def _nll_chunk(self, h: Tensor, targets: Tensor) -> Tensor:
        return self._nll_from_logits(self._logits_chunk(h), targets)

    def _ensure_cache(self, B: int, T: int, dtype, device) -> DualCache:
        c = self._cache
        if (
            c is None
            or c.B != B
            or c.k.shape[2] < T
            or c.k.dtype != dtype
            or c.k.device != torch.device(device)
        ):
            self._cache = DualCache(
                self.n_layers - self.cfg.dest - 1,
                self.cfg.dest + 1,
                B,
                self.kv_heads,
                T,
                self.head_dim,
                self.hidden_size,
                dtype,
                device,
            )
            self._cache.bind_rope(self._ensure_rope(dtype, torch.device(device)))
        return self._cache

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
            raise ValueError(f"wire requires 1 <= T <= sliding_window ({self.window})")
        if not input_ids.is_cuda:
            raise ValueError("the canonical wire is CUDA-only (D10)")
        device = input_ids.device
        dtype = self.embed.weight.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"the canonical wire is half-precision only, model is {dtype}")
        cfg = self.cfg
        rope = self._ensure_rope(dtype, device)
        cf, sf = rope["full_attention"]
        cs, ss = rope["sliding_attention"]

        # Pass A: layers 0..dest for all columns in parallel (exact: the
        # refresh never touches these layers, so nothing here ever sees
        # recirculated state). No cache — slab KVs are never read later.
        h_dest = self._prefill_c(
            input_ids, cf[:, :T], sf[:, :T], cs[:, :T], ss[:, :T]
        )

        # Serial slab: layers dest+1..top, batched dual pass. Slicing
        # rope_dual at [t-1 : t] yields (position t for the first half,
        # position t-1 for the second).
        cache = self._ensure_cache(B, T, dtype, device)
        tops = cache.tops[:, :T]
        rope_dual = cache.rope_dual
        h_s_prev: Tensor | None = None
        for t in range(T):
            cache.step(t)
            if t == 0:
                x, h_s = self._slab_first_c(
                    h_dest[:, :1], cf[:, :1], sf[:, :1], cs[:, :1], ss[:, :1]
                )
            else:
                if alpha_fn is None:
                    a = cfg.alpha_at(t - 1)
                    ab = (a, 1.0 - a)
                else:
                    ab = alpha_fn(t - 1, h_s_prev, h_dest[:, t - 1 : t])
                cf, sf = rope_dual["full_attention"]
                cs, ss = rope_dual["sliding_attention"]
                x, h_s = self._step_c(
                    h_dest[:, t : t + 1],
                    h_dest[:, t - 1 : t],
                    h_s_prev,
                    *ab,
                    cf[:, t - 1 : t],
                    sf[:, t - 1 : t],
                    cs[:, t - 1 : t],
                    ss[:, t - 1 : t],
                )
                cache.commit(t)
            tops[:, t : t + 1] = x[:B]
            h_s_prev = h_s

        # Deferred readout: chunked over positions to bound the fp32
        # softmax footprint at Gemma3's 262k vocab. The NLL-only compiled
        # path fuses log-softmax + target gather and never materializes the
        # fp32 [tokens, vocab] tensor.
        nlls, logits, chunk = [], [], max(1, 1024 // B)
        if return_logits:
            for i in range(0, T, chunk):
                lg = self._logits_chunk_c(tops[:, i : i + chunk])
                logits.append(lg)
                hi = min(i + chunk, T - 1)
                if i < hi:
                    targets = input_ids[:, i + 1 : hi + 1]
                    nlls.append(self._nll_from_logits_c(lg[:, : hi - i], targets))
        else:
            for i in range(0, T - 1, chunk):
                hi = min(i + chunk, T - 1)
                targets = input_ids[:, i + 1 : hi + 1]
                nlls.append(self._nll_chunk_c(tops[:, i:hi], targets))
        nll = torch.cat(nlls, dim=1) if nlls else torch.empty(B, 0, device=device)
        return nll, (torch.cat(logits, dim=1) if return_logits else None)

    def teacher_forced_logits(self, input_ids: Tensor, alpha_fn=None) -> Tensor:
        """First-pass logits [B, T, V]; small T only (full-vocab memory)."""
        return self.teacher_forced(input_ids, alpha_fn, return_logits=True)[1]
