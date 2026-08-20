"""Two-pass recirculation engine for Gemma3 (design.md D1, D10).

Canonical single path (ratified 2026-08-20): CUDA, half precision,
FlashAttention kvcache for the dual pass, packed frozen projections,
and fullgraph compilation of every tensor-heavy region. The sdpa
dual-pass implementation, its mask machinery, and the
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
   whole sequence in ONE compiled parallel FA2 prefill. Only the top
   slab is serial.
2. First pass of column t and second pass of column t-1 share a cache
   snapshot (the paper runs them concurrently), so they run as ONE
   interleaved [2B] call over the serial slab.
3. The two branches share the refreshed prefix 0..t-2 exactly. It is
   stored once, while one side slot retains lane 0's prior first-pass
   KV. FA2 reads the common prefix through adjacent cache_batch_idx
   rows; a compiled two-key tail and fp32 LSE merge add each branch's
   distinct keys. The refresh write happens only after both queries,
   preserving same-snapshot semantics while halving KV storage.
4. The tensor work is split into fullgraph regions for prefill, the
   first slab step, the recurrent mix+slab step, and readout. A manual
   steady-step CUDA graph puts device-side position selection, recurrent
   state, cache writes, top-state storage, and counters behind one host
   replay per position. flash-attn's raw PyCapsule entries are wrapped
   as torch custom ops.
5. Gemma's frozen Q/K/V and gate/up weights are packed once, reducing
   each decoder layer from seven GEMMs to four. The original Parameters
   become disjoint views of those packed tensors, so plain forwards and
   serialization retain their values without keeping duplicate storage.
6. Pass A uses one reusable B=64 compiled signature and writes each
   chunk directly into the destination-state cache. Recurrent RoPE keeps
   only the compact position tables; the compiled step gathers and
   broadcasts its two live positions instead of retaining a [2B,T,D]
   expansion.

NLL / logits are computed in compiled chunks at the end from collected
top hiddens. The NLL-only graph fuses fp32 log-softmax with target
gather, avoiding the full fp32 [tokens, vocab] materialization.

Attention is FA2 everywhere (D10 addendum, ratified 2026-08-20): wire
steps use the kvcache custom op; pass A uses a compiled causal-prefill
custom op; plain HF forwards on the flipped model defer to HF's
flash_attention_2 interface with the paired mask-interface
registration — stock sdpa is fully retired, and
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

# Hard requirement (D10): no degraded mode — the module refuses to import
# without flash-attn, by design.
from flash_attn import flash_attn_func as _fa2_prefill
from flash_attn import flash_attn_with_kvcache as _fa2_kvcache
from torch import Tensor
from torch.nn import functional as F
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

_PREFILL_BATCH = 64

# Per-shape specialization is the canon (D10): every distinct T compiles
# its own graph, and a task grid (D11) legitimately visits dozens of
# prompt+k lengths. Dynamo's default guardrail (8 recompiles per frame,
# a hard abort under fullgraph) would kill such a grid mid-run; the
# guard we actually rely on against *accidental* recompiles is G0's
# unique-graph audit over timed evaluation.
torch._dynamo.config.recompile_limit = 256


def _clone_packed_state_dict_views(module, state_dict, prefix, local_metadata) -> None:
    """Give serializers independent tensors without retaining them at runtime."""
    del local_metadata
    for name in module._wire_packed_state_dict_keys:
        key = prefix + name
        if key in state_dict:
            state_dict[key] = state_dict[key].clone()


@torch.library.custom_op("wire::fa2_prefill", mutates_args=())
def _fa2_prefill_op(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Traceable, maskless causal FA2 prefill for the <=window wire."""
    return _fa2_prefill(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)


@_fa2_prefill_op.register_fake
def _(q, k, v, scale):
    return torch.empty_like(q)


def _merge_branch_tails(
    q: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    side_k: Tensor,
    side_v: Tensor,
    prefix: Tensor,
    prefix_lse: Tensor,
    cache_seqlens: Tensor,
    scale: float,
) -> Tensor:
    """Small exact two-key softmax kept separate from the 21-layer graph."""
    q0, q1 = q[0::2], q[1::2]
    k0, k1 = k_new[0::2], k_new[1::2]
    v0, v1 = v_new[0::2], v_new[1::2]
    score_prev = (q0 * side_k).sum(dim=-1) * scale
    score_new = (q0 * k0).sum(dim=-1) * scale
    score_prev = score_prev.float()
    score_new = score_new.float()
    tail_maximum0 = torch.maximum(score_prev, score_new)
    prev_weight = torch.exp(score_prev - tail_maximum0)
    new_weight = torch.exp(score_new - tail_maximum0)
    tail_denom0 = prev_weight + new_weight
    tail_lse0 = tail_maximum0 + torch.log(tail_denom0)
    tail_out0 = (
        prev_weight.unsqueeze(-1) * side_v.float()
        + new_weight.unsqueeze(-1) * v0.float()
    ) / tail_denom0.unsqueeze(-1)
    tail_lse1 = ((q1 * k1).sum(dim=-1) * scale).float()
    tail_out1 = v1.expand(-1, -1, q.shape[2], -1).float()
    tail = torch.stack([tail_out0, tail_out1], dim=1).reshape_as(prefix)
    tail_lse = torch.stack([tail_lse0, tail_lse1], dim=1).reshape(
        q.shape[0], q.shape[1], q.shape[2]
    )
    prefix_lse = torch.where(
        cache_seqlens[:1] > 0,
        prefix_lse.transpose(1, 2),
        torch.full_like(prefix_lse.transpose(1, 2), float("-inf")),
    )
    maximum = torch.maximum(prefix_lse, tail_lse)
    prefix_weight = torch.exp(prefix_lse - maximum)
    tail_weight = torch.exp(tail_lse - maximum)
    return (
        (
            prefix_weight.unsqueeze(-1) * prefix.float()
            + tail_weight.unsqueeze(-1) * tail
        )
        / (prefix_weight + tail_weight).unsqueeze(-1)
    ).to(q.dtype)


_merge_branch_tails_c = torch.compile(
    _merge_branch_tails, fullgraph=True, dynamic=False
)


@torch.library.custom_op(
    "wire::dual_branch_attention",
    mutates_args=("k_cache", "v_cache", "side_k", "side_v"),
)
def _dual_branch_attention_op(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    side_k: Tensor,
    side_v: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    cache_seqlens: Tensor,
    cache_batch_idx: Tensor,
    write_index: Tensor,
    scale: float,
) -> Tensor:
    prefix, prefix_lse = _fa2_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        softmax_scale=scale,
        return_softmax_lse=True,
    )
    out = _merge_branch_tails_c(
        q,
        k_new,
        v_new,
        side_k,
        side_v,
        prefix,
        prefix_lse,
        cache_seqlens,
        scale,
    )
    k0, k1 = k_new[0::2], k_new[1::2]
    v0, v1 = v_new[0::2], v_new[1::2]
    k_cache.index_copy_(1, write_index, k1)
    v_cache.index_copy_(1, write_index, v1)
    side_k.copy_(k0)
    side_v.copy_(v0)
    return out


@_dual_branch_attention_op.register_fake
def _(
    q,
    k_cache,
    v_cache,
    side_k,
    side_v,
    k_new,
    v_new,
    cache_seqlens,
    cache_batch_idx,
    write_index,
    scale,
):
    return torch.empty_like(q)


from transformers.integrations.flash_attention import flash_attention_forward
from transformers.masking_utils import flash_attention_mask


def _wire_attention(module, query, key, value, attention_mask, **kwargs):
    """Delegate ordinary model forwards to HF's FA2 interface."""
    # HF resolves the flash package by reading this name; scope it to the
    # stock implementation for exactly this call while keeping the custom
    # mask interface and dispatcher name outside it.
    mc = module.config
    mc._attn_implementation = "flash_attention_2"
    try:
        return flash_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )
    finally:
        mc._attn_implementation = "wire_attention"


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


class BranchCache:
    """Shared refreshed prefix plus one first-pass branch KV per layer.

    At steady step ``t`` both branch queries see the same refreshed slots
    ``0..t-2``.  Lane 0 adds the prior first-pass KV and its own new KV;
    lane 1 adds only its own recomputed KV.  Storing that decomposition
    once halves the serial slab cache and lets FA2 reuse the common prefix.
    """

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
        shape = (n_slab, batch, max_len, kv_heads, head_dim)
        # Every visible slot is written before it is read. Empty allocation
        # avoids zeroing several GiB when a new shape bucket is created.
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty(shape, dtype=dtype, device=device)
        side_shape = (n_slab, batch, 1, kv_heads, head_dim)
        self.side_k = torch.empty(side_shape, dtype=dtype, device=device)
        self.side_v = torch.empty(side_shape, dtype=dtype, device=device)
        self.seq_table = (
            torch.arange(max_len, dtype=torch.int32, device=device)
            .unsqueeze(1)
            .expand(-1, 2 * batch)
            .contiguous()
        )
        self.index_table = torch.arange(max_len, dtype=torch.int64, device=device)
        self.cache_batch_idx = torch.arange(
            batch, dtype=torch.int32, device=device
        ).repeat_interleave(2)
        prefill_rows = ((batch + _PREFILL_BATCH - 1) // _PREFILL_BATCH) * _PREFILL_BATCH
        self.prefill_rows = (
            torch.arange(prefill_rows, dtype=torch.int64, device=device)
            .clamp_max(batch - 1)
            .view(-1, _PREFILL_BATCH)
        )
        self.seqlens = self.seq_table[0]
        self.write_index = self.index_table[:1]
        self.rows = batch
        # At most one readout chunk of padding makes every LM-head/NLL call
        # the same shape, avoiding a separate final-chunk compilation.
        readout_chunk = max(1, 1024 // batch)
        top_capacity = max_len + readout_chunk - 1
        self.tops = torch.empty(
            batch, top_capacity, hidden_size, dtype=dtype, device=device
        )
        self.readout_targets = torch.empty(
            batch, top_capacity, dtype=torch.int64, device=device
        )
        self.alpha: Tensor | None = None
        self.beta: Tensor | None = None
        # Fixed-address storage used by the manually captured recurrent
        # CUDA graph.  The graph indexes these buffers with device-resident
        # counters, so one capture covers every steady position.
        self.h_dest = torch.empty(
            batch, max_len, hidden_size, dtype=dtype, device=device
        )
        self.h_s_state = torch.empty(batch, 1, hidden_size, dtype=dtype, device=device)
        self.graph_t = torch.ones(1, dtype=torch.int64, device=device)
        self.graph_prev = torch.zeros(1, dtype=torch.int64, device=device)
        self.graph_seqlens = self.seq_table[0].clone() if max_len > 1 else None

    def bind_alpha(self, cfg: WireConfig) -> None:
        values = [cfg.alpha_at(t) for t in range(self.k.shape[2])]
        self.alpha = torch.tensor(values, dtype=self.k.dtype, device=self.k.device)
        self.beta = 1 - self.alpha

    def step(self, t: int) -> None:
        if t == 0:
            self.rows = self.B
        else:
            self.rows = 2 * self.B
            self.seqlens = self.seq_table[t - 1]
            self.write_index = self.index_table[t - 1 : t]


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
        self._cache: BranchCache | None = None
        self._rope: dict[str, tuple[Tensor, Tensor]] | None = None
        self._steady_graph: torch.cuda.CUDAGraph | None = None
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

    @staticmethod
    def _pack_parameter_family(parameters) -> Tensor:
        """Pack along output rows and make each Parameter a disjoint view."""
        packed = torch.cat([p.detach() for p in parameters], dim=0).contiguous()
        for parameter, view in zip(
            parameters, packed.split([p.shape[0] for p in parameters], dim=0)
        ):
            parameter.data = view
        return packed

    def _build_packed_weights(self) -> None:
        """Pack frozen projections and retire their duplicate original storage."""
        qkv_weights, qkv_biases, gate_up_weights = [], [], []
        aliased_parameters = []
        for layer in self.layers:
            attn = layer.self_attn
            weights = (attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight)
            qkv_weights.append(self._pack_parameter_family(weights))
            aliased_parameters.extend(weights)
            biases = (attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias)
            if any(bias is None for bias in biases):
                if not all(bias is None for bias in biases):
                    raise ValueError("Q/K/V projections must agree on bias presence")
                qkv_biases.append(None)
            else:
                qkv_biases.append(self._pack_parameter_family(biases))
                aliased_parameters.extend(biases)
            gate_up = (layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight)
            gate_up_weights.append(self._pack_parameter_family(gate_up))
            aliased_parameters.extend(gate_up)
        self._qkv_weights = tuple(qkv_weights)
        self._qkv_biases = tuple(qkv_biases)
        self._gate_up_weights = tuple(gate_up_weights)

        # Safetensors rejects shared backing storage even when views are
        # disjoint. Clone only while producing a state_dict; the live model
        # keeps the packed aliases, and load_state_dict writes through them.
        parameter_names = {id(p): name for name, p in self.model.named_parameters()}
        keys = {parameter_names[id(p)] for p in aliased_parameters}
        if not hasattr(self.model, "_wire_packed_state_dict_keys"):
            self.model._wire_packed_state_dict_keys = set()
            self.model._wire_packed_state_dict_hook = (
                self.model.register_state_dict_post_hook(_clone_packed_state_dict_views)
            )
        self.model._wire_packed_state_dict_keys.update(keys)

    def _ensure_rope(
        self, dtype: torch.dtype, device: torch.device
    ) -> dict[str, tuple[Tensor, Tensor]]:
        rope = self._rope
        if (
            rope is None
            or next(iter(rope.values()))[0].dtype != dtype
            or next(iter(rope.values()))[0].device != device
        ):
            marker = torch.empty(1, 1, self.hidden_size, dtype=dtype, device=device)
            pos_ids = torch.arange(self.window, device=device).unsqueeze(0)
            self._rope = {
                lt: self.rotary(marker, pos_ids, lt) for lt in set(self.layer_types)
            }
        return self._rope

    def mix(self, h_s: Tensor, h_d: Tensor, alpha, beta) -> Tensor:
        ratio = h_d.norm(dim=-1, keepdim=True) / h_s.norm(
            dim=-1, keepdim=True
        ).clamp_min(self.cfg.eps)
        return beta * h_d + alpha * ratio * h_s

    def _dual_attention(
        self, i: int, q: Tensor, k_new: Tensor, v_new: Tensor, scale: float
    ) -> Tensor:
        """Exact two-branch attention with one physical refreshed prefix."""
        cache = self._cache
        j = i - cache.first
        return _dual_branch_attention_op(
            q,
            cache.k[j],
            cache.v[j],
            cache.side_k[j],
            cache.side_v[j],
            k_new,
            v_new,
            cache.seqlens,
            cache.cache_batch_idx,
            cache.write_index,
            scale,
        )

    def _layer(self, i: int, x: Tensor, pe, cache: BranchCache | None) -> Tensor:
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
        if cache is not None and cache.rows != cache.B:
            # pe holds only [current, previous]. Reshape the interleaved
            # branch batch so those two rows broadcast over every example;
            # Inductor fuses the gather/broadcast with rotary pointwise work.
            cos, sin = pe
            q = q.reshape(cache.B, 2, *q.shape[1:])
            k = k.reshape(cache.B, 2, *k.shape[1:])
            cos = cos.reshape(1, 2, 1, 1, self.head_dim)
            sin = sin.reshape(1, 2, 1, 1, self.head_dim)

            def rotate_half(z: Tensor) -> Tensor:
                left, right = z.chunk(2, dim=-1)
                return torch.cat((-right, left), dim=-1)

            q = (q * cos + rotate_half(q) * sin).flatten(0, 1)
            k = (k * cos + rotate_half(k) * sin).flatten(0, 1)
        else:
            q, k = apply_rotary_pos_emb(q, k, *pe)

        if cache is None:
            a = _fa2_prefill_op(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn.scaling
            )
        elif cache.rows == cache.B:
            q_new = q.transpose(1, 2)
            k_new = k.transpose(1, 2)
            v_new = v.transpose(1, 2)
            a = _fa2_prefill_op(q_new, k_new, v_new, attn.scaling)
            j = i - cache.first
            cache.side_k[j].copy_(k_new)
            cache.side_v[j].copy_(v_new)
        else:
            a = self._dual_attention(
                i,
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
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
            pe = (
                (cos_f, sin_f)
                if self.layer_types[i] == "full_attention"
                else (cos_s, sin_s)
            )
            h = self._layer(i, h, pe, None)
        return h

    def _slab(self, x: Tensor, cos_f, sin_f, cos_s, sin_s):
        """One compiled step over layers dest+1..top."""
        cache = self._cache
        h_s = x[: cache.B] if cache.rows == cache.B else x[0::2]
        for i in range(self.cfg.dest + 1, self.n_layers):
            pe = (
                (cos_f, sin_f)
                if self.layer_types[i] == "full_attention"
                else (cos_s, sin_s)
            )
            x = self._layer(i, x, pe, cache)
            if i == self.cfg.source:
                h_s = x[: cache.B] if cache.rows == cache.B else x[0::2]
        return x, h_s

    def _step(
        self,
        h_t: Tensor,
        h_prev: Tensor,
        h_s_prev: Tensor,
        alpha,
        beta,
        position: Tensor,
        previous: Tensor,
        *rope,
    ):
        x = torch.stack([h_t, self.mix(h_s_prev, h_prev, alpha, beta)], dim=1).flatten(
            0, 1
        )
        positions = torch.cat((position, previous))
        selected_rope = tuple(
            table.index_select(1, positions).squeeze(0) for table in rope
        )
        return self._slab(x, *selected_rope)

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

    def _capture_steady_graph(self, cache: BranchCache) -> None:
        """Capture one position-independent recurrent step.

        All position selection, recurrent-state threading, top-state storage,
        cache writes, and counter advancement live in the CUDA graph.  Its
        only host work per position is ``graph.replay()``.
        """
        if cache.graph_seqlens is None:
            return
        cache.rows = 2 * cache.B
        cache.seqlens = cache.graph_seqlens
        cache.write_index = cache.graph_prev
        # The captured path gathers contiguous one-position inputs whereas
        # the eager controller passes strided views.  Compile that exact
        # signature before stream capture: Dynamo consults the CUDA RNG
        # state while compiling, which CUDA correctly forbids mid-capture.
        t = cache.graph_t
        prev = cache.graph_prev
        rope = self._ensure_rope(cache.k.dtype, cache.k.device)
        cf, sf = rope["full_attention"]
        cs, ss = rope["sliding_attention"]
        cache.h_s_state.zero_()
        self._step_c(
            cache.h_dest.index_select(1, t),
            cache.h_dest.index_select(1, prev),
            cache.h_s_state,
            cache.alpha.index_select(0, prev),
            cache.beta.index_select(0, prev),
            t,
            prev,
            cf,
            sf,
            cs,
            ss,
        )
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream(device=cache.k.device)
        current = torch.cuda.current_stream(cache.k.device)
        capture_stream.wait_stream(current)
        with (
            torch.cuda.stream(capture_stream),
            torch.cuda.graph(graph, stream=capture_stream),
        ):
            t = cache.graph_t
            prev = cache.graph_prev
            h_t = cache.h_dest.index_select(1, t)
            h_prev = cache.h_dest.index_select(1, prev)
            x, h_s = self._step_c(
                h_t,
                h_prev,
                cache.h_s_state,
                cache.alpha.index_select(0, prev),
                cache.beta.index_select(0, prev),
                t,
                prev,
                cf,
                sf,
                cs,
                ss,
            )
            cache.tops.index_copy_(1, t, x[0::2])
            cache.h_s_state.copy_(h_s)
            cache.graph_seqlens.add_(1)
            cache.graph_t.add_(1)
            cache.graph_prev.add_(1)
        current.wait_stream(capture_stream)
        self._steady_graph = graph

    def _ensure_cache(self, B: int, T: int, dtype, device) -> BranchCache:
        c = self._cache
        if (
            c is None
            or c.B != B
            or c.k.shape[2] < T
            or c.k.dtype != dtype
            or c.k.device != torch.device(device)
        ):
            # flash-attn 2.8's kvcache kernel requires a physical cache
            # length of at least four even when the visible sequence is
            # shorter. cache_seqlens still enforces the exact T=1..3 view.
            capacity = max(T, 4)
            self._cache = BranchCache(
                self.n_layers - self.cfg.dest - 1,
                self.cfg.dest + 1,
                B,
                self.kv_heads,
                capacity,
                self.head_dim,
                self.hidden_size,
                dtype,
                device,
            )
            self._cache.bind_alpha(self.cfg)
            self._steady_graph = None
        return self._cache

    def _run(self, input_ids: Tensor, alpha_fn) -> BranchCache:
        """Pass A + the serial slab; fills cache.tops[:, :T] with the final
        layer's states. Shared core of the two readouts (teacher-forced NLL
        and answer-position logits); callers hold inference_mode."""
        B, T = input_ids.shape
        if not 1 <= T <= self.window:
            raise ValueError(f"wire requires 1 <= T <= sliding_window ({self.window})")
        if not input_ids.is_cuda:
            raise ValueError("the canonical wire is CUDA-only (D10)")
        device = input_ids.device
        dtype = self.embed.weight.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"the canonical wire is half-precision only, model is {dtype}"
            )
        rope = self._ensure_rope(dtype, device)
        cf, sf = rope["full_attention"]
        cs, ss = rope["sliding_attention"]

        # Allocate the shape bucket before pass A, then run one fixed B=64
        # compiled prefill signature. Padding only the final chunk avoids
        # batch-tail recompiles; its duplicate rows are discarded. Each
        # result is copied immediately into the destination-state cache, so
        # no full-batch pass-A output remains live at the prefill peak.
        cache = self._ensure_cache(B, T, dtype, device)
        h_dest = cache.h_dest[:, :T]
        for chunk_index, start in enumerate(range(0, B, _PREFILL_BATCH)):
            valid = min(_PREFILL_BATCH, B - start)
            # index_select gives full and padded chunks the same dispatch
            # keys and layout, so Dynamo retains one prefill graph even
            # when B is not divisible by 64.
            ids = input_ids.index_select(0, cache.prefill_rows[chunk_index])
            prefill = self._prefill_c(
                ids, cf[:, :T], sf[:, :T], cs[:, :T], ss[:, :T]
            )
            h_dest[start : start + valid].copy_(prefill[:valid])

        # Serial slab: layers dest+1..top, batched dual pass. The compiled
        # recurrent step gathers [position t, position t-1] from the compact
        # RoPE tables and broadcasts that pair across interleaved branches.
        tops = cache.tops[:, :T]
        h_s_prev: Tensor | None = None
        cache.step(0)
        x, h_s = self._slab_first_c(
            h_dest[:, :1], cf[:, :1], sf[:, :1], cs[:, :1], ss[:, :1]
        )
        tops[:, :1] = x[:B]
        h_s_prev = h_s

        if T > 1 and alpha_fn is None:
            if self._steady_graph is None:
                self._capture_steady_graph(cache)
                # The exact-signature warmup writes branch/cache scratch.
                # Re-run the unique first step to restore its side KV and
                # source state before the first real replay.
                cache.step(0)
                x, h_s_prev = self._slab_first_c(
                    h_dest[:, :1], cf[:, :1], sf[:, :1], cs[:, :1], ss[:, :1]
                )
                tops[:, :1] = x[:B]
            cache.h_s_state.copy_(h_s_prev)
            cache.graph_t.fill_(1)
            cache.graph_prev.zero_()
            cache.graph_seqlens.copy_(cache.seq_table[0])
            cache.rows = 2 * B
            cache.seqlens = cache.graph_seqlens
            cache.write_index = cache.graph_prev
            for _ in range(1, T):
                self._steady_graph.replay()
        else:
            for t in range(1, T):
                cache.step(t)
                if alpha_fn is None:
                    ab = (cache.alpha[t - 1], cache.beta[t - 1])
                else:
                    ab = alpha_fn(t - 1, h_s_prev, h_dest[:, t - 1 : t])
                    ab = tuple(
                        q
                        if isinstance(q, Tensor)
                        else torch.tensor(q, dtype=dtype, device=device)
                        for q in ab
                    )
                x, h_s = self._step_c(
                    h_dest[:, t : t + 1],
                    h_dest[:, t - 1 : t],
                    h_s_prev,
                    *ab,
                    cache.index_table[t : t + 1],
                    cache.index_table[t - 1 : t],
                    cf,
                    sf,
                    cs,
                    ss,
                )
                tops[:, t : t + 1] = x[0::2]
                h_s_prev = h_s
        return cache

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
        cache = self._run(input_ids, alpha_fn)
        device = input_ids.device

        # Deferred readout: chunked over positions to bound the fp32
        # softmax footprint at Gemma3's 262k vocab. Scratch-padding the
        # final chunk gives every compiled call one shape; padded results
        # are discarded. The NLL-only graph fuses log-softmax + gather.
        nlls, logits = [], []
        work_len = T if return_logits else T - 1
        if work_len == 0:
            return torch.empty(B, 0, device=device), None
        chunk = min(work_len, max(1, 1024 // B))
        padded_len = ((work_len + chunk - 1) // chunk) * chunk
        if padded_len > T:
            cache.tops[:, T:padded_len] = cache.tops[:, T - 1 : T]
        cache.readout_targets[:, : T - 1].copy_(input_ids[:, 1:T])
        if padded_len > T - 1:
            cache.readout_targets[:, T - 1 : padded_len] = input_ids[:, -1:]

        if return_logits:
            for i in range(0, padded_len, chunk):
                lg = self._logits_chunk_c(cache.tops[:, i : i + chunk])
                logits.append(lg)
                nlls.append(
                    self._nll_from_logits_c(lg, cache.readout_targets[:, i : i + chunk])
                )
        else:
            for i in range(0, padded_len, chunk):
                nlls.append(
                    self._nll_chunk_c(
                        cache.tops[:, i : i + chunk],
                        cache.readout_targets[:, i : i + chunk],
                    )
                )
        nll = torch.cat(nlls, dim=1)[:, : T - 1]
        return nll, (torch.cat(logits, dim=1)[:, :T] if return_logits else None)

    def teacher_forced_logits(self, input_ids: Tensor, alpha_fn=None) -> Tensor:
        """First-pass logits [B, T, V]; small T only (full-vocab memory)."""
        return self.teacher_forced(input_ids, alpha_fn, return_logits=True)[1]

    @torch.inference_mode()
    def answer_logits(self, input_ids: Tensor, alpha_fn=None) -> Tensor:
        """Full-vocab logits [B, V] at the final position only.

        The task readout (design.md D3/D4): the last think token predicts
        the answer, so forced-choice scoring needs exactly one position —
        no per-dot NLL is ever computed, mirroring D9's answer-only
        supervision. Same tops and compiled head as teacher_forced_logits'
        last column, at chunk width 1."""
        T = input_ids.shape[1]
        cache = self._run(input_ids, alpha_fn)
        return self._logits_chunk_c(cache.tops[:, T - 1 : T]).squeeze(1)
