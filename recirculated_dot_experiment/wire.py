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
column), lane 1 the refresh view (refreshed history + its own recompute;
the newest column is masked out). Returns are zero-copy views; the only
data movement is four one-slot writes per layer per step, with lane 0's
refresh committed via `commit(i)` after that layer's attention has
consumed its view.

NLL / logits are computed chunked at the end from collected top hiddens
— per-step fp32 softmax over Gemma3's 262k vocab would dominate runtime
and memory.

Requirements the constructor enforces: sdpa attention (pass A relies on
sdpa synthesizing causality for maskless q_len > 1; eager would attend
bidirectionally there) and sequences within one sliding window, where
sliding and full attention coincide. `final_logit_softcapping` is
applied when the config carries it (None on Gemma3-1B).

Inference-only in v0 (`inference_mode`): the in-place lane buffers break
autograd. The BPTT training path gets a functional token-chunked cache
(design.md, "training-path design") instead. The next perf lever, when
training throughput binds, is torch.compile over the slab step with
bucketed KV lengths — measured 4.8x on the slab in isolation; the
two-lane cache is the shape-stability groundwork for it.

Layer indexing: `source`/`dest` are 0-based indices into model.layers,
and the tapped value is the hidden state *after* that layer. G0
empirically confirmed the paper's Gemma3-1B pair {11, 4} is 0-based
(-8.8% ppl on PG19 vs -1.4% for the {10, 3} reading).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


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
    attention attends over exactly what it returns (plus the engine's
    mask on lane 1's newest slot). Lane buffers are allocated lazily per
    layer, so only slab layers ever hold memory."""

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

    def _lane(self, store: dict[int, Tensor], layer_idx: int) -> Tensor:
        if layer_idx not in store:
            store[layer_idx] = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        return store[layer_idx]

    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int, *a, **kw):
        B, t = self.B, self.t
        k, v = self._lane(self.k, layer_idx), self._lane(self.v, layer_idx)
        if not self.dual:
            k[:, :, t : t + 1] = torch.cat([key_states] * 2)
            v[:, :, t : t + 1] = torch.cat([value_states] * 2)
            return k[:B, :, : t + 1], v[:B, :, : t + 1]
        k1, k2 = key_states[:B], key_states[B:]
        v1, v2 = value_states[:B], value_states[B:]
        k[:B, :, t : t + 1] = k1
        k[B:, :, t : t + 1] = k1  # lane 1's newest slot is masked; content moot
        k[B:, :, t - 1 : t] = k2  # lane 1 attends its own recomputed t-1
        v[:B, :, t : t + 1] = v1
        v[B:, :, t : t + 1] = v1
        v[B:, :, t - 1 : t] = v2
        self.pending[layer_idx] = (k2, v2)
        return k[:, :, : t + 1], v[:, :, : t + 1]

    def commit(self, layer_idx: int) -> None:
        """Refresh lane 0's slot t-1 — call after this layer's attention
        has consumed its first-pass view (same-snapshot semantics)."""
        k2, v2 = self.pending.pop(layer_idx)
        self.k[layer_idx][: self.B, :, self.t - 1 : self.t] = k2
        self.v[layer_idx][: self.B, :, self.t - 1 : self.t] = v2


class RecirculationEngine:
    """Wraps a Gemma3ForCausalLM (weights untouched, model in eval mode)."""

    def __init__(self, model, cfg: WireConfig):
        mc = model.config
        if not 0 <= cfg.dest < cfg.source < mc.num_hidden_layers:
            raise ValueError(f"need 0 <= dest < source < {mc.num_hidden_layers}")
        # pass A relies on sdpa synthesizing causality for maskless q_len > 1;
        # eager attention would attend bidirectionally there (Codex repro:
        # 0.21 max logit error on a tiny model)
        if mc._attn_implementation != "sdpa":
            raise ValueError(f"engine requires sdpa attention, got {mc._attn_implementation!r}")
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

    def mix(self, h_s: Tensor, h_d: Tensor, alpha, beta) -> Tensor:
        ratio = h_d.norm(dim=-1, keepdim=True) / h_s.norm(dim=-1, keepdim=True).clamp_min(
            self.cfg.eps
        )
        return beta * h_d + alpha * ratio * h_s

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
        slab = range(cfg.dest + 1, self.n_layers)
        cache = DualCache(B, self.kv_heads, T, self.head_dim, dtype, device)
        neg = torch.finfo(dtype).min
        tops = torch.empty(B, T, h.shape[-1], dtype=dtype, device=device)
        mask_buf = torch.zeros(2 * B, 1, 1, T, dtype=dtype, device=device)
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
            if t == 0:
                cache.t, cache.dual = 0, False
                x = h_dest[:, :1]
                mask = None
                pe = {lt: (c[:, :1], s[:, :1]) for lt, (c, s) in rope.items()}
            else:
                if alpha_fn is None:
                    a = cfg.alpha_at(t - 1)
                    ab = (a, 1.0 - a)
                else:
                    ab = alpha_fn(t - 1, h_s_prev, h_dest[:, t - 1 : t])
                x = torch.cat([h_dest[:, t : t + 1], self.mix(h_s_prev, h_dest[:, t - 1 : t], *ab)])
                cache.t, cache.dual = t, True
                mask = mask_buf[:, :, :, : t + 1]
                mask[B:, ..., t] = neg  # second half runs concurrently: no slot t
                pe = {lt: (c[:, t - 1 : t], s[:, t - 1 : t]) for lt, (c, s) in rope_dual.items()}
            for i in slab:
                x = self.layers[i](
                    x,
                    position_embeddings=pe[self.layer_types[i]],
                    attention_mask=mask,
                    past_key_values=cache,
                )
                if i == cfg.source:
                    h_s = x[:B]
                if t:
                    cache.commit(i)
            tops[:, t : t + 1] = x[:B]
            if t:
                mask[B:, ..., t] = 0  # restore the reusable mask buffer
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
