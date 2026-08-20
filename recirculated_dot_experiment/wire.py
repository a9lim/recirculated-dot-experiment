"""Two-pass recirculation engine for Gemma3 (design.md D1).

Drives the HF decoder layers manually — no fork of layer internals, no
hooks. Per wall-clock step t, two forwards sharing one cache snapshot:
the first pass of column t (full stack; its logits are the readout) reads
column t-1 at first-pass fidelity, then the second pass of column t-1
(layers dest+1..top, with the layer-dest output replaced by the alpha-mix
of Eq. 1) overwrites t-1's KV entries for those layers. Later columns see
the refreshed state through ordinary attention — that is the wire.

Masking is implicit: WireCache returns exactly the visible prefix, and a
single-token query with attention_mask=None attends to everything returned
(the sdpa interface sets is_causal=False when q_len == 1). This requires
sequences to fit one sliding window (config.sliding_window), where sliding
and full attention coincide and both layer types share the plain prefix
cache; the engine asserts it.

Inference-only in v0: WireCache writes in place, which is fine under
no_grad but not for BPTT — the training path will keep per-step tensors
and cat instead.

Layer indexing: `source`/`dest` are 0-based indices into model.layers, and
the tapped value is the hidden state *after* that layer. The paper's
Gemma3-1B pair {11, 4} is assumed 0-based (their sweeps include a layer 0
destination); the heatmaps are smooth, so if G0 comes in weak, sweep +-1.
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


class WireCache:
    """Preallocated per-layer KV with externally controlled write position
    and visible prefix. `update` matches the transformers Cache calling
    convention: attention attends over exactly what it returns."""

    def __init__(
        self,
        n_layers: int,
        batch: int,
        kv_heads: int,
        max_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        shape = (batch, kv_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.write_pos = 0
        self.visible = 0

    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int, *a, **kw):
        p, n = self.write_pos, key_states.shape[2]
        self.k[layer_idx][:, :, p : p + n] = key_states
        self.v[layer_idx][:, :, p : p + n] = value_states
        return (
            self.k[layer_idx][:, :, : self.visible],
            self.v[layer_idx][:, :, : self.visible],
        )


class RecirculationEngine:
    """Wraps a Gemma3ForCausalLM (weights untouched, model in eval mode)."""

    def __init__(self, model, cfg: WireConfig):
        mc = model.config
        if not 0 <= cfg.dest < cfg.source < mc.num_hidden_layers:
            raise ValueError(f"need 0 <= dest < source < {mc.num_hidden_layers}")
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

    def mix(self, h_s: Tensor, h_d: Tensor, alpha, beta) -> Tensor:
        ratio = h_d.norm(dim=-1, keepdim=True) / h_s.norm(dim=-1, keepdim=True).clamp_min(
            self.cfg.eps
        )
        return beta * h_d + alpha * ratio * h_s

    @torch.no_grad()
    def teacher_forced(
        self, input_ids: Tensor, alpha_fn=None, return_logits: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        """Full-scope recirculation; returns (nll [B, T-1], logits | None).

        NLL of each next token is computed per step in fp32 and the
        full-vocab logits are discarded unless requested — at Gemma3's
        262k vocab, T=512 logits alone are ~4 GB per batch of 16.

        alpha_fn(t, h_s, h_d) -> (alpha, beta), scalars or [B, 1, D]
        tensors; None uses the config constants with the ramp (convex,
        beta = 1 - alpha_t). This is the hook the adaptive gate MLP
        replaces later.
        """
        B, T = input_ids.shape
        if T > self.window:
            raise ValueError(f"wire v0 requires T <= sliding_window ({self.window})")
        device = input_ids.device
        dtype = self.embed.weight.dtype
        cache = WireCache(
            self.n_layers, B, self.kv_heads, T, self.head_dim, dtype, device
        )
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        probe = self.embed(input_ids[:, :1])
        rope = {lt: self.rotary(probe, pos_ids, lt) for lt in set(self.layer_types)}

        logits: list[Tensor] = []
        nlls: list[Tensor] = []
        h_s_prev = h_d_prev = None
        for t in range(T):
            h = self.embed(input_ids[:, t : t + 1])
            cache.write_pos, cache.visible = t, t + 1
            h_s = h_d = None
            for i, layer in enumerate(self.layers):
                lt = self.layer_types[i]
                cos, sin = rope[lt]
                h = layer(
                    h,
                    position_embeddings=(cos[:, t : t + 1], sin[:, t : t + 1]),
                    attention_mask=None,
                    position_ids=pos_ids[:, t : t + 1],
                    past_key_values=cache,
                )
                if i == self.cfg.dest:
                    h_d = h
                elif i == self.cfg.source:
                    h_s = h
            logit_t = self.lm_head(self.final_norm(h))
            if t < T - 1:
                logprobs = torch.log_softmax(logit_t.float().squeeze(1), dim=-1)
                target = input_ids[:, t + 1 : t + 2]
                nlls.append(-logprobs.gather(-1, target))
            if return_logits:
                logits.append(logit_t)

            # Second pass of column t-1: runs after column t's first pass
            # has read the cache (same-snapshot semantics), then overwrites.
            if t >= 1:
                if alpha_fn is None:
                    a = self.cfg.alpha_at(t - 1)
                    ab = (a, 1.0 - a)
                else:
                    ab = alpha_fn(t - 1, h_s_prev, h_d_prev)
                h2 = self.mix(h_s_prev, h_d_prev, *ab)
                cache.write_pos, cache.visible = t - 1, t
                for i in range(self.cfg.dest + 1, self.n_layers):
                    lt = self.layer_types[i]
                    cos, sin = rope[lt]
                    h2 = self.layers[i](
                        h2,
                        position_embeddings=(cos[:, t - 1 : t], sin[:, t - 1 : t]),
                        attention_mask=None,
                        position_ids=pos_ids[:, t - 1 : t],
                        past_key_values=cache,
                    )
            h_s_prev, h_d_prev = h_s, h_d
        nll = torch.cat(nlls, dim=1)
        return nll, (torch.cat(logits, dim=1) if return_logits else None)

    def teacher_forced_logits(self, input_ids: Tensor, alpha_fn=None) -> Tensor:
        """First-pass logits [B, T, V]; small T only (full-vocab memory)."""
        return self.teacher_forced(input_ids, alpha_fn, return_logits=True)[1]
