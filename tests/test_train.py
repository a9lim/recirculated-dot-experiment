from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("flash_attn")

from recirculated_dot_experiment.train import (
    GateMLP,
    PromptStateCache,
    Surface,
    _checkpoint_for,
    _execution_plan,
    _gate_mix_math,
    _mix,
    _step_choice,
)


def test_checkpoint_policy_uses_measured_memory_knee():
    assert not _checkpoint_for("auto", 256, 8)
    assert _checkpoint_for("auto", 512, 8)
    assert _checkpoint_for("always", 1, 1)
    assert not _checkpoint_for("never", 4096, 32)


def test_execution_plan_keeps_effective_batch_and_compiled_shapes():
    assert _execution_plan("auto", 512, 4, 26).microbatch == 512
    assert _execution_plan("auto", 512, 8, 26).microbatch == 256
    assert _execution_plan("auto", 512, 16, 26).microbatch == 512
    assert not _execution_plan("auto", 512, 8, 26).checkpoint_layers
    assert _execution_plan("auto", 512, 16, 26).checkpoint_layers == (
        frozenset(range(26)) - {7, 13, 19, 25}
    )
    assert len(_execution_plan("auto", 512, 32, 26).checkpoint_layers) == 26
    assert _execution_plan("auto", 510, 8, 26).microbatch == 255


def test_tensor_gate_mix_matches_module_path_and_gradients():
    torch.manual_seed(17)
    gate = GateMLP(8).to(dtype=torch.bfloat16)
    with torch.no_grad():
        gate.out.weight.normal_(std=1e-3)
    source = torch.randn(3, 1, 8, dtype=torch.bfloat16, requires_grad=True)
    dest = torch.randn(3, 1, 8, dtype=torch.bfloat16, requires_grad=True)
    alpha, beta = gate(source, dest)
    expected = _mix(1e-6, source, dest, alpha, beta)
    actual = _gate_mix_math(
        1e-6,
        source,
        dest,
        gate.norm.weight,
        gate.norm.bias,
        gate.h1.weight,
        gate.h1.bias,
        gate.h2.weight,
        gate.h2.bias,
        gate.out.weight,
        gate.out.bias,
    )
    assert torch.equal(expected, actual)

    expected.sum().backward(retain_graph=True)
    expected_grads = [parameter.grad.clone() for parameter in gate.parameters()]
    gate.zero_grad(set_to_none=True)
    actual.sum().backward()
    assert all(
        torch.equal(want, parameter.grad)
        for want, parameter in zip(expected_grads, gate.parameters())
    )


def test_prompt_cache_packed_views_restore_state_without_copies():
    hidden = torch.arange(12, dtype=torch.bfloat16).view(2, 1, 6)
    kv = [
        (
            torch.arange(16, dtype=torch.bfloat16).view(2, 2, 2, 2),
            torch.arange(16, 32, dtype=torch.bfloat16).view(2, 2, 2, 2),
        ),
        (
            torch.arange(32, 48, dtype=torch.bfloat16).view(2, 2, 2, 2),
            torch.arange(48, 64, dtype=torch.bfloat16).view(2, 2, 2, 2),
        ),
    ]
    flat = PromptStateCache._flatten((hidden, kv))
    packed = torch.cat([tensor.reshape(-1) for tensor in flat])
    restored_hidden, restored_kv = PromptStateCache._views(
        packed, tuple(tensor.shape for tensor in flat)
    )
    restored = PromptStateCache._flatten((restored_hidden, restored_kv))

    assert all(torch.equal(before, after) for before, after in zip(flat, restored))
    assert all(
        after.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
        for after in restored
    )


def test_step_schedule_is_addressable_and_deterministic():
    tasks = ["a", "b", "c"]
    ks = [1, 2, 4]
    assert _step_choice(11, 93, tasks, ks) == _step_choice(11, 93, tasks, ks)
    assert [_step_choice(11, i, tasks, ks) for i in range(8)] != [
        _step_choice(12, i, tasks, ks) for i in range(8)
    ]


def test_surface_has_only_bf16_parameters():
    embedding = nn.Embedding(128, 16, dtype=torch.bfloat16)
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=16),
        model=SimpleNamespace(embed_tokens=embedding),
    )
    surface = Surface(model, 7)
    assert {parameter.dtype for parameter in surface.parameters()} == {torch.bfloat16}


def test_surface_rejects_non_bf16_base():
    embedding = nn.Embedding(128, 16, dtype=torch.float32)
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=16),
        model=SimpleNamespace(embed_tokens=embedding),
    )
    with pytest.raises(TypeError, match="requires BF16"):
        Surface(model, 7)
