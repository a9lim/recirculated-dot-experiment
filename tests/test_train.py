from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("flash_attn")

from recirculated_dot_experiment.train import (
    GateMLP,
    PromptStateCache,
    Surface,
    _execution_plan,
    _gate_mix_math,
    _lr_at,
    _mix,
    _step_choice,
)


def test_execution_plan_keeps_effective_batch_and_compiled_shapes():
    assert _execution_plan(512, 4, 26).microbatch == 512
    assert _execution_plan(512, 8, 26).microbatch == 256
    assert _execution_plan(512, 16, 26).microbatch == 512
    assert not _execution_plan(512, 8, 26).checkpoint_layers
    assert _execution_plan(512, 16, 26).checkpoint_layers == (
        frozenset(range(26)) - {7, 13, 19, 25}
    )
    assert len(_execution_plan(512, 32, 26).checkpoint_layers) == 26
    assert _execution_plan(510, 8, 26).microbatch == 255


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
    weights = [1.0, 2.0, 4.0]
    assert _step_choice(11, 93, tasks, ks) == _step_choice(11, 93, tasks, ks)
    assert _step_choice(11, 93, tasks, ks, weights) == _step_choice(
        11, 93, tasks, ks, weights
    )
    assert [_step_choice(11, i, tasks, ks) for i in range(8)] != [
        _step_choice(12, i, tasks, ks) for i in range(8)
    ]


def test_step_schedule_weights_tilt_and_exclude():
    ks = [2, 4, 8, 16, 32]
    weights = [float(k) for k in ks]
    drawn = [_step_choice(7, i, ["a"], ks, weights)[1] for i in range(2000)]
    assert 1 not in drawn
    assert drawn.count(32) > 3 * drawn.count(2)
    # the task draw is unaffected by the k weighting
    assert all(
        _step_choice(7, i, ["a", "b"], ks)[0]
        == _step_choice(7, i, ["a", "b"], ks, weights)[0]
        for i in range(50)
    )


def test_lr_schedule_is_pure_and_run_length_independent():
    base = {"lr": 1e-3, "warmup": 0.05, "cosine": 2000, "lr_floor": 1e-4}
    args = SimpleNamespace(**base, steps=2000)
    assert _lr_at(0, args) == 0.0
    assert _lr_at(50, args) == pytest.approx(5e-4)
    assert _lr_at(100, args) == pytest.approx(1e-3)  # cosine starts at the peak
    assert _lr_at(1050, args) == pytest.approx((1e-3 + 1e-4) / 2)  # midpoint
    assert _lr_at(2000, args) == pytest.approx(1e-4)  # horizon ends at the floor
    assert _lr_at(9000, args) == pytest.approx(1e-4)  # flat tail past the horizon
    # the schedule never reads the run length
    longer = SimpleNamespace(**base, steps=5000)
    assert all(_lr_at(s, args) == _lr_at(s, longer) for s in range(0, 5001, 97))
    flat = SimpleNamespace(**{**base, "cosine": 0}, steps=2000)
    assert _lr_at(1100, flat) == pytest.approx(1e-3)
    assert _lr_at(50, flat) == pytest.approx(1e-3)


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
