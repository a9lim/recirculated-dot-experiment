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
    gate = GateMLP(8)
    with torch.no_grad():
        gate.out.weight.normal_(std=1e-3 / gate.out_scale)
    source = torch.randn(3, 1, 8, dtype=torch.bfloat16, requires_grad=True)
    dest = torch.randn(3, 1, 8, dtype=torch.bfloat16, requires_grad=True)
    alpha, beta = gate(source, dest)
    assert alpha.dtype == beta.dtype == torch.float32
    expected = _mix(1e-6, source, dest, alpha, beta)
    assert expected.dtype == torch.bfloat16
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
        gate.out_scale,
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


def test_lr_schedule_scales_a_group_by_its_own_peak_and_floor():
    base = {"lr": 1e-3, "warmup": 0.05, "cosine": 2000, "lr_floor": 1e-4}
    args = SimpleNamespace(**base, steps=2000)
    gate_peak, gate_floor = 1e-4, 1e-5  # gate_lr/lr = 0.1, floor scaled likewise
    # warmup and cosine keep the 10x ratio at every step
    for step in (0, 50, 100, 1050, 2000, 9000):
        assert _lr_at(step, args, gate_peak, gate_floor) == pytest.approx(
            0.1 * _lr_at(step, args)
        )
    # cosine 0 is flat at the group's own peak
    flat = SimpleNamespace(**{**base, "cosine": 0}, steps=2000)
    assert _lr_at(500, flat, gate_peak, gate_floor) == pytest.approx(gate_peak)


def test_gate_starts_at_paper_constants_and_scales_output_by_fan_in():
    torch.manual_seed(3)
    d = 8
    gate = GateMLP(d)
    assert gate.out_scale == 1.0 / d
    source = torch.randn(5, 1, d, dtype=torch.bfloat16)
    dest = torch.randn(5, 1, d, dtype=torch.bfloat16)
    alpha, beta = gate(source, dest)
    # zero-init output: exactly the paper constants, input-independent
    assert torch.allclose(alpha, torch.full_like(alpha, 0.1), atol=1e-6)
    assert torch.allclose(beta, torch.full_like(beta, 0.9), atol=1e-6)
    with torch.no_grad():
        gate.out.weight.normal_()
    alpha, beta = gate(source, dest)
    x = torch.cat((source, dest), dim=-1).float()
    x = torch.nn.functional.gelu(gate.h1(gate.norm(x)))
    x = torch.nn.functional.gelu(gate.h2(x))
    z = x @ gate.out.weight.T / d + gate.out.bias
    want_alpha, want_beta = torch.sigmoid(z).chunk(2, dim=-1)
    assert torch.allclose(alpha, want_alpha, atol=1e-6)
    assert torch.allclose(beta, want_beta, atol=1e-6)


def test_mix_computes_in_fp32_and_returns_the_residual_dtype():
    torch.manual_seed(5)
    source = torch.randn(4, 1, 8, dtype=torch.bfloat16)
    dest = torch.randn(4, 1, 8, dtype=torch.bfloat16)
    alpha = torch.rand(4, 1, 8)
    beta = torch.rand(4, 1, 8)
    mixed = _mix(1e-6, source, dest, alpha, beta)
    assert mixed.dtype == torch.bfloat16
    s, d = source.float(), dest.float()
    ratio = d.norm(dim=-1, keepdim=True) / s.norm(dim=-1, keepdim=True)
    assert torch.equal(mixed, (beta * d + alpha * ratio * s).to(torch.bfloat16))


def test_surface_holds_fp32_parameters_over_a_bf16_base():
    embedding = nn.Embedding(128, 16, dtype=torch.bfloat16)
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=16),
        model=SimpleNamespace(embed_tokens=embedding),
    )
    surface = Surface(model, 7)
    assert {parameter.dtype for parameter in surface.parameters()} == {torch.float32}
    assert torch.equal(surface.row, embedding.weight[7].float())


def test_surface_rejects_non_bf16_base():
    embedding = nn.Embedding(128, 16, dtype=torch.float32)
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=16),
        model=SimpleNamespace(embed_tokens=embedding),
    )
    with pytest.raises(TypeError, match="requires BF16"):
        Surface(model, 7)
