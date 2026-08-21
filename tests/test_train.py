from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("flash_attn")

from recirculated_dot_experiment.train import (
    Surface,
    _checkpoint_for,
    _step_choice,
)


def test_checkpoint_policy_uses_measured_memory_knee():
    assert not _checkpoint_for("auto", 256, 8)
    assert _checkpoint_for("auto", 512, 8)
    assert _checkpoint_for("always", 1, 1)
    assert not _checkpoint_for("never", 4096, 32)


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
