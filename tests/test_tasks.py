from types import SimpleNamespace

import pytest
import torch
from torch import nn

from recirculated_dot_experiment import tasks


class CharacterTokenizer:
    bos_token_id = 1

    def __init__(self):
        self.ids = {chr(code): code + 100 for code in range(128)}
        self.ids[tasks.DOT] = 99

    def __call__(self, text, add_special_tokens=True):
        encoded = [self.ids[text]] if text == tasks.DOT else [self.ids[c] for c in text]
        if add_special_tokens:
            encoded = [self.bos_token_id, *encoded]
        return SimpleNamespace(input_ids=encoded)


def test_cot_uses_natural_tokenizer_spacing_and_actual_budget():
    tok = CharacterTokenizer()
    instance = tasks.sample(tasks.TASKS["parity"], 1, 7, length=4)[0]
    trace = tasks.render_cot(instance)
    full = tasks.encode(tok, instance, "cot")
    limited = tasks.encode(tok, instance, "cot", 3)
    expected = tok(trace, add_special_tokens=False).input_ids

    assert list(full.ids[full.think[0] :]) == expected
    assert list(limited.ids[limited.think[0] :]) == expected[:3]
    assert tok.ids[" "] in expected


def test_scopes_are_explicitly_named():
    assert tasks.CONDITIONS["dots+think-wire"] == ("dots", "think")
    assert tasks.CONDITIONS["dots+full-wire"] == ("dots", "full")
    assert "dots+wire" not in tasks.CONDITIONS


def test_encode_dot_sweep_reuses_instances_across_budgets():
    tok = CharacterTokenizer()
    instances = tasks.sample(tasks.TASKS["parity"], 3, 19, length=4)
    encoded = tasks.encode_dot_sweep(tok, instances, [1, 2, 4])

    assert list(encoded) == [1, 2, 4]
    assert [len(row.ids) for row in encoded[4]] == [
        len(row.ids) + 3 for row in encoded[1]
    ]
    assert [[row.label for row in encoded[k]] for k in encoded] == [
        [instance.label for instance in instances]
    ] * 3
    with pytest.raises(ValueError, match="positive k"):
        tasks.encode_dot_sweep(tok, instances, [])


class ToyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyInner()


class CountingRunner:
    def __init__(self):
        self.answer_calls = 0
        self.logit_calls = 0

    def answer_hiddens(self, ids, ks):
        self.answer_calls += 1
        return (
            torch.tensor(ks, dtype=torch.bfloat16)
            .view(1, -1, 1)
            .expand(ids.shape[0], -1, -1)
        )

    def logits_from_hidden(self, hidden):
        self.logit_calls += 1
        return torch.cat((-hidden, hidden), dim=-1)


def test_dot_sweep_runs_max_k_once_per_batch():
    encoded = {
        k: [
            tasks.Encoded(
                ids=(1, *([99] * k)),
                answer=1,
                label=1,
                label_ids=(0, 1),
                think=(1, 1 + k),
            )
            for _ in range(3)
        ]
        for k in (1, 2, 4)
    }
    runner = CountingRunner()
    results = tasks.evaluate_dot_sweep(ToyModel(), encoded, runner, batch=4)

    assert runner.answer_calls == 1
    assert runner.logit_calls == 3
    assert all(result["acc"] == 1.0 for result in results.values())


class FreeRunner:
    """Per-position logits shared across rows, looked up by position."""

    def __init__(self, logits):
        self.logits = logits  # [k, V]

    def answer_hiddens(self, ids, ks):
        idx = torch.tensor(ks, dtype=torch.float32).view(1, -1, 1)
        return idx.expand(ids.shape[0], -1, -1)

    def logits_from_hidden(self, hidden):
        return self.logits[hidden[..., 0].long() - 1]


def _free_rows(n, ids=(1, 99, 99, 99)):
    return [
        tasks.Encoded(ids=ids, answer=3, label=1, label_ids=(2, 3), think=(1, 4))
        for _ in range(n)
    ]


def _free_logits(rows):
    """rows: per position, {token: logit}; everything else is ~impossible."""
    logits = torch.full((len(rows), 100), -1e9)
    for position, values in enumerate(rows):
        for token, value in values.items():
            logits[position, token] = value
    return logits


def test_free_running_greedy_and_analytic_match_monte_carlo():
    logits = _free_logits(
        [
            {99: 2.0, 3: 1.0, 2: 0.0, 7: -1.0},  # argmax dot: continue
            {99: 1.0, 3: 2.0, 2: 0.0, 7: -1.0},  # argmax gold: greedy halt
            {99: 0.0, 3: 0.5, 2: 0.5, 7: 2.0},  # argmax illegal
        ]
    )
    result = tasks.evaluate_free_running(
        ToyModel(), _free_rows(3), FreeRunner(logits), batch=2
    )

    assert (result["halt"], result["acc"], result["legal"]) == (1.0, 1.0, 1.0)
    assert result["k_halt"] == 2.0

    probs = logits.float().softmax(-1)
    p_dot = probs[:, 99]
    assert abs(result["p_halt"] - (1 - p_dot.prod()).item()) < 1e-6
    assert result["p_gold"] <= result["p_legal"] <= result["p_halt"]

    generator = torch.Generator().manual_seed(0)
    n = 200_000
    samples = torch.stack(
        [
            torch.multinomial(probs[t], n, replacement=True, generator=generator)
            for t in range(3)
        ]
    )
    positions = torch.arange(1, 4).view(3, 1).expand(3, n)
    first = torch.where(samples != 99, positions, torch.full_like(positions, 99))
    first = first.min(0).values
    halted = first < 99
    emitted = samples.gather(0, (first.clamp(max=3) - 1).unsqueeze(0)).squeeze(0)
    assert abs(result["p_halt"] - halted.float().mean().item()) < 0.01
    assert (
        abs(result["p_gold"] - (halted & (emitted == 3)).float().mean().item()) < 0.01
    )
    legal = halted & ((emitted == 2) | (emitted == 3))
    assert abs(result["p_legal"] - legal.float().mean().item()) < 0.01
    mc_k = first[halted].float().mean().item()
    assert abs(result["k_soft"] - mc_k) < 0.05


def test_free_running_reports_never_halting():
    logits = _free_logits([{99: 9.0, 3: 0.0}] * 3)
    result = tasks.evaluate_free_running(
        ToyModel(), _free_rows(2), FreeRunner(logits), batch=4
    )

    assert (result["halt"], result["acc"], result["k_halt"]) == (0.0, 0.0, 0.0)
    probs = logits.float().softmax(-1)
    assert abs(result["p_halt"] - (1 - probs[:, 99].prod()).item()) < 1e-6


def test_free_running_rejects_non_dot_spans():
    with pytest.raises(ValueError, match="dot spans"):
        tasks.evaluate_free_running(
            ToyModel(),
            _free_rows(2, ids=(1, 99, 50, 99)),
            FreeRunner(_free_logits([{99: 1.0}] * 3)),
        )
