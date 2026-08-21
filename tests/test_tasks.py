from types import SimpleNamespace

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
