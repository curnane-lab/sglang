import torch

from sglang.srt.models.dflash import CandidateSelector
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=20, suite="base-a-test-npu")


def _expected_path(candidate_ids, scores, uniforms, temperatures, greedy_mask, row):
    previous = 0
    tokens = []
    for slot in range(candidate_ids.shape[1]):
        probabilities = torch.softmax(
            scores[row, slot, previous].float() / temperatures[row], dim=-1
        )
        if greedy_mask[row]:
            previous = scores[row, slot, previous].argmax()
        else:
            previous = (uniforms[row, slot] >= probabilities.cumsum(dim=-1)).sum()
            previous = previous.clamp_max(candidate_ids.shape[-1] - 1)
        tokens.append(candidate_ids[row, slot, previous])
    return torch.stack(tokens)


def test_selector_sample_path_uses_supported_npu_backend():
    selector = CandidateSelector(hidden_size=4, vocab_size=128, state_rank=2, top_k=4)
    candidate_ids = torch.arange(2 * 3 * 4, device="npu").reshape(2, 3, 4)
    scores = torch.randn(2, 3, 4, 4, device="npu")
    uniforms = torch.rand(2, 3, device="npu")
    temperatures = torch.ones(2, device="npu")
    greedy_mask = torch.tensor([True, False], device="npu")

    tokens, q_rows = selector.sample_path(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )

    torch.testing.assert_close(
        tokens[0], _expected_path(candidate_ids, scores, uniforms, temperatures, greedy_mask, 0)
    )
    torch.testing.assert_close(
        tokens[1], _expected_path(candidate_ids, scores, uniforms, temperatures, greedy_mask, 1)
    )
    assert q_rows.shape == (2, 3, 4)
    assert torch.isfinite(q_rows).all()
    path_indices = (candidate_ids[0] == tokens[0][:, None]).to(torch.int64).argmax(dim=-1)
    torch.testing.assert_close(
        q_rows[0], torch.nn.functional.one_hot(path_indices, 4).float()
    )
    torch.testing.assert_close(q_rows[1].sum(dim=-1), torch.ones(3, device="npu"))
