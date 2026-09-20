import pytest

torch = pytest.importorskip("torch")

from scm.recall import SUITE, Recall, RecallError, Score, mean_logprob, span_tokens


class WordTok:
    def encode(self, text, add_special_tokens=True):
        return [abs(hash(w)) % 1000 for w in text.split()]


TOK = WordTok()


def probe(**kw) -> Recall:
    base = dict(name="p", prefix="notes here ", span="the code is ZX9 ",
                bridge="I used that code later ", question="what was the code ",
                answer="ZX9")
    return Recall(**{**base, **kw})


def test_full_and_edited_differ_by_the_span():
    p = probe()
    assert p.full == "notes here the code is ZX9 I used that code later what was the code "
    assert "ZX9" not in p.edited


def test_a_probe_without_a_bridge_is_rejected():
    """No bridge means nothing downstream remembers, so there is no residue path."""
    with pytest.raises(RecallError, match="no bridge"):
        probe(bridge="")


def test_an_answer_visible_outside_the_span_is_rejected():
    """Otherwise the model reads it off instead of recalling it."""
    with pytest.raises(RecallError, match="appears outside the span"):
        probe(bridge="I used ZX9 later ")
    with pytest.raises(RecallError, match="appears outside the span"):
        probe(question="was the code ZX9 ")


def test_empty_span_or_answer_is_rejected():
    with pytest.raises(RecallError, match="span is empty"):
        probe(span="")
    with pytest.raises(RecallError, match="no answer"):
        probe(answer="")


@pytest.mark.parametrize("p", SUITE, ids=lambda p: p.name)
def test_every_suite_probe_cuts_cleanly(p):
    span_tokens(p, TOK)


@pytest.mark.parametrize("p", SUITE, ids=lambda p: p.name)
def test_every_suite_probe_hides_its_answer_downstream(p):
    assert p.answer not in p.bridge and p.answer not in p.question


def test_span_tokens_covers_the_fact():
    assert span_tokens(probe(), TOK) == (2, 6)


def test_a_ragged_span_is_rejected():
    class Gluing:
        def encode(self, text, add_special_tokens=True):
            return [abs(hash(text[i:i + 3])) % 1000 for i in range(0, len(text), 3)]

    with pytest.raises(RecallError, match="token boundaries"):
        span_tokens(probe(), Gluing())


# --- Score.recoverable -----------------------------------------------------


def test_recoverable_is_one_when_the_splice_matches_full():
    assert Score(full=-1.0, reprefill=-5.0, spliced=-1.0).recoverable == 1.0


def test_recoverable_is_zero_when_the_splice_matches_reprefill():
    assert Score(full=-1.0, reprefill=-5.0, spliced=-5.0).recoverable == 0.0


def test_recoverable_is_a_fraction_in_between():
    assert Score(full=-1.0, reprefill=-5.0, spliced=-3.0).recoverable == 0.5


def test_recoverable_refuses_when_the_fact_did_not_help():
    """Nothing to recover if knowing the fact made the answer no likelier."""
    with pytest.raises(RecallError, match="nothing to recover"):
        Score(full=-5.0, reprefill=-5.0, spliced=-5.0).recoverable
    with pytest.raises(RecallError, match="nothing to recover"):
        Score(full=-6.0, reprefill=-5.0, spliced=-5.0).recoverable


# --- mean_logprob ----------------------------------------------------------


def test_a_confident_prediction_scores_near_zero():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    assert mean_logprob(logits, torch.tensor([0])) > -0.01


def test_an_unlikely_prediction_scores_low():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    assert mean_logprob(logits, torch.tensor([2])) < -9.0


def test_scores_average_across_tokens():
    logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    both = mean_logprob(logits, torch.tensor([0, 1]))
    assert mean_logprob(logits, torch.tensor([0, 0])) > both > mean_logprob(
        logits, torch.tensor([1, 1]))


def test_mismatched_lengths_are_rejected():
    with pytest.raises(RecallError, match="2 logit rows for 3 targets"):
        mean_logprob(torch.zeros(2, 5), torch.tensor([0, 1, 2]))


def test_no_targets_is_rejected():
    with pytest.raises(RecallError, match="no target tokens"):
        mean_logprob(torch.zeros(0, 5), torch.tensor([], dtype=torch.long))
