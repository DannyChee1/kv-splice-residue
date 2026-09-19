import pytest
import torch

from scm.agreement import (
    BOTH,
    FULL,
    NEITHER,
    REPREFILL,
    Decode,
    Trial,
    common_prefix_length,
    kl_divergence,
    summarize,
    top_k_overlap,
)


def logits(*weights: float) -> torch.Tensor:
    return torch.tensor(weights, dtype=torch.float32)


def decode(*tokens: int, first=None) -> Decode:
    return Decode(tokens=tokens, first_logits=first)


def trial(full, reprefill, spliced) -> Trial:
    return Trial(full=full, reprefill=reprefill, spliced=spliced)


# --- common_prefix_length --------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((1, 2, 3), (1, 2, 3), 3),
        ((1, 2, 3), (1, 2, 9), 2),
        ((1, 2, 3), (9, 2, 3), 0),
        ((1, 2), (1, 2, 3), 2),
        ((), (1,), 0),
    ],
)
def test_common_prefix_length(a, b, expected):
    assert common_prefix_length(a, b) == expected


# --- kl_divergence ---------------------------------------------------------


def test_kl_is_zero_for_identical_distributions():
    p = logits(1.0, 2.0, 3.0)
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)


def test_kl_grows_as_distributions_separate():
    p = logits(5.0, 0.0, 0.0)
    near = kl_divergence(p, logits(4.0, 0.0, 0.0))
    far = kl_divergence(p, logits(0.0, 0.0, 5.0))
    assert 0 < near < far


def test_kl_ignores_a_constant_shift():
    """Softmax is shift-invariant, so adding a constant must change nothing."""
    p, q = logits(1.0, 2.0, 3.0), logits(0.0, 1.0, 5.0)
    assert kl_divergence(p, q) == pytest.approx(kl_divergence(p + 7.0, q + 7.0))


def test_kl_is_not_symmetric():
    p, q = logits(5.0, 0.0, 0.0), logits(0.0, 1.0, 2.0)
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))


def test_kl_rejects_mismatched_vocabularies():
    with pytest.raises(ValueError, match="shapes differ"):
        kl_divergence(logits(1.0, 2.0), logits(1.0, 2.0, 3.0))


# --- top_k_overlap ---------------------------------------------------------


def test_identical_logits_overlap_completely():
    p = logits(5.0, 4.0, 3.0, 2.0, 1.0)
    assert top_k_overlap(p, p, k=3) == 1.0


def test_reversed_rankings_do_not_overlap():
    assert top_k_overlap(logits(5.0, 4.0, 1.0, 0.0),
                         logits(0.0, 1.0, 4.0, 5.0), k=2) == 0.0


def test_partial_overlap_is_a_fraction():
    assert top_k_overlap(logits(5.0, 4.0, 0.0, 0.0),
                         logits(5.0, 0.0, 4.0, 0.0), k=2) == 0.5


def test_k_larger_than_the_vocabulary_is_clamped():
    p = logits(3.0, 2.0, 1.0)
    assert top_k_overlap(p, p, k=99) == 1.0


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        top_k_overlap(logits(1.0, 2.0), logits(1.0, 2.0), k=0)


# --- Decode ----------------------------------------------------------------


def test_an_empty_decode_is_rejected():
    with pytest.raises(ValueError, match="no tokens"):
        Decode(tokens=())


# --- Trial -----------------------------------------------------------------


def test_references_diverge_only_when_the_edit_changed_the_answer():
    assert trial(decode(34), decode(0), decode(34)).references_diverge
    assert not trial(decode(34), decode(34), decode(34)).references_diverge


def test_leyline_case_tracks_full():
    """Their worked example: full says 34, reprefill says 0, splice says 34."""
    t = trial(decode(34, 1, 2), decode(0, 9, 9), decode(34, 1, 2))
    assert t.references_diverge
    assert t.verdict == FULL


def test_scm_case_tracks_reprefill():
    """SCM recomputes under the edited context, so it lands on the other side."""
    t = trial(decode(34, 1, 2), decode(0, 9, 9), decode(0, 9, 9))
    assert t.verdict == REPREFILL


def test_matching_both_references_means_they_never_separated():
    t = trial(decode(7), decode(7), decode(7))
    assert t.verdict == BOTH
    assert not t.references_diverge


def test_matching_no_reference_is_its_own_verdict():
    assert trial(decode(1), decode(2), decode(3)).verdict == NEITHER


def test_prefix_lengths_measure_against_each_reference():
    t = trial(decode(1, 2, 3, 4), decode(1, 9, 9, 9), decode(1, 2, 9, 9))
    assert t.prefix_lengths() == {FULL: 2, REPREFILL: 1}


def test_divergences_report_kl_against_each_reference():
    t = trial(
        decode(1, first=logits(5.0, 0.0, 0.0)),
        decode(2, first=logits(0.0, 5.0, 0.0)),
        decode(1, first=logits(5.0, 0.0, 0.0)),
    )
    kls = t.divergences()
    assert kls[FULL] == pytest.approx(0.0, abs=1e-6)
    assert kls[REPREFILL] > 1.0


def test_divergences_need_logits_on_every_path():
    t = trial(decode(1), decode(2), decode(1))
    with pytest.raises(ValueError, match="no logits"):
        t.divergences()

    t = trial(decode(1), decode(2), decode(1, first=logits(1.0, 2.0)))
    with pytest.raises(ValueError, match="full reference has no logits"):
        t.divergences()


# --- summarize -------------------------------------------------------------


def test_summarize_counts_only_informative_trials():
    """Trials where the references agree carry no signal and must not dilute."""
    informative = trial(decode(34), decode(0), decode(34))
    flat = trial(decode(7), decode(7), decode(7))
    report = summarize([informative, flat, flat])

    assert report.trials == 3
    assert report.informative == 1
    assert report.verdicts == {FULL: 1}


def test_summarize_reproduces_the_leyline_pattern():
    """14/17 tracking full and 0 tracking reprefill is their Moonlight result."""
    trials = [trial(decode(34), decode(0), decode(34)) for _ in range(14)]
    trials += [trial(decode(34), decode(0), decode(99)) for _ in range(3)]
    report = summarize(trials)

    assert report.informative == 17
    assert report.verdicts == {FULL: 14, NEITHER: 3}
    assert report.tracks_reprefill == 0.0


def test_summarize_reports_the_result_scm_needs():
    trials = [trial(decode(34), decode(0), decode(0)) for _ in range(9)]
    trials += [trial(decode(34), decode(0), decode(34))]
    report = summarize(trials)

    assert report.verdicts == {REPREFILL: 9, FULL: 1}
    assert report.tracks_reprefill == pytest.approx(0.9)


def test_summarize_averages_prefix_lengths_over_informative_trials():
    trials = [
        trial(decode(1, 2, 3), decode(9, 9, 9), decode(1, 2, 9)),
        trial(decode(1, 2, 3), decode(9, 9, 9), decode(1, 9, 9)),
    ]
    report = summarize(trials)
    assert report.mean_prefix[FULL] == pytest.approx(1.5)
    assert report.mean_prefix[REPREFILL] == 0.0


def test_summarize_of_nothing_does_not_divide_by_zero():
    report = summarize([])
    assert report.trials == 0
    assert report.informative == 0
    assert report.verdicts == {}
    assert report.tracks_reprefill == 0.0
    assert report.mean_prefix == {FULL: 0.0, REPREFILL: 0.0}


def test_summarize_with_no_informative_trials_is_not_a_division_by_zero():
    report = summarize([trial(decode(7), decode(7), decode(7))])
    assert report.trials == 1
    assert report.informative == 0
    assert report.tracks_reprefill == 0.0
