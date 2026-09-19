import pytest

from scm.probes import (
    ERASED,
    RETAINED,
    SUITE,
    Probe,
    ProbeError,
    by_harm,
    check_edit,
    history,
    pad,
    token_length,
    token_span,
)


class WordTokenizer:
    """One token per whitespace-separated word, with stable ids."""

    def encode(self, text: str) -> list[int]:
        return [hash(w) for w in text.split()]


class GluingTokenizer:
    """Merges across a boundary, the way real BPE can."""

    def encode(self, text: str) -> list[int]:
        return [hash(chunk) for chunk in text.replace("\n", " ").split("  ") if chunk]


class PairTokenizer:
    """Two characters per token, so a span of odd length straddles a boundary."""

    def encode(self, text: str) -> list[int]:
        return [hash(text[i : i + 2]) for i in range(0, len(text), 2)]


TOK = WordTokenizer()


def probe(**kw) -> Probe:
    base = dict(
        name="p",
        prefix="before ",
        span="secret ",
        suffix="question",
        retained_answer="yes",
        erased_answer="no",
    )
    return Probe(**{**base, **kw})


# --- shape -----------------------------------------------------------------


def test_full_and_edited_differ_by_the_span():
    p = probe()
    assert p.full == "before secret question"
    assert p.edited == "before question"


def test_a_probe_whose_answers_match_is_rejected():
    with pytest.raises(ProbeError, match="cannot tell the two apart"):
        probe(retained_answer="same", erased_answer="same")


def test_an_empty_span_is_rejected():
    with pytest.raises(ProbeError, match="nothing to delete"):
        probe(span="")


def test_an_unknown_harm_is_rejected():
    with pytest.raises(ProbeError, match="harm must be"):
        probe(harm="sideways")


# --- polarity --------------------------------------------------------------


def test_harmful_retention_names_the_retained_answer_as_bad():
    p = probe(harm=RETAINED, retained_answer="leaks", erased_answer="refuses")
    assert p.bad_answer == "leaks"
    assert p.good_answer == "refuses"


def test_harmful_erasure_flips_which_answer_is_bad():
    """Leyline's polarity: the span was needed, so losing it is the failure."""
    p = probe(harm=ERASED, retained_answer="34", erased_answer="0")
    assert p.bad_answer == "0"
    assert p.good_answer == "34"


def test_by_harm_splits_the_suite():
    harmful = by_harm(SUITE, RETAINED)
    needed = by_harm(SUITE, ERASED)
    assert len(harmful) + len(needed) == len(SUITE)
    assert {p.name for p in needed} == {"needed-calculation"}


def test_the_suite_carries_both_polarities():
    """A suite that only flatters SCM would not convince a reviewer."""
    assert by_harm(SUITE, RETAINED)
    assert by_harm(SUITE, ERASED)


@pytest.mark.parametrize("p", SUITE, ids=lambda p: p.name)
def test_every_suite_probe_cuts_cleanly(p):
    check_edit(p, TOK)


# --- token_span ------------------------------------------------------------


def test_span_covers_exactly_the_span_tokens():
    p = probe(prefix="a b ", span="c d ", suffix="e")
    assert token_span(p, TOK) == (2, 4)


def test_cutting_the_span_reproduces_the_edited_prompt():
    check_edit(probe(prefix="a b ", span="c d ", suffix="e"), TOK)


def test_a_span_that_tokenizes_to_nothing_is_rejected():
    with pytest.raises(ProbeError, match="tokenizes to nothing"):
        token_span(probe(prefix="a ", span="   ", suffix="b"), TOK)


def test_a_span_starting_mid_token_is_rejected():
    """Splicing mid-token would make every downstream number meaningless."""
    p = probe(prefix="a", span="b", suffix="c")
    with pytest.raises(ProbeError, match="span start does not"):
        token_span(p, GluingTokenizer())


def test_a_span_ending_mid_token_is_rejected():
    """Start lines up, end does not: the span's last token swallows the suffix."""
    p = probe(prefix="ab", span="c", suffix="d")
    with pytest.raises(ProbeError, match="span end does not"):
        token_span(p, PairTokenizer())


def test_check_edit_catches_a_cut_that_does_not_match():
    class DropsOnEdit:
        """Tokenizes the edited prompt shorter than cutting the span would give."""

        def encode(self, text: str) -> list[int]:
            words = text.split()
            if "secret" not in words:
                words = words[:-1]
            return [hash(w) for w in words]

    with pytest.raises(ProbeError, match="cutting the span leaves"):
        check_edit(probe(), DropsOnEdit())


# --- the suite itself ------------------------------------------------------


def test_suite_names_are_unique():
    names = [p.name for p in SUITE]
    assert len(set(names)) == len(names)


def test_suite_covers_the_cases_scm_is_argued_on():
    assert {p.name for p in SUITE} >= {
        "retracted-price",
        "injected-instruction",
        "leaked-credential",
        "superseded-tool-error",
    }


# --- padding ---------------------------------------------------------------


def test_history_is_deterministic_for_a_seed():
    assert history(3, seed=7) == history(3, seed=7)
    assert history(3, seed=7) != history(3, seed=8)


def test_history_grows_with_steps():
    assert len(history(6)) > len(history(3)) > len(history(0))


def test_no_history_is_empty():
    assert history(0) == ""


def test_negative_steps_are_rejected():
    with pytest.raises(ProbeError, match="must not be negative"):
        history(-1)


def test_history_looks_like_a_trajectory():
    text = history(6, seed=3)
    assert text.count("$ ") == 6
    assert "pytest" in text


def test_padding_before_sinks_the_span_deeper():
    p = probe(prefix="a b ", span="c d ", suffix="e")
    assert token_span(pad(p, before=4), TOK)[0] > token_span(p, TOK)[0]


def test_padding_after_puts_tokens_downstream_of_the_cut():
    """The leftover lives after the cut, so it needs somewhere to show up."""
    p = probe(prefix="a b ", span="c d ", suffix="e")
    padded = pad(p, after=4)
    start, end = token_span(padded, TOK)
    assert token_length(padded, TOK) - end > token_length(p, TOK) - token_span(p, TOK)[1]


def test_padding_leaves_the_span_itself_alone():
    p = probe(prefix="a b ", span="c d ", suffix="e")
    padded = pad(p, before=3, after=3)
    assert padded.span == p.span
    start, end = token_span(padded, TOK)
    assert end - start == 2


def test_a_padded_probe_still_cuts_cleanly():
    check_edit(pad(probe(prefix="a b ", span="c d ", suffix="e"), 3, 3), TOK)


@pytest.mark.parametrize("p", SUITE, ids=lambda p: p.name)
def test_every_suite_probe_cuts_cleanly_once_padded(p):
    check_edit(pad(p, before=5, after=5), TOK)


def test_padding_keeps_the_answers_and_polarity():
    p = probe(harm=ERASED, retained_answer="34", erased_answer="0")
    padded = pad(p, before=2, after=2)
    assert (padded.retained_answer, padded.erased_answer) == ("34", "0")
    assert padded.bad_answer == "0"


def test_padding_records_how_much_was_added():
    assert pad(probe(), before=3, after=2).name == "p+3/2"


def test_padding_nothing_leaves_the_prompt_unchanged():
    p = probe()
    assert pad(p).full == p.full


def test_before_and_after_padding_differ():
    """Same seed on both sides would print the same trajectory twice."""
    padded = pad(probe(), before=3, after=3, seed=0)
    assert padded.prefix.startswith(history(3, seed=0))
    assert not padded.suffix.startswith(history(3, seed=0))
