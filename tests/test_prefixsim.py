import pytest

from scm.prefixsim import (
    PrefixCache,
    message_spans,
    simple_render,
    simulate,
    summarize,
)
from scm.traces import ChangeKind, ContextChange, Decider, Message, Trace, Turn


class WordTokenizer:
    """One token per whitespace-separated word; enough to count with."""

    def encode(self, text: str) -> list[int]:
        return [hash(w) for w in text.split()]


TOK = WordTokenizer()


def m(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def turn(index: int, messages, change=None) -> Turn:
    return Turn(index=index, messages=tuple(messages), change_before=change)


def trace(*turns: Turn) -> Trace:
    return Trace(program_id="p", harness="h", model="m", turns=turns)


def run(t: Trace, block_size: int = 1):
    return simulate(t, simple_render, TOK, block_size=block_size)


# --- message_spans ---------------------------------------------------------


def test_spans_tile_the_prompt_without_gaps():
    messages = [m("system", "a b"), m("user", "c d e"), m("tool", "f")]
    spans = message_spans(messages, simple_render, TOK)
    assert spans[0][0] == 0
    assert [s[1] for s in spans[:-1]] == [s[0] for s in spans[1:]]
    assert spans[-1][1] == len(TOK.encode(simple_render(messages)))


def test_spans_measure_each_message():
    """The role marker fuses with the first word here, and the spans still fit."""
    messages = [m("system", "a b"), m("user", "c d e")]
    spans = message_spans(messages, simple_render, TOK)
    assert [end - start for start, end in spans] == [2, 3]


def test_spans_of_empty_message_list():
    assert message_spans([], simple_render, TOK) == []


def test_a_renderer_that_shrinks_the_prompt_is_an_error():
    def shrinking_render(messages):
        return " ".join("word" for _ in range(10 - len(messages)))

    with pytest.raises(ValueError, match="shrank the prompt"):
        message_spans([m("user", "a"), m("user", "b")], shrinking_render, TOK)


# --- PrefixCache -----------------------------------------------------------


def test_cache_misses_everything_when_empty():
    assert PrefixCache().longest_hit([1, 2, 3]) == 0


def test_cache_matches_longest_shared_prefix():
    cache = PrefixCache()
    cache.add([1, 2, 3])
    cache.add([1, 2, 9, 9, 9])
    assert cache.longest_hit([1, 2, 3, 4]) == 3
    assert cache.longest_hit([1, 2, 9, 9]) == 4
    assert cache.longest_hit([7, 7]) == 0


def test_cache_hit_cannot_exceed_the_new_prompt():
    cache = PrefixCache()
    cache.add([1, 2, 3, 4, 5])
    assert cache.longest_hit([1, 2]) == 2


def test_block_size_rounds_a_hit_down():
    cache = PrefixCache(block_size=16)
    cache.add(list(range(40)))
    assert cache.longest_hit(list(range(35))) == 32
    assert cache.longest_hit(list(range(10))) == 0


def test_block_size_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        PrefixCache(block_size=0)


# --- simulate --------------------------------------------------------------


def test_append_only_growth_is_all_cached_or_novel():
    """The classic chatbot shape: nothing is ever repeated."""
    first = [m("system", "s"), m("user", "u")]
    stats = run(trace(
        turn(0, first),
        turn(1, first + [m("assistant", "a"), m("tool", "t")]),
    ))
    assert stats.reused_tokens == 0
    assert stats.turns[1].cached_tokens == stats.turns[0].prompt_tokens
    assert stats.reprefill_share == 0.0


def test_deleting_a_middle_message_forces_a_repeat():
    """Dropping a stale tool result breaks the prefix, so the tail repeats."""
    system, user, tool, tail = (
        m("system", "s"), m("user", "u"), m("tool", "big output"), m("assistant", "a"),
    )
    change = ContextChange(
        kind=ChangeKind.CLEAR,
        decider=Decider.HARNESS,
        removed_spans=((2, 3),),
    )
    stats = run(trace(
        turn(0, [system, user, tool, tail]),
        turn(1, [system, user, tail], change=change),
    ))
    after = stats.turns[1]
    assert after.cached_tokens > 0            # system + user still match
    assert after.reused_tokens > 0            # the tail has to be redone
    assert after.novel_tokens == 0            # nothing in turn 1 is new
    assert stats.reused_by_decider() == {"harness": after.reused_tokens}


def test_novel_content_is_not_counted_as_repeat():
    system = m("system", "s")
    change = ContextChange(kind=ChangeKind.CLEAR, decider=Decider.HARNESS,
                           removed_spans=((1, 2),))
    stats = run(trace(
        turn(0, [system, m("tool", "stale"), m("assistant", "keep")]),
        turn(1, [system, m("assistant", "keep"), m("tool", "brand new")], change=change),
    ))
    after = stats.turns[1]
    assert after.reused_tokens > 0
    assert after.novel_tokens > 0


def test_first_turn_is_entirely_novel():
    stats = run(trace(turn(0, [m("system", "s"), m("user", "u")])))
    only = stats.turns[0]
    assert only.cached_tokens == 0
    assert only.reused_tokens == 0
    assert only.novel_tokens == only.prompt_tokens


def test_token_accounting_balances_every_turn():
    change = ContextChange(kind=ChangeKind.COMPACT, decider=Decider.HARNESS,
                           removed_spans=((1, 3),))
    stats = run(trace(
        turn(0, [m("system", "s"), m("user", "u"), m("tool", "x y z")]),
        turn(1, [m("system", "s"), m("user", "summary"), m("tool", "x y z")],
             change=change),
    ))
    for t in stats.turns:
        assert t.cached_tokens + t.computed_tokens == t.prompt_tokens
    assert stats.cached_tokens + stats.computed_tokens == stats.prompt_tokens


def test_larger_blocks_never_help():
    change = ContextChange(kind=ChangeKind.CLEAR, decider=Decider.HARNESS,
                           removed_spans=((1, 2),))
    messages = [m("system", " ".join(f"w{i}" for i in range(50)))]
    t = trace(
        turn(0, messages + [m("tool", "stale"), m("assistant", "keep")]),
        turn(1, messages + [m("assistant", "keep")], change=change),
    )
    assert run(t, block_size=16).cached_tokens <= run(t, block_size=1).cached_tokens


def test_model_decided_change_is_attributed_separately():
    system = m("system", "s")
    tail = m("assistant", "a")
    stats = run(trace(
        turn(0, [system, m("tool", "x"), tail]),
        turn(1, [system, tail], change=ContextChange(
            kind=ChangeKind.COMPACT, decider=Decider.MODEL, removed_spans=((1, 2),))),
    ))
    assert set(stats.reused_by_decider()) == {"model"}


def test_repeat_without_a_recorded_change_is_attributed_to_none():
    system, tail = m("system", "s"), m("assistant", "a")
    stats = run(trace(
        turn(0, [system, m("tool", "x"), tail]),
        turn(1, [system, tail]),
    ))
    assert set(stats.reused_by_decider()) == {"none"}


# --- summarize -------------------------------------------------------------


def test_summarize_adds_traces_up():
    change = ContextChange(kind=ChangeKind.CLEAR, decider=Decider.HARNESS,
                           removed_spans=((1, 2),))
    system, tail = m("system", "s"), m("assistant", "a")
    one = run(trace(
        turn(0, [system, m("tool", "x"), tail]),
        turn(1, [system, tail], change=change),
    ))
    both = summarize([one, one])

    assert both["traces"] == 2
    assert both["reused_tokens"] == 2 * one.reused_tokens
    assert both["harness_attributable"] == 2 * one.reused_tokens
    assert both["novel_tokens"] == both["computed_tokens"] - both["reused_tokens"]
    assert both["reprefill_share"] == pytest.approx(one.reprefill_share)


def test_summarize_of_nothing_is_zero_not_an_error():
    assert summarize([]) == {
        "traces": 0, "prompt_tokens": 0, "computed_tokens": 0, "reused_tokens": 0,
        "novel_tokens": 0, "reprefill_share": 0.0, "harness_attributable": 0,
        "reused_by_decider": {},
    }
