import json

import pytest

from scm.traces import (
    ChangeKind,
    ContextChange,
    Decider,
    Message,
    SchemaError,
    Trace,
    Trigger,
    Turn,
    Usage,
    iter_traces,
    read_trace,
    write_trace,
)


def msgs(*roles: str) -> tuple[Message, ...]:
    return tuple(Message(role=r, content=f"{r} content") for r in roles)


def compaction_trace() -> Trace:
    """Three turns; the last one compacts messages 1-3 into a summary."""
    turn0 = Turn(index=0, messages=msgs("system", "user"), output="ls")
    turn1 = Turn(
        index=1,
        messages=msgs("system", "user", "assistant", "tool"),
        output="pytest",
    )
    compact = ContextChange(
        kind=ChangeKind.COMPACT,
        decider=Decider.HARNESS,
        removed_spans=((1, 4),),
        inserted=(Message(role="user", content="summary of prior work"),),
        view_size=4,
        trigger=Trigger(metric="prompt_tokens", value=118_000, threshold=100_000),
    )
    turn2 = Turn(
        index=2,
        messages=msgs("system", "user"),
        output="done",
        change_before=compact,
        usage=Usage(prompt_tokens=900, cached_prompt_tokens=512),
    )
    return Trace(
        program_id="astropy__astropy-12907",
        harness="openhands",
        model="gpt-5",
        turns=(turn0, turn1, turn2),
    )


def test_roundtrip_preserves_trace():
    trace = compaction_trace()
    assert Trace.from_dict(json.loads(json.dumps(trace.to_dict()))) == trace


def test_roundtrip_via_disk(tmp_path):
    trace = compaction_trace()
    path = write_trace(tmp_path / "t.json", trace)
    assert read_trace(path) == trace


def test_iter_traces_reads_directory_in_order(tmp_path):
    for pid in ("b", "a"):
        write_trace(tmp_path / f"{pid}.json", Trace(
            program_id=pid, harness="h", model="m",
            turns=(Turn(index=0, messages=msgs("user")),),
        ))
    assert [t.program_id for t in iter_traces(tmp_path)] == ["a", "b"]


def test_optional_fields_are_omitted_not_null():
    turn = Turn(index=0, messages=msgs("user"))
    d = turn.to_dict()
    assert "change_before" not in d and "usage" not in d and "t_request" not in d


def test_changes_lists_turn_index_and_change():
    trace = compaction_trace()
    (index, change), = trace.changes
    assert index == 2
    assert change.kind is ChangeKind.COMPACT
    assert change.removed_count == 3


def test_trigger_reports_threshold_crossing():
    assert Trigger("prompt_tokens", 118_000, 100_000).fired_on_threshold
    assert not Trigger("prompt_tokens", 90_000, 100_000).fired_on_threshold
    assert not Trigger("prompt_tokens", 118_000).fired_on_threshold


def test_rejects_noncontiguous_turn_indices():
    with pytest.raises(SchemaError, match="expected turn"):
        Trace(
            program_id="p", harness="h", model="m",
            turns=(
                Turn(index=0, messages=msgs("user")),
                Turn(index=2, messages=msgs("user")),
            ),
        )


def test_rejects_change_on_first_turn():
    change = ContextChange(kind=ChangeKind.CLEAR, decider=Decider.HARNESS)
    with pytest.raises(SchemaError, match="turn 0"):
        Trace(
            program_id="p", harness="h", model="m",
            turns=(Turn(index=0, messages=msgs("user"), change_before=change),),
        )


def test_rejects_span_beyond_the_recorded_view():
    with pytest.raises(SchemaError, match="span ends at 5, view held 4"):
        ContextChange(kind=ChangeKind.CLEAR, decider=Decider.HARNESS,
                      removed_spans=((0, 5),), view_size=4)


def test_span_filling_the_whole_view_is_allowed():
    change = ContextChange(kind=ChangeKind.CLEAR, decider=Decider.HARNESS,
                           removed_spans=((0, 4),), view_size=4)
    assert change.removed_count == 4


def test_span_is_unchecked_when_no_view_size_was_recorded():
    change = ContextChange(kind=ChangeKind.CLEAR, decider=Decider.HARNESS,
                           removed_spans=((0, 999),))
    assert change.removed_count == 999


def test_a_change_may_span_more_than_the_previous_prompt():
    """The view grows past the last prompt before an edit lands."""
    change = ContextChange(kind=ChangeKind.CLEAR, decider=Decider.HARNESS,
                           removed_spans=((3, 4),), view_size=5)
    trace = Trace(
        program_id="p", harness="h", model="m",
        turns=(
            Turn(index=0, messages=msgs("system", "user")),
            Turn(index=1, messages=msgs("user"), change_before=change),
        ),
    )
    assert trace.changes[0][1].view_size == 5


@pytest.mark.parametrize("spans", [((3, 1),), ((-1, 2),)])
def test_rejects_malformed_spans(spans):
    with pytest.raises(SchemaError, match="bad span"):
        ContextChange(kind=ChangeKind.CLEAR, decider=Decider.HARNESS,
                      removed_spans=spans)


def test_rejects_overlapping_spans():
    with pytest.raises(SchemaError, match="overlap"):
        ContextChange(kind=ChangeKind.CLEAR, decider=Decider.HARNESS,
                      removed_spans=((0, 3), (2, 5)))


def test_accepts_adjacent_spans():
    change = ContextChange(kind=ChangeKind.CLEAR, decider=Decider.HARNESS,
                           removed_spans=((0, 2), (2, 4)))
    assert change.removed_count == 4


def test_rejects_empty_trace_and_empty_turn():
    with pytest.raises(SchemaError, match="no turns"):
        Trace(program_id="p", harness="h", model="m", turns=())
    with pytest.raises(SchemaError, match="no messages"):
        Turn(index=0, messages=())


def test_rejects_empty_role_and_program_id():
    with pytest.raises(SchemaError, match="role"):
        Message(role="", content="x")
    with pytest.raises(SchemaError, match="program_id"):
        Trace(program_id="", harness="h", model="m",
              turns=(Turn(index=0, messages=msgs("user")),))


def test_unknown_change_kind_is_rejected():
    with pytest.raises(ValueError):
        ContextChange.from_dict({"kind": "teleport", "decider": "harness"})


def test_extra_and_raw_survive_roundtrip():
    change = ContextChange(
        kind=ChangeKind.OTHER, decider=Decider.UNKNOWN,
        raw={"condenser": "LLMSummarizingCondenser"},
    )
    turn1 = Turn(
        index=1,
        messages=(Message("tool", "out", extra={"tool_call_id": "c1"}),),
        change_before=change,
    )
    trace = Trace(program_id="p", harness="h", model="m",
                  turns=(Turn(index=0, messages=msgs("user")), turn1))
    back = Trace.from_dict(trace.to_dict())
    assert back.turns[1].messages[0].extra == {"tool_call_id": "c1"}
    assert back.turns[1].change_before.raw["condenser"] == "LLMSummarizingCondenser"
