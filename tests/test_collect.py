import pytest

from scm.collect import (
    CollectError,
    _span_of,
    build_trace,
    default_to_message,
)
from scm.traces import ChangeKind, Decider

COUNTER = iter(range(1, 10_000))


def ev(kind: str, **fields) -> dict:
    return {"kind": kind, "id": fields.pop("id", f"e{next(COUNTER)}"), **fields}


def system(text="you are an agent"):
    return ev("SystemPromptEvent", text=text, source="environment")


def user(text="fix the bug"):
    return ev("MessageEvent", text=text, source="user")


def action(text="run tests", response_id=None):
    return ev("ActionEvent", text=text, source="agent",
              llm_response_id=response_id or f"r{next(COUNTER)}")


def observation(text="3 tests failed"):
    return ev("ObservationEvent", text=text, source="environment")


def condensation(forgotten, summary=None, offset=None, cid="c1"):
    e = ev("Condensation", id=cid, source="environment",
           forgotten_event_ids=list(forgotten))
    if summary is not None:
        e["summary"] = summary
        e["summary_offset"] = offset
    return e


def build(events, **kw):
    return build_trace(events, program_id="p", model="m", **kw)


# --- default_to_message ----------------------------------------------------


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("SystemPromptEvent", "system"),
        ("ActionEvent", "assistant"),
        ("ObservationEvent", "tool"),
        ("AgentErrorEvent", "tool"),
        ("CondensationSummaryEvent", "user"),
    ],
)
def test_each_event_kind_maps_to_a_role(kind, expected):
    assert default_to_message(ev(kind, text="x")).role == expected


def test_message_event_role_follows_its_source():
    assert default_to_message(ev("MessageEvent", text="x", source="user")).role == "user"
    assert default_to_message(
        ev("MessageEvent", text="x", source="agent")
    ).role == "assistant"


def test_bookkeeping_events_are_not_shown_to_the_model():
    assert default_to_message(ev("Condensation", forgotten_event_ids=[])) is None
    assert default_to_message(ev("CondensationRequest", source="user")) is None
    assert default_to_message(ev("SomethingNew", text="x")) is None


def test_message_keeps_its_event_id():
    assert default_to_message(ev("ActionEvent", id="e9", text="x")).extra == {"id": "e9"}


def test_text_is_found_in_nested_content():
    nested = ev("MessageEvent", source="user",
                content={"content": [{"text": "a"}, {"text": "b"}]})
    assert default_to_message(nested).content == "ab"


def test_missing_text_becomes_empty_not_an_error():
    assert default_to_message(ev("ActionEvent")).content == ""


# --- _span_of --------------------------------------------------------------


def test_span_covers_the_forgotten_run():
    assert _span_of(["a", "b", "c", "d"], {"b", "c"}) == (1, 3)


def test_span_rejects_a_broken_up_range():
    with pytest.raises(CollectError, match="broken-up range"):
        _span_of(["a", "b", "c", "d"], {"a", "c"})


def test_span_rejects_ids_that_are_not_in_the_view():
    with pytest.raises(CollectError, match="forgets nothing"):
        _span_of(["a", "b"], {"zzz"})


# --- build_trace -----------------------------------------------------------


def test_each_agent_step_becomes_a_turn():
    trace = build([system(), user(), action(), observation(), action()])
    assert len(trace.turns) == 2
    assert [m.role for m in trace.turns[0].messages] == ["system", "user"]
    assert [m.role for m in trace.turns[1].messages] == [
        "system", "user", "assistant", "tool",
    ]


def test_parallel_tool_calls_share_one_turn():
    """Two actions from one LLM response are one step, not two."""
    trace = build([
        system(), user(),
        action(response_id="r1"), action(response_id="r1"),
        observation(), action(response_id="r2"),
    ])
    assert len(trace.turns) == 2


def test_condensation_records_the_span_and_the_summary():
    sys_e, usr_e = system(), user()
    stale, keep = observation("stale"), observation("keep")
    trace = build([
        sys_e, usr_e, action(), stale, keep, action(),
        condensation([stale["id"]], summary="we looked at the bug", offset=2),
        action(),
    ])
    change = trace.turns[-1].change_before
    assert change.kind is ChangeKind.COMPACT
    assert change.decider is Decider.HARNESS
    assert change.removed_spans == ((3, 4),)
    assert change.inserted[0].content == "we looked at the bug"
    assert change.raw == {"condensation_id": "c1"}


def test_condensed_view_drops_and_inserts_in_the_right_places():
    sys_e, usr_e = system(), user()
    a1, stale, keep = action(), observation("stale"), observation("keep")
    trace = build([
        sys_e, usr_e, a1, stale, keep,
        condensation([stale["id"]], summary="SUMMARY", offset=3),
        action(),
    ])
    final = trace.turns[-1].messages
    assert [m.content for m in final] == [
        "you are an agent", "fix the bug", "run tests", "SUMMARY", "keep",
    ]


def test_threshold_condensation_carries_the_view_size():
    stale = observation("stale")
    trace = build(
        [system(), user(), action(), stale, action(),
         condensation([stale["id"]]), action()],
        max_size=240,
    )
    trigger = trace.turns[-1].change_before.trigger
    assert trigger.metric == "view_events"
    assert trigger.value == 5
    assert trigger.threshold == 240


def test_a_requested_condensation_is_attributed_to_the_model():
    stale = observation("stale")
    trace = build([
        system(), user(), action(), stale, action(),
        ev("CondensationRequest", source="user"),
        condensation([stale["id"]]),
        action(),
    ])
    change = trace.turns[-1].change_before
    assert change.decider is Decider.MODEL
    assert change.trigger is None


def test_a_request_only_covers_the_next_condensation():
    first, second = observation("one"), observation("two")
    trace = build([
        system(), user(), action(), first, second, action(),
        ev("CondensationRequest", source="user"),
        condensation([first["id"]], cid="c1"),
        action(),
        condensation([second["id"]], cid="c2"),
        action(),
    ])
    assert trace.turns[2].change_before.decider is Decider.MODEL
    assert trace.turns[3].change_before.decider is Decider.HARNESS


def test_two_condensations_between_steps_keep_only_the_later_one():
    """The schema hangs one change on a turn; the earlier is folded into the view."""
    one, two = observation("one"), observation("two")
    trace = build([
        system(), user(), action(), one, two, action(),
        condensation([one["id"]], cid="c1"),
        condensation([two["id"]], cid="c2"),
        action(),
    ])
    assert trace.turns[-1].change_before.raw == {"condensation_id": "c2"}
    assert [m.content for m in trace.turns[-1].messages] == [
        "you are an agent", "fix the bug", "run tests", "run tests",
    ]


def test_first_turn_never_carries_a_change():
    trace = build([system(), user(), action()])
    assert trace.turns[0].change_before is None


def test_timestamp_becomes_epoch_seconds():
    step = action()
    step["timestamp"] = "2026-09-19T12:00:00"
    assert build([system(), user(), step]).turns[0].t_request is not None


def test_unparseable_timestamp_is_dropped_not_fatal():
    step = action()
    step["timestamp"] = "last tuesday"
    assert build([system(), user(), step]).turns[0].t_request is None


def test_summary_offset_past_the_end_of_the_view_is_rejected():
    stale = observation("stale")
    with pytest.raises(CollectError, match="outside a view"):
        build([system(), user(), action(), stale, action(),
               condensation([stale["id"]], summary="s", offset=99), action()])


def test_event_without_an_id_is_rejected():
    with pytest.raises(CollectError, match="no id"):
        build([{"kind": "ActionEvent", "source": "agent"}])


def test_stream_with_no_agent_step_is_rejected():
    with pytest.raises(CollectError, match="no agent steps"):
        build([system(), user()])


def test_summary_without_an_offset_is_not_inserted():
    stale = observation("stale")
    trace = build([system(), user(), action(), stale, action(),
                   condensation([stale["id"]], summary="ignored"), action()])
    assert trace.turns[-1].change_before.inserted == ()
    assert "ignored" not in [m.content for m in trace.turns[-1].messages]


def test_a_custom_to_message_replaces_the_default():
    from scm.traces import Message

    trace = build(
        [system(), user(), action()],
        to_message=lambda e: Message(role="user", content=e["kind"]),
    )
    assert [m.content for m in trace.turns[0].messages] == [
        "SystemPromptEvent", "MessageEvent",
    ]


def test_a_condensation_before_the_first_step_is_folded_into_turn_0():
    """Turn 0's prompt already reflects the edit, so there is no before to record."""
    sys_e, usr_e = system(), user()
    trace = build([sys_e, usr_e, condensation([usr_e["id"]]), action()])
    assert trace.turns[0].change_before is None
    assert [m.content for m in trace.turns[0].messages] == ["you are an agent"]


def test_a_change_attaches_to_one_turn_only():
    stale = observation("stale")
    trace = build([
        system(), user(), action(), stale, action(),
        condensation([stale["id"]]),
        action(), observation("fresh"), action(),
    ])
    assert trace.turns[2].change_before is not None
    assert trace.turns[3].change_before is None
