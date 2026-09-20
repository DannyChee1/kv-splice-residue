"""Replay an OpenHands event stream into the prompts the model saw.

A Condensation never records why it fired, so we infer it: a pending
CondensationRequest means somebody asked, anything else means a threshold.
`to_message` is a parameter because it is guesswork until checked against a
real dump.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from scm.traces import (
    ChangeKind,
    ContextChange,
    Decider,
    Message,
    Trace,
    Trigger,
    Turn,
)

EventToMessage = Callable[[dict], Message | None]

CONDENSATION = "Condensation"
CONDENSATION_REQUEST = "CondensationRequest"

ROLE_BY_KIND = {
    "SystemPromptEvent": "system",
    "ActionEvent": "assistant",
    "ObservationEvent": "tool",
    "AgentErrorEvent": "tool",
    "CondensationSummaryEvent": "user",
}


class CollectError(ValueError):
    pass


def _text_of(event: dict) -> str:
    """Dig the text out of an event, trying the shapes the SDK uses."""
    for key in ("summary", "text", "content", "message"):
        value = event.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            inner = value.get("content", value.get("text"))
            if isinstance(inner, str):
                return inner
            if isinstance(inner, list):
                return "".join(
                    part.get("text", "") for part in inner if isinstance(part, dict)
                )
    return ""


def default_to_message(event: dict) -> Message | None:
    """Best-effort event to Message. None for events the LLM never sees."""
    kind = event.get("kind", "")
    if kind in (CONDENSATION, CONDENSATION_REQUEST):
        return None
    role = ROLE_BY_KIND.get(kind)
    if role is None:
        if kind != "MessageEvent":
            return None
        role = "user" if event.get("source") == "user" else "assistant"
    return Message(role=role, content=_text_of(event), extra={"id": event["id"]})


def _span_of(view_ids: Sequence[str], forgotten: set[str]) -> tuple[int, int]:
    """Where the forgotten events sit in the view, as one half-open range."""
    positions = sorted(i for i, eid in enumerate(view_ids) if eid in forgotten)
    if not positions:
        raise CollectError("condensation forgets nothing that is in the view")
    start, end = positions[0], positions[-1] + 1
    if end - start != len(positions):
        raise CollectError(
            f"condensation forgets a broken-up range {start}..{end}"
        )
    return start, end


def _apply(
    event: dict,
    view: list[dict],
    to_message: EventToMessage,
    requested: bool,
    max_size: int | None,
) -> ContextChange:
    """Drop the forgotten events and slot the summary in."""
    span = _span_of([e["id"] for e in view], set(event.get("forgotten_event_ids", [])))
    before = len(view)
    del view[span[0] : span[1]]

    inserted: tuple[Message, ...] = ()
    offset = event.get("summary_offset")
    if event.get("summary") is not None and offset is not None:
        if not 0 <= offset <= len(view):
            raise CollectError(
                f"summary_offset {offset} is outside a view of {len(view)}"
            )
        summary = {
            "kind": "CondensationSummaryEvent",
            "id": f"{event['id']}-summary",
            "summary": event["summary"],
            "source": "environment",
        }
        view.insert(offset, summary)
        message = to_message(summary)
        inserted = (message,) if message else ()

    return ContextChange(
        kind=ChangeKind.COMPACT,
        decider=Decider.MODEL if requested else Decider.HARNESS,
        removed_spans=(span,),
        inserted=inserted,
        view_size=before,
        trigger=None if requested else Trigger("view_events", before, max_size),
        raw={"condensation_id": event["id"]},
    )


def build_trace(
    events: Iterable[dict],
    *,
    program_id: str,
    model: str,
    harness: str = "openhands",
    to_message: EventToMessage = default_to_message,
    max_size: int | None = 240,
) -> Trace:
    """A turn ends when llm_response_id changes, so parallel tool calls stay together."""
    view: list[dict] = []
    turns: list[Turn] = []
    pending: ContextChange | None = None
    requested = False
    response_id: str | None = None

    for event in events:
        if "id" not in event:
            raise CollectError(f"event has no id: {sorted(event)}")
        kind = event.get("kind", "")

        if kind == CONDENSATION_REQUEST:
            requested = True
            continue

        if kind == CONDENSATION:
            pending = _apply(event, view, to_message, requested, max_size)
            requested = False
            continue

        if event.get("source") == "agent":
            current = event.get("llm_response_id")
            if current is None or current != response_id:
                messages = _messages_of(view, to_message)
                if messages:
                    turns.append(
                        Turn(
                            index=len(turns),
                            messages=messages,
                            output=_text_of(event),
                            change_before=pending if turns else None,
                            t_request=_seconds(event.get("timestamp")),
                        )
                    )
                    pending = None
                response_id = current

        view.append(event)

    if not turns:
        raise CollectError(f"{program_id}: no agent steps in {len(view)} events")

    return Trace(
        program_id=program_id, harness=harness, model=model, turns=tuple(turns)
    )


def _messages_of(
    view: Sequence[dict], to_message: EventToMessage
) -> tuple[Message, ...]:
    return tuple(m for m in (to_message(e) for e in view) if m is not None)


def _seconds(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(timestamp).timestamp()
    except ValueError:
        return None
