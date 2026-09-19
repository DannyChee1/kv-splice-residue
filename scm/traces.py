"""Schema for agent rollout traces.

A Trace is one agent run, say a single SWE-bench instance. A Turn is one LLM
call inside it. Turns keep messages rather than token ids, because we collect
traces with one model and replay them against another.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator


class ChangeKind(StrEnum):
    COMPACT = "compact"
    CLEAR = "clear"
    STRIP = "strip"
    HANDOFF = "handoff"
    OTHER = "other"


class Decider(StrEnum):
    """Who decided the change. Harness-decided ones are the predictable ones."""

    HARNESS = "harness"
    MODEL = "model"
    UNKNOWN = "unknown"


class SchemaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.role:
            raise SchemaError("message role is empty")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        return cls(role=d["role"], content=d["content"], extra=d.get("extra", {}))


@dataclass(frozen=True, slots=True)
class Trigger:
    """Why a change fired: some metric crossed a threshold."""

    metric: str
    value: float
    threshold: float | None = None

    @property
    def fired_on_threshold(self) -> bool:
        return self.threshold is not None and self.value >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"metric": self.metric, "value": self.value}
        if self.threshold is not None:
            d["threshold"] = self.threshold
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trigger:
        return cls(
            metric=d["metric"], value=d["value"], threshold=d.get("threshold")
        )


@dataclass(frozen=True, slots=True)
class ContextChange:
    """An edit to the previous turn's messages.

    Spans are half-open [start, end) positions in that earlier list. We count
    messages rather than tokens because tokenizing happens later.
    """

    kind: ChangeKind
    decider: Decider
    removed_spans: tuple[tuple[int, int], ...] = ()
    inserted: tuple[Message, ...] = ()
    trigger: Trigger | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for start, end in self.removed_spans:
            if start < 0 or end < start:
                raise SchemaError(f"bad span ({start}, {end})")
        ordered = sorted(self.removed_spans)
        for (_, prev_end), (next_start, _) in zip(ordered, ordered[1:]):
            if next_start < prev_end:
                raise SchemaError("spans overlap")

    @property
    def removed_count(self) -> int:
        return sum(end - start for start, end in self.removed_spans)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": str(self.kind), "decider": str(self.decider)}
        if self.removed_spans:
            d["removed_spans"] = [list(s) for s in self.removed_spans]
        if self.inserted:
            d["inserted"] = [m.to_dict() for m in self.inserted]
        if self.trigger is not None:
            d["trigger"] = self.trigger.to_dict()
        if self.raw:
            d["raw"] = self.raw
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContextChange:
        trigger = d.get("trigger")
        return cls(
            kind=ChangeKind(d["kind"]),
            decider=Decider(d["decider"]),
            removed_spans=tuple(tuple(s) for s in d.get("removed_spans", ())),
            inserted=tuple(Message.from_dict(m) for m in d.get("inserted", ())),
            trigger=Trigger.from_dict(trigger) if trigger else None,
            raw=d.get("raw", {}),
        )


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts as the provider reported them. cached_prompt_tokens is what
    its prefix cache already had, so we get that signal without our own engine."""

    prompt_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        fields = ("prompt_tokens", "cached_prompt_tokens", "completion_tokens")
        return {f: getattr(self, f) for f in fields if getattr(self, f) is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Usage:
        return cls(
            prompt_tokens=d.get("prompt_tokens"),
            cached_prompt_tokens=d.get("cached_prompt_tokens"),
            completion_tokens=d.get("completion_tokens"),
        )


@dataclass(frozen=True, slots=True)
class Turn:
    index: int
    messages: tuple[Message, ...]
    output: str = ""
    change_before: ContextChange | None = None
    t_request: float | None = None
    t_response: float | None = None
    usage: Usage | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise SchemaError(f"turn index {self.index} is negative")
        if not self.messages:
            raise SchemaError(f"turn {self.index} has no messages")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "messages": [m.to_dict() for m in self.messages],
            "output": self.output,
        }
        if self.change_before is not None:
            d["change_before"] = self.change_before.to_dict()
        for key in ("t_request", "t_response"):
            if getattr(self, key) is not None:
                d[key] = getattr(self, key)
        if self.usage is not None:
            d["usage"] = self.usage.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Turn:
        change = d.get("change_before")
        usage = d.get("usage")
        return cls(
            index=d["index"],
            messages=tuple(Message.from_dict(m) for m in d["messages"]),
            output=d.get("output", ""),
            change_before=ContextChange.from_dict(change) if change else None,
            t_request=d.get("t_request"),
            t_response=d.get("t_response"),
            usage=Usage.from_dict(usage) if usage else None,
        )


@dataclass(frozen=True, slots=True)
class Trace:
    program_id: str
    harness: str
    model: str
    turns: tuple[Turn, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.program_id:
            raise SchemaError("program_id is empty")
        validate(self)

    @property
    def changes(self) -> list[tuple[int, ContextChange]]:
        return [(t.index, t.change_before) for t in self.turns if t.change_before]

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "harness": self.harness,
            "model": self.model,
            "turns": [t.to_dict() for t in self.turns],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        return cls(
            program_id=d["program_id"],
            harness=d["harness"],
            model=d["model"],
            turns=tuple(Turn.from_dict(t) for t in d["turns"]),
            meta=d.get("meta", {}),
        )


def validate(trace: Trace) -> None:
    """Checks that need to see more than one turn.

    Everything checkable from a single object lives in that object's __post_init__.
    """
    if not trace.turns:
        raise SchemaError(f"{trace.program_id}: no turns")

    for position, turn in enumerate(trace.turns):
        if turn.index != position:
            raise SchemaError(
                f"{trace.program_id}: expected turn {position}, got {turn.index}"
            )

    if trace.turns[0].change_before is not None:
        raise SchemaError(f"{trace.program_id}: turn 0 cannot have a change_before")

    for turn in trace.turns[1:]:
        change = turn.change_before
        if change is None:
            continue
        prior = len(trace.turns[turn.index - 1].messages)
        for start, end in change.removed_spans:
            if end > prior:
                raise SchemaError(
                    f"{trace.program_id} turn {turn.index}: span ends at {end}, "
                    f"previous turn has {prior} messages"
                )


def write_trace(path: Path | str, trace: Trace) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")
    return path


def read_trace(path: Path | str) -> Trace:
    return Trace.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def iter_traces(directory: Path | str) -> Iterator[Trace]:
    for path in sorted(Path(directory).glob("*.json")):
        yield read_trace(path)
