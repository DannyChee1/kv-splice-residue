"""How much prefill does an agent run repeat?

Simulates a prefix cache with no capacity limit, which is the friendliest cache
that could exist. Whatever it still has to recompute is work no cache can avoid,
so the numbers here are a floor, not an estimate. Real engines evict and share
GPUs with other requests, which only pushes the number up.

Each turn's prompt splits three ways:

    cached    the longest prefix the cache already holds
    reused    past the prefix, but this message appeared in an earlier turn
    novel     past the prefix, and genuinely new

Only `reused` is worth attacking. `novel` is a new tool result or the model's
own last reply, and nobody can have that ready ahead of time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, Sequence

from scm.traces import Message, Trace


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


Renderer = Callable[[Sequence[Message]], str]


def simple_render(messages: Sequence[Message]) -> str:
    """A stand-in template for tests. Real traces need the model's own."""
    return "".join(f"<|{m.role}|>{m.content}\n" for m in messages)


def message_spans(
    messages: Sequence[Message], render: Renderer, tokenizer: Tokenizer
) -> list[tuple[int, int]]:
    """Token range each message occupies in the rendered prompt.

    Renders growing prefixes instead of messages alone, so a template that
    merges tokens across a boundary still lands in the right place.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(len(messages)):
        end = len(tokenizer.encode(render(messages[: index + 1])))
        if end < start:
            raise ValueError(
                f"message {index} shrank the prompt from {start} to {end} tokens"
            )
        spans.append((start, end))
        start = end
    return spans


class PrefixCache:
    """Holds every prompt it has seen, and matches new ones by common prefix.

    Because any prefix of a stored sequence counts as a hit, this is the best a
    prefix cache can do. `block_size` rounds a hit down to a whole block, the
    way vLLM does at 16; leave it at 1 for the friendliest case.
    """

    def __init__(self, block_size: int = 1) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be at least 1, got {block_size}")
        self.block_size = block_size
        self._seen: list[list[int]] = []

    def longest_hit(self, tokens: Sequence[int]) -> int:
        best = 0
        for stored in self._seen:
            shared = 0
            for a, b in zip(stored, tokens):
                if a != b:
                    break
                shared += 1
            best = max(best, shared)
        return best - best % self.block_size

    def add(self, tokens: Sequence[int]) -> None:
        self._seen.append(list(tokens))


@dataclass(frozen=True, slots=True)
class TurnStats:
    index: int
    prompt_tokens: int
    cached_tokens: int
    reused_tokens: int
    novel_tokens: int
    decider: str = "none"

    @property
    def computed_tokens(self) -> int:
        return self.reused_tokens + self.novel_tokens


@dataclass(frozen=True, slots=True)
class TraceStats:
    program_id: str
    turns: tuple[TurnStats, ...] = field(default_factory=tuple)

    def _total(self, attr: str) -> int:
        return sum(getattr(t, attr) for t in self.turns)

    @property
    def prompt_tokens(self) -> int:
        return self._total("prompt_tokens")

    @property
    def cached_tokens(self) -> int:
        return self._total("cached_tokens")

    @property
    def computed_tokens(self) -> int:
        return self._total("computed_tokens")

    @property
    def reused_tokens(self) -> int:
        return self._total("reused_tokens")

    @property
    def novel_tokens(self) -> int:
        return self._total("novel_tokens")

    @property
    def reprefill_share(self) -> float:
        """Of everything the cache could not supply, how much was a repeat."""
        return self.reused_tokens / self.computed_tokens if self.computed_tokens else 0.0

    def reused_by_decider(self) -> dict[str, int]:
        """Repeated tokens grouped by who caused the edit that turn."""
        totals: dict[str, int] = {}
        for turn in self.turns:
            if turn.reused_tokens:
                totals[turn.decider] = totals.get(turn.decider, 0) + turn.reused_tokens
        return totals


def simulate(
    trace: Trace,
    render: Renderer,
    tokenizer: Tokenizer,
    block_size: int = 1,
) -> TraceStats:
    cache = PrefixCache(block_size=block_size)
    seen: set[tuple[str, str]] = set()
    stats: list[TurnStats] = []

    for turn in trace.turns:
        tokens = tokenizer.encode(render(turn.messages))
        spans = message_spans(turn.messages, render, tokenizer)
        hit = cache.longest_hit(tokens)

        reused = novel = 0
        for message, (start, end) in zip(turn.messages, spans):
            missing = end - max(start, hit)
            if missing <= 0:
                continue
            if (message.role, message.content) in seen:
                reused += missing
            else:
                novel += missing

        change = turn.change_before
        stats.append(
            TurnStats(
                index=turn.index,
                prompt_tokens=len(tokens),
                cached_tokens=hit,
                reused_tokens=reused,
                novel_tokens=novel,
                decider=str(change.decider) if change else "none",
            )
        )
        cache.add(tokens)
        seen.update((m.role, m.content) for m in turn.messages)

    return TraceStats(program_id=trace.program_id, turns=tuple(stats))


def summarize(all_stats: Iterable[TraceStats]) -> dict[str, float | int]:
    """Roll several traces into the handful of numbers the gate turns on."""
    traces = list(all_stats)
    prompt = sum(s.prompt_tokens for s in traces)
    computed = sum(s.computed_tokens for s in traces)
    reused = sum(s.reused_tokens for s in traces)

    by_decider: dict[str, int] = {}
    for stats in traces:
        for decider, count in stats.reused_by_decider().items():
            by_decider[decider] = by_decider.get(decider, 0) + count

    return {
        "traces": len(traces),
        "prompt_tokens": prompt,
        "computed_tokens": computed,
        "reused_tokens": reused,
        "novel_tokens": computed - reused,
        "reprefill_share": reused / computed if computed else 0.0,
        "harness_attributable": by_decider.get("harness", 0),
        "reused_by_decider": by_decider,
    }
