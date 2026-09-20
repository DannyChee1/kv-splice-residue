"""Can the model still recall a fact whose tokens have been cut out?

The earlier probes asked about the span directly, so deleting it left nothing to
answer from and every path agreed. That tests nothing: direct attention is gone
whether you splice or reprefill honestly.

A recall probe puts a bridge between the fact and the question. The bridge refers
back to the fact without repeating it ("I connected to that host"), so the bridge
tokens' own cache entries are the only place the fact survives a cut. Those
entries are exactly what a positional splice carries over untouched and an honest
reprefill rebuilds without ever seeing the fact.

    full        fact present
    reprefill   fact gone, bridge recomputed without it
    spliced     fact's tokens gone, bridge entries kept from when it was there

Scoring is the mean log-probability of the answer under teacher forcing, not an
argmax. A residue that shifts the answer from unlikely to likely is real even
when it never wins the argmax, and argmax is what made the last run unreadable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class RecallError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Recall:
    """A fact, a bridge that leans on it, and the question it answers."""

    name: str
    prefix: str
    span: str
    bridge: str
    question: str
    answer: str

    def __post_init__(self) -> None:
        if not self.span:
            raise RecallError(f"{self.name}: span is empty")
        if not self.bridge:
            raise RecallError(
                f"{self.name}: no bridge, so nothing downstream remembers the fact"
            )
        if not self.answer:
            raise RecallError(f"{self.name}: no answer to score")
        if self.answer in self.bridge or self.answer in self.question:
            raise RecallError(
                f"{self.name}: the answer appears outside the span, so the model "
                "can read it off instead of recalling it"
            )

    @property
    def full(self) -> str:
        return self.prefix + self.span + self.bridge + self.question

    @property
    def edited(self) -> str:
        return self.prefix + self.bridge + self.question


SUITE: tuple[Recall, ...] = (
    Recall(
        name="database-host",
        prefix="Deployment notes for the staging cluster.\n",
        span="The primary database host is 10.0.4.22.\n",
        bridge="I connected to that host and ran the migration. It finished in "
               "four seconds and the replica synced with no errors.\n",
        question="What is the IP address of the primary database host?\n",
        answer="10.0.4.22",
    ),
    Recall(
        name="error-code",
        prefix="Build log follows.\n",
        span="The build failed with error E4471.\n",
        bridge="I looked that error code up and it turned out to be a missing "
               "linker flag. Adding the flag made the build pass.\n",
        question="Which error code did the build fail with?\n",
        answer="E4471",
    ),
    Recall(
        name="pinned-version",
        prefix="Dependency notes.\n",
        span="We pinned numpy to 1.26.4 last week.\n",
        bridge="That pin resolved the ABI mismatch and the whole suite went "
               "green again on the nightly runner.\n",
        question="Which numpy version did we pin?\n",
        answer="1.26.4",
    ),
    Recall(
        name="assigned-reviewer",
        prefix="Meeting minutes.\n",
        span="The reviewer assigned to the proposal is Dr Halvorsen.\n",
        bridge="I emailed that reviewer and they confirmed they are free on "
               "Thursday afternoon to go through it.\n",
        question="Who was assigned to review the proposal?\n",
        answer="Halvorsen",
    ),
    Recall(
        name="ticket-number",
        prefix="Support queue.\n",
        span="The escalation is tracked under ticket QF-8823.\n",
        bridge="I added a note to that ticket and moved it into the current "
               "sprint so it does not get lost again.\n",
        question="What is the tracking ticket for the escalation?\n",
        answer="QF-8823",
    ),
)


def span_tokens(probe: Recall, tokenizer) -> tuple[int, int]:
    """Token range of the fact inside the full prompt, refusing a ragged cut."""
    full = tokenizer.encode(probe.full)
    head = tokenizer.encode(probe.prefix)
    through = tokenizer.encode(probe.prefix + probe.span)

    if full[: len(head)] != head or full[: len(through)] != through:
        raise RecallError(f"{probe.name}: the span does not sit on token boundaries")
    cut = full[: len(head)] + full[len(through):]
    if cut != tokenizer.encode(probe.edited):
        raise RecallError(f"{probe.name}: cutting the span does not give the edited prompt")
    return len(head), len(through)


@dataclass(frozen=True, slots=True)
class Score:
    """Mean log-probability of the answer, per path."""

    full: float
    reprefill: float
    spliced: float

    @property
    def recoverable(self) -> float:
        """How much of the fact's value the splice keeps that a reprefill loses.

        1.0 means the splice recalls as well as if the fact were never cut;
        0.0 means it recalls no better than an honest reprefill. Residue that
        matters behaviourally shows up here and nowhere else.
        """
        room = self.full - self.reprefill
        if room <= 0:
            raise RecallError(
                "the fact did not help the model, so there is nothing to recover"
            )
        return (self.spliced - self.reprefill) / room


def mean_logprob(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean log-probability the given logits assign to the target tokens."""
    if logits.shape[0] != targets.shape[0]:
        raise RecallError(
            f"{logits.shape[0]} logit rows for {targets.shape[0]} targets"
        )
    if targets.numel() == 0:
        raise RecallError("no target tokens to score")
    logprobs = torch.log_softmax(logits.to(torch.float32), dim=-1)
    picked = logprobs.gather(-1, targets.view(-1, 1)).squeeze(-1)
    return picked.mean().item()
