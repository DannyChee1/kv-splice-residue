"""Which reference does a spliced cache behave like?

Every edit is run three ways and the decodes compared:

    full        the original prompt, prefilled honestly, edit never applied
    reprefill   the edited prompt, prefilled honestly
    spliced     the original prefilled, then patched in place

`full` is the model still under the deleted span's influence; `reprefill` is the
model genuinely rid of it. A trial only tells us anything when those two part
ways, so `references_diverge` gates every count.

Leyline reports its splice tracking `full`, which is the residue their positional
contract leaves behind. SCM recomputes under the edited context, so it should
track `reprefill` by construction. Running both through the same verdict is the
experiment.

Metrics follow theirs so the numbers line up: first-token agreement, mean common
prefix over a greedy decode, KL, and top-k overlap. Tensor distance is rel-L2 in
scm.rotate, next to the rotation whose error it measures.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

FULL = "full"
REPREFILL = "reprefill"
BOTH = "both"
NEITHER = "neither"


def common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    length = 0
    for x, y in zip(a, b):
        if x != y:
            break
        length += 1
    return length


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p || q) in nats, from raw logits over one vocabulary."""
    if p_logits.shape != q_logits.shape:
        raise ValueError(f"shapes differ: {p_logits.shape} vs {q_logits.shape}")
    p_log = torch.log_softmax(p_logits.to(torch.float32), dim=-1)
    q_log = torch.log_softmax(q_logits.to(torch.float32), dim=-1)
    return torch.sum(p_log.exp() * (p_log - q_log)).item()


def top_k_overlap(p_logits: torch.Tensor, q_logits: torch.Tensor, k: int = 10) -> float:
    """Share of the top-k the two distributions agree on."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    k = min(k, p_logits.shape[-1])
    p_top = set(torch.topk(p_logits, k).indices.tolist())
    q_top = set(torch.topk(q_logits, k).indices.tolist())
    return len(p_top & q_top) / k


@dataclass(frozen=True, slots=True)
class Decode:
    """A greedy decode, plus the logits at the first generated position."""

    tokens: tuple[int, ...]
    first_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("decode has no tokens")


@dataclass(frozen=True, slots=True)
class Trial:
    full: Decode
    reprefill: Decode
    spliced: Decode

    @property
    def references_diverge(self) -> bool:
        """Did the edit change the model's answer at all? Nothing to see if not."""
        return self.full.tokens[0] != self.reprefill.tokens[0]

    @property
    def verdict(self) -> str:
        first = self.spliced.tokens[0]
        matches_full = first == self.full.tokens[0]
        matches_rp = first == self.reprefill.tokens[0]
        if matches_full and matches_rp:
            return BOTH
        if matches_full:
            return FULL
        if matches_rp:
            return REPREFILL
        return NEITHER

    def prefix_lengths(self) -> dict[str, int]:
        return {
            FULL: common_prefix_length(self.spliced.tokens, self.full.tokens),
            REPREFILL: common_prefix_length(self.spliced.tokens, self.reprefill.tokens),
        }

    def divergences(self) -> dict[str, float]:
        """KL of the spliced next-token distribution against each reference."""
        if self.spliced.first_logits is None:
            raise ValueError("trial has no logits; record first_logits to use this")
        out = {}
        for name, ref in ((FULL, self.full), (REPREFILL, self.reprefill)):
            if ref.first_logits is None:
                raise ValueError(f"{name} reference has no logits")
            out[name] = kl_divergence(self.spliced.first_logits, ref.first_logits)
        return out


@dataclass(frozen=True, slots=True)
class Report:
    trials: int
    informative: int
    verdicts: dict[str, int]
    mean_prefix: dict[str, float]

    @property
    def tracks_reprefill(self) -> float:
        """Share of informative trials where the splice behaved like honest prefill."""
        if not self.informative:
            return 0.0
        return self.verdicts.get(REPREFILL, 0) / self.informative


def summarize(trials: Iterable[Trial]) -> Report:
    """Count verdicts over the trials where the references actually separate."""
    trials = list(trials)
    informative = [t for t in trials if t.references_diverge]

    verdicts = Counter(t.verdict for t in informative)
    prefixes = [t.prefix_lengths() for t in informative]
    mean_prefix = {
        name: sum(p[name] for p in prefixes) / len(prefixes) if prefixes else 0.0
        for name in (FULL, REPREFILL)
    }
    return Report(
        trials=len(trials),
        informative=len(informative),
        verdicts=dict(verdicts),
        mean_prefix=mean_prefix,
    )
