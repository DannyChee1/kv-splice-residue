"""Run one probe three ways through a real model.

    full        prefill the original prompt, decode
    reprefill   prefill the shortened prompt, decode
    spliced     prefill the original, cut the span out of the cache, decode

All three prefill everything but the last token and then feed that token as the
first decode step. That way the spliced path has somewhere to put its first
forward pass: its cache is a token short, so the token it is handed lands at the
right position without being cached twice. The span sits in the middle of the
prompt, so the last token is the same one in all three paths.

Layer 0 is the check that the plumbing works. Its keys and values come from the
token embedding alone, so nothing upstream can colour them, and spliced must
match reprefill there to rotation precision. Deeper layers are allowed to
disagree, and that disagreement is the whole measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scm.agreement import Decode, Trial
from scm.probes import Probe, check_edit, token_span
from scm.rotate import HALF_SPLIT, INTERLEAVED
from scm.splice import Layer, splice_cache

# Most open models rotate the whole key NeoX-style; MLA keeps a position-free
# head and rotates only a trailing slice.
PRESETS: dict[str, tuple[str, int | None]] = {
    "llama": (HALF_SPLIT, None),
    "qwen": (HALF_SPLIT, None),
    "mistral": (HALF_SPLIT, None),
    "gemma": (HALF_SPLIT, None),
    "deepseek": (INTERLEAVED, 64),
    "moonlight": (INTERLEAVED, 64),
}


class RunnerError(RuntimeError):
    pass


def preset_for(name: str) -> tuple[str, int | None]:
    """Guess layout and rotated width from a model name. Override if it guesses wrong."""
    lowered = name.lower()
    for key, value in PRESETS.items():
        if key in lowered:
            return value
    raise RunnerError(
        f"no RoPE preset matches {name!r}; pass layout and rope_dim explicitly"
    )


@dataclass(frozen=True, slots=True)
class Target:
    model: object
    tokenizer: object
    layout: str = HALF_SPLIT
    rope_dim: int | None = None
    base: float = 10000.0

    @property
    def device(self):
        return next(self.model.parameters()).device


def load(
    name: str,
    dtype: str = "float32",
    device: str = "cpu",
    layout: str | None = None,
    rope_dim: int | None = None,
) -> Target:
    """Load weights. fp32 by default: bf16 storage swamps the effect we measure."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if layout is None:
        layout, preset_dim = preset_for(name)
        rope_dim = preset_dim if rope_dim is None else rope_dim

    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=getattr(torch, dtype), trust_remote_code=True
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return Target(model, tokenizer, layout, rope_dim, rope_base(model.config))


def rope_base(config) -> float:
    """Read the RoPE base, wherever this transformers version keeps it.

    Guessing 10000 would be silently wrong on Llama 3.1 and friends at 500000,
    and a wrong base rotates by the wrong angle, so refuse instead of guessing.
    """
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict) and "rope_theta" in params:
        return float(params["rope_theta"])
    try:
        return float(config.rope_theta)
    except AttributeError:
        pass
    inner = getattr(config, "text_config", None)
    if inner is not None and inner is not config:
        return rope_base(inner)
    raise RunnerError(
        f"cannot find the RoPE base on {type(config).__name__}; pass base explicitly"
    )


def read_cache(past) -> list[Layer]:
    """Pull key/value tensors out of whichever Cache class this version uses."""
    if hasattr(past, "layers"):
        return [Layer(keys=la.keys, values=la.values) for la in past.layers]
    if hasattr(past, "key_cache"):
        return [Layer(keys=k, values=v) for k, v in zip(past.key_cache, past.value_cache)]
    return [Layer(keys=k, values=v) for k, v in past]


def write_cache(layers: list[Layer]):
    from transformers import DynamicCache

    cache = DynamicCache()
    for index, layer in enumerate(layers):
        cache.update(layer.keys, layer.values, index)
    return cache


def _forward(target: Target, ids: torch.Tensor, past=None, offset: int = 0):
    positions = torch.arange(
        offset, offset + ids.shape[-1], device=target.device
    )
    with torch.no_grad():
        return target.model(
            input_ids=ids,
            past_key_values=past,
            cache_position=positions,
            use_cache=True,
        )


def prefill(target: Target, token_ids: list[int]):
    """Cache everything but the last token; that one starts the decode."""
    if len(token_ids) < 2:
        raise RunnerError(f"need at least 2 tokens to prefill, got {len(token_ids)}")
    ids = torch.tensor([token_ids[:-1]], device=target.device)
    return _forward(target, ids).past_key_values


def decode(target: Target, past, first_token: int, steps: int) -> Decode:
    """Greedy decode, keeping the logits at the first generated position."""
    if steps < 1:
        raise RunnerError(f"steps must be at least 1, got {steps}")
    offset = read_cache(past)[0].length
    token = first_token
    produced: list[int] = []
    first_logits = None

    for _ in range(steps):
        ids = torch.tensor([[token]], device=target.device)
        out = _forward(target, ids, past, offset)
        past, offset = out.past_key_values, offset + 1
        logits = out.logits[0, -1].float()
        if first_logits is None:
            first_logits = logits.cpu()
        token = int(logits.argmax())
        produced.append(token)

    return Decode(tokens=tuple(produced), first_logits=first_logits)


def run_probe(target: Target, probe: Probe, steps: int = 128) -> Trial:
    """Prefill, cut, and decode the three paths for one probe."""
    check_edit(probe, target.tokenizer)
    start, end = token_span(probe, target.tokenizer)

    full_ids = target.tokenizer.encode(probe.full)
    edited_ids = target.tokenizer.encode(probe.edited)
    if full_ids[-1] != edited_ids[-1]:
        raise RunnerError(
            f"{probe.name}: prompts end differently, so the span reaches the end"
        )
    last = full_ids[-1]

    full_cache = prefill(target, full_ids)
    spliced = write_cache(
        splice_cache(
            read_cache(full_cache), start, end,
            target.layout, target.rope_dim, target.base,
        )
    )
    return Trial(
        full=decode(target, full_cache, last, steps),
        reprefill=decode(target, prefill(target, edited_ids), last, steps),
        spliced=decode(target, spliced, last, steps),
    )


def layer_zero_error(target: Target, probe: Probe) -> float:
    """rel-L2 between the spliced and honestly reprefilled layer-0 keys.

    Layer 0 depends only on token embeddings, so this should sit at rotation
    precision. Anything larger means the splice or the positions are wrong, not
    that residue was found.
    """
    from scm.rotate import relative_l2

    start, end = token_span(probe, target.tokenizer)
    spliced = splice_cache(
        read_cache(prefill(target, target.tokenizer.encode(probe.full))),
        start, end, target.layout, target.rope_dim, target.base,
    )[0]
    honest = read_cache(prefill(target, target.tokenizer.encode(probe.edited)))[0]
    return relative_l2(spliced.keys, honest.keys)
