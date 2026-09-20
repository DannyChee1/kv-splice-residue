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

# Ordinary attention caches keys and values and rotates the whole key. MLA
# caches neither: the keys slot holds the compressed latent and the values slot
# holds k_pe, so `mla` says which tensor the rotation belongs on.
PRESETS: dict[str, tuple[str, int | None, bool]] = {
    "llama": (HALF_SPLIT, None, False),
    "qwen": (HALF_SPLIT, None, False),
    "mistral": (HALF_SPLIT, None, False),
    "gemma": (HALF_SPLIT, None, False),
    "deepseek": (HALF_SPLIT, None, True),
    "moonlight": (HALF_SPLIT, None, True),
}


class RunnerError(RuntimeError):
    pass


# Keyed on config.model_type, which says what the architecture is rather than
# what somebody called the upload.
BY_TYPE: dict[str, tuple[str, int | None, bool]] = {
    "llama": (HALF_SPLIT, None, False),
    "qwen2": (HALF_SPLIT, None, False),
    "qwen3": (HALF_SPLIT, None, False),
    "qwen3_moe": (HALF_SPLIT, None, False),
    "mistral": (HALF_SPLIT, None, False),
    "gemma2": (HALF_SPLIT, None, False),
    "gemma3": (HALF_SPLIT, None, False),
    "deepseek_v2": (HALF_SPLIT, None, True),
    "deepseek_v3": (HALF_SPLIT, None, True),
    "deepseek_v32": (HALF_SPLIT, None, True),
    "kimi_k2": (HALF_SPLIT, None, True),
}


def preset_for_type(model_type: str) -> tuple[str, int | None, bool]:
    """Settings for an architecture, by the name the config gives itself."""
    try:
        return BY_TYPE[model_type]
    except KeyError:
        raise RunnerError(
            f"no RoPE preset for model_type {model_type!r}; "
            "pass layout and rope_dim explicitly"
        ) from None


def preset_for(name: str) -> tuple[str, int | None, bool]:
    """Fallback guess from a repo name, for when no config is to hand.

    Refuses an ambiguous name rather than letting dict order decide. A repo
    called DeepSeek-R1-Distill-Llama-8B is a Llama, but nothing in the string
    says so, and picking the wrong one rotates the wrong tensor.
    """
    lowered = name.lower()
    hits = [key for key in PRESETS if key in lowered]
    if not hits:
        raise RunnerError(
            f"no RoPE preset matches {name!r}; pass layout and rope_dim explicitly"
        )
    if len(hits) > 1:
        raise RunnerError(
            f"{name!r} matches {', '.join(sorted(hits))}; load the config and use "
            "preset_for_type, or pass layout and rope_dim explicitly"
        )
    return PRESETS[hits[0]]


@dataclass(frozen=True, slots=True)
class Target:
    model: object
    tokenizer: object
    layout: str = HALF_SPLIT
    rope_dim: int | None = None
    base: float = 10000.0
    mla: bool = False
    freqs: object = None

    @property
    def device(self):
        return next(self.model.parameters()).device


def load(
    name: str,
    dtype: str = "float32",
    device: str = "cpu",
    layout: str | None = None,
    rope_dim: int | None = None,
    remote_code: bool = False,
) -> Target:
    """Load weights, preferring the implementation that ships with transformers.

    `remote_code` stays off on purpose. A repo's own modeling file is written
    against whatever transformers existed when it was uploaded, and more to the
    point, everything we know about the MLA cache layout was measured against
    the native classes. Custom code could cache differently and the splice would
    be wrong without saying so.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from transformers import AutoConfig

    probe = AutoConfig.from_pretrained(name, trust_remote_code=remote_code)
    try:
        guessed_layout, preset_dim, mla = preset_for_type(probe.model_type)
    except RunnerError:
        guessed_layout, preset_dim, mla = preset_for(name)
    if layout is None:
        layout = guessed_layout
        rope_dim = preset_dim if rope_dim is None else rope_dim

    # device_map streams shards straight onto the GPU. Loading to CPU and then
    # calling .to() needs the whole model in RAM first, which stalls a 30GB load
    # on a box sized for the weights alone.
    kwargs = {"dtype": getattr(torch, dtype), "trust_remote_code": remote_code}
    if device != "cpu":
        kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if device == "cpu":
        model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=remote_code)

    # Do not read config.rope_interleave here. It describes how the weights are
    # laid out, not the cache: DeepSeek's interleaved path reorders its input and
    # emits output bit-identical to the rotate_half form, so what lands in the
    # cache is half-split either way. Measured, not assumed; see the test.

    width = rope_width(model.config) if mla else None
    return Target(model, tokenizer, layout, rope_dim,
                  rope_base(model.config), mla, model_freqs(model, width))


def rope_width(config) -> int | None:
    return getattr(config, "qk_rope_head_dim", None)


def model_freqs(model, width: int | None = None):
    """Take the model's own inv_freq rather than rebuilding it from a base.

    YaRN and the other scalings bend the frequency ladder, so a ladder rebuilt
    from `base` rotates by the wrong angle and every number downstream is quietly
    wrong. DeepSeek-V2-Lite is a yarn model, so this is not hypothetical.
    """
    for module in model.modules():
        freqs = getattr(module, "inv_freq", None)
        if freqs is None:
            continue
        if width is not None and freqs.shape[-1] != width // 2:
            freqs = freqs[..., : width // 2]
        return freqs.detach()
    return None


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
            target.mla, target.freqs,
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
        target.mla, target.freqs,
    )[0]
    honest = read_cache(prefill(target, target.tokenizer.encode(probe.edited)))[0]
    return relative_l2(spliced.keys, honest.keys)


def score_answer(target: Target, past, first_token: int, answer_ids: list[int]) -> float:
    """Teacher-force an answer through a cache and report its mean log-probability.

    Sensitive where an argmax is not: a leftover that lifts the answer from
    unlikely to plausible shows up here even when it never wins the argmax.
    """
    from scm.recall import mean_logprob

    if not answer_ids:
        raise RunnerError("no answer tokens to score")
    ids = torch.tensor([[first_token, *answer_ids[:-1]]], device=target.device)
    offset = read_cache(past)[0].length
    out = _forward(target, ids, past, offset)
    return mean_logprob(
        out.logits[0], torch.tensor(answer_ids, device=out.logits.device)
    )


def run_recall(target: Target, probe, steps: int = 0) -> dict:
    """Score one recall probe three ways and report what the splice kept."""
    from scm.recall import Score, span_tokens

    start, end = span_tokens(probe, target.tokenizer)
    full_ids = target.tokenizer.encode(probe.full)
    edited_ids = target.tokenizer.encode(probe.edited)
    answer_ids = target.tokenizer.encode(probe.answer, add_special_tokens=False)
    if not answer_ids:
        raise RunnerError(f"{probe.name}: answer tokenizes to nothing")

    full_cache = prefill(target, full_ids)
    spliced = write_cache(
        splice_cache(read_cache(full_cache), start, end, target.layout,
                     target.rope_dim, target.base, target.mla, target.freqs)
    )
    last = full_ids[-1]
    score = Score(
        full=score_answer(target, full_cache, last, answer_ids),
        reprefill=score_answer(target, prefill(target, edited_ids), last, answer_ids),
        spliced=score_answer(target, spliced, last, answer_ids),
    )
    return {
        "probe": probe.name,
        "prompt_tokens": len(full_ids),
        "answer_tokens": len(answer_ids),
        "full": score.full,
        "reprefill": score.reprefill,
        "spliced": score.spliced,
        "fact_helped": score.full - score.reprefill,
        "recoverable": score.recoverable if score.full > score.reprefill else None,
        "layer0_error": layer_zero_error_ids(target, full_ids, edited_ids, start, end),
    }


def layer_zero_error_ids(target: Target, full_ids, edited_ids, start, end) -> float:
    from scm.rotate import relative_l2

    spliced = splice_cache(
        read_cache(prefill(target, full_ids)), start, end, target.layout,
        target.rope_dim, target.base, target.mla, target.freqs,
    )[0]
    honest = read_cache(prefill(target, edited_ids))[0]
    left = spliced.keys if target.mla else spliced.values
    right = honest.keys if target.mla else honest.values
    return relative_l2(left[..., :start, :], right[..., :start, :])
