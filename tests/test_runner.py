"""Plumbing checks against a tiny randomly-weighted model.

The numbers mean nothing; a random model has no opinions. What is being checked
is that positions, cache surgery and the decode loop line up, so that when a real
model disagrees across paths it is the residue talking and not a bug here.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from scm.probes import Probe, pad  # noqa: E402
from scm.rotate import HALF_SPLIT, relative_l2  # noqa: E402
from scm.runner import (  # noqa: E402
    PRESETS,
    RunnerError,
    Target,
    decode,
    layer_zero_error,
    prefill,
    preset_for,
    read_cache,
    rope_base,
    run_probe,
    write_cache,
)


class CharTokenizer:
    """One token per character, so every span lands on a boundary."""

    def encode(self, text: str) -> list[int]:
        return [min(ord(c), 255) for c in text]


@pytest.fixture(scope="module")
def target():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    model = LlamaForCausalLM(config).eval()
    return Target(model, CharTokenizer(), HALF_SPLIT, None, rope_base(config))


@pytest.fixture
def probe():
    return Probe(
        name="tiny",
        prefix="the quick brown fox jumps over ",
        span="the lazy dog sleeps soundly ",
        suffix="and then what happened next",
        retained_answer="a",
        erased_answer="b",
    )


# --- cache adapters --------------------------------------------------------


def test_cache_survives_a_read_and_write(target):
    past = prefill(target, target.tokenizer.encode("hello world"))
    layers = read_cache(past)
    rebuilt = read_cache(write_cache(layers))

    assert len(rebuilt) == len(layers)
    for before, after in zip(layers, rebuilt):
        assert torch.equal(before.keys, after.keys)
        assert torch.equal(before.values, after.values)


def test_prefill_caches_everything_but_the_last_token(target):
    ids = target.tokenizer.encode("hello world")
    assert read_cache(prefill(target, ids))[0].length == len(ids) - 1


def test_prefill_needs_more_than_one_token(target):
    with pytest.raises(RunnerError, match="at least 2 tokens"):
        prefill(target, [5])


# --- decode ----------------------------------------------------------------


def test_decode_returns_the_requested_number_of_tokens(target):
    past = prefill(target, target.tokenizer.encode("hello world"))
    assert len(decode(target, past, 100, steps=7).tokens) == 7


def test_decode_records_logits_at_the_first_position(target):
    past = prefill(target, target.tokenizer.encode("hello world"))
    out = decode(target, past, 100, steps=3)
    assert out.first_logits is not None
    assert out.first_logits.shape[-1] == 256


def test_decode_is_deterministic(target):
    ids = target.tokenizer.encode("hello world")
    one = decode(target, prefill(target, ids), 100, steps=5)
    two = decode(target, prefill(target, ids), 100, steps=5)
    assert one.tokens == two.tokens


def test_decode_needs_at_least_one_step(target):
    past = prefill(target, target.tokenizer.encode("hello world"))
    with pytest.raises(RunnerError, match="at least 1"):
        decode(target, past, 100, steps=0)


# --- the splice, end to end ------------------------------------------------


def test_layer_zero_matches_an_honest_reprefill(target, probe):
    """Layer 0 sees only embeddings, so the splice must reproduce it exactly."""
    assert layer_zero_error(target, probe) < 1e-4


def test_layer_zero_still_matches_once_the_span_sits_deep(target, probe):
    assert layer_zero_error(target, pad(probe, before=3, after=3)) < 1e-4


def test_deeper_layers_are_allowed_to_disagree(target, probe):
    """Where layer 0 agrees and layer 2 does not, that gap is the residue."""
    from scm.splice import splice_cache
    from scm.probes import token_span

    start, end = token_span(probe, target.tokenizer)
    spliced = splice_cache(
        read_cache(prefill(target, target.tokenizer.encode(probe.full))),
        start, end, target.layout, target.rope_dim, target.base,
    )
    honest = read_cache(prefill(target, target.tokenizer.encode(probe.edited)))

    assert relative_l2(spliced[0].keys, honest[0].keys) < 1e-4
    assert relative_l2(spliced[-1].values, honest[-1].values) > 1e-4


def test_the_spliced_cache_is_as_long_as_an_honest_one(target, probe):
    from scm.splice import splice_cache
    from scm.probes import token_span

    start, end = token_span(probe, target.tokenizer)
    spliced = splice_cache(
        read_cache(prefill(target, target.tokenizer.encode(probe.full))),
        start, end, target.layout, target.rope_dim, target.base,
    )
    honest = read_cache(prefill(target, target.tokenizer.encode(probe.edited)))
    assert spliced[0].length == honest[0].length


def test_run_probe_decodes_all_three_paths(target, probe):
    trial = run_probe(target, probe, steps=4)
    assert len(trial.full.tokens) == 4
    assert len(trial.reprefill.tokens) == 4
    assert len(trial.spliced.tokens) == 4


def test_run_probe_gives_a_usable_verdict(target, probe):
    trial = run_probe(target, probe, steps=4)
    assert trial.verdict in ("full", "reprefill", "both", "neither")


def test_a_span_reaching_the_end_of_the_prompt_is_rejected(target):
    ending = Probe(
        name="ends-with-span", prefix="abc ", span="xyz", suffix="",
        retained_answer="a", erased_answer="b",
    )
    with pytest.raises(RunnerError, match="span reaches the end"):
        run_probe(target, ending, steps=2)


# --- presets ---------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("meta-llama/Llama-3.1-8B", (HALF_SPLIT, None)),
    ("Qwen/Qwen3-4B", (HALF_SPLIT, None)),
    ("deepseek-ai/DeepSeek-V2-Lite", ("interleaved", 64)),
    ("moonshotai/Moonlight-16B-A3B", ("interleaved", 64)),
])
def test_presets_match_known_models(name, expected):
    assert preset_for(name) == expected


def test_an_unknown_model_refuses_to_guess():
    with pytest.raises(RunnerError, match="no RoPE preset"):
        preset_for("some-lab/BrandNewArch-7B")


def test_every_preset_names_a_real_layout():
    for layout, rope_dim in PRESETS.values():
        assert layout in ("interleaved", "half_split")
        assert rope_dim is None or rope_dim > 0


# --- rope base -------------------------------------------------------------


def test_rope_base_reads_the_modern_config_layout():
    class Config:
        rope_parameters = {"rope_theta": 500000.0, "rope_type": "default"}

    assert rope_base(Config()) == 500000.0


def test_rope_base_reads_the_older_attribute():
    class Config:
        rope_theta = 10000.0

    assert rope_base(Config()) == 10000.0


def test_rope_base_looks_inside_a_text_config():
    class Inner:
        rope_theta = 1000000.0

    class Outer:
        text_config = Inner()

    assert rope_base(Outer()) == 1000000.0


def test_rope_base_refuses_rather_than_guessing():
    """A wrong base rotates by the wrong angle and quietly ruins every number."""
    class Config:
        pass

    with pytest.raises(RunnerError, match="cannot find the RoPE base"):
        rope_base(Config())
