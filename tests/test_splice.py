import pytest
import torch

from scm.rotate import HALF_SPLIT, INTERLEAVED, apply_rope, relative_l2, rotate
from scm.splice import (
    Layer,
    SpliceError,
    splice_cache,
    splice_keys,
    splice_layer,
    splice_values,
)

DIM = 64
SEQ = 12


def tensor(seq: int = SEQ, dim: int = DIM, seed: int = 0) -> torch.Tensor:
    return torch.randn(1, 2, seq, dim, generator=torch.manual_seed(seed))


def placed(seq: int = SEQ, layout: str = INTERLEAVED) -> torch.Tensor:
    """Keys sitting at positions 0..seq-1, the way a prefill leaves them."""
    return apply_rope(tensor(seq), torch.arange(seq), layout)


def layer(seq: int = SEQ) -> Layer:
    return Layer(keys=placed(seq), values=tensor(seq, seed=1))


# --- what the splice removes -----------------------------------------------


def test_splicing_shortens_the_cache_by_the_span():
    out = splice_keys(placed(), 4, 7)
    assert out.shape[-2] == SEQ - 3


def test_the_prefix_is_left_alone():
    keys = placed()
    assert torch.equal(splice_keys(keys, 4, 7)[..., :4, :], keys[..., :4, :])


def test_the_tail_is_re_anchored_by_the_shift():
    keys = placed()
    spliced = splice_keys(keys, 4, 7)
    assert torch.equal(spliced[..., 4:, :], rotate(keys[..., 7:, :], -3))


def test_re_anchored_keys_sit_where_an_honest_prefill_would_put_them():
    """Positions line up again after the cut. That is all the rotation buys."""
    raw = tensor()
    keys = apply_rope(raw, torch.arange(SEQ), INTERLEAVED)
    spliced = splice_keys(keys, 4, 7)

    kept = torch.cat((raw[..., :4, :], raw[..., 7:, :]), dim=-2)
    honest = apply_rope(kept, torch.arange(SEQ - 3), INTERLEAVED)
    assert relative_l2(spliced, honest) < 1e-5


def test_splicing_nothing_returns_the_cache_unchanged():
    keys, values = placed(), tensor()
    assert splice_keys(keys, 5, 5) is keys
    assert splice_values(values, 5, 5) is values


def test_splicing_everything_leaves_an_empty_cache():
    assert splice_keys(placed(), 0, SEQ).shape[-2] == 0
    assert splice_values(tensor(), 0, SEQ).shape[-2] == 0


# --- what the splice leaves behind -----------------------------------------


def test_values_after_the_span_are_carried_over_untouched():
    """The residue, stated as a test: these were computed attending to the span."""
    values = tensor()
    spliced = splice_values(values, 4, 7)
    assert torch.equal(spliced[..., 4:, :], values[..., 7:, :])
    assert torch.equal(spliced[..., :4, :], values[..., :4, :])


def test_values_are_never_rotated():
    values = tensor()
    spliced = splice_values(values, 4, 7)
    rotated = rotate(values[..., 7:, :], -3)
    assert not torch.allclose(spliced[..., 4:, :], rotated)


def test_the_position_free_part_of_an_mla_key_survives_the_splice():
    """MLA splits the key; RoPE only reaches the tail, so the head is residue too."""
    keys = tensor()
    spliced = splice_keys(keys, 4, 7, rope_dim=16)
    assert torch.equal(spliced[..., 4:, :-16], keys[..., 7:, :-16])
    assert not torch.equal(spliced[..., 4:, -16:], keys[..., 7:, -16:])


def test_rope_dim_covering_the_whole_key_matches_no_rope_dim():
    keys = placed()
    assert torch.equal(splice_keys(keys, 4, 7, rope_dim=DIM), splice_keys(keys, 4, 7))


@pytest.mark.parametrize("rope_dim", [0, -4, DIM + 8])
def test_an_impossible_rope_dim_is_rejected(rope_dim):
    with pytest.raises(SpliceError, match="does not fit"):
        splice_keys(placed(), 4, 7, rope_dim=rope_dim)


# --- layout ----------------------------------------------------------------


def test_layout_reaches_the_splice():
    keys = placed(layout=HALF_SPLIT)
    assert not torch.allclose(
        splice_keys(keys, 4, 7, HALF_SPLIT), splice_keys(keys, 4, 7, INTERLEAVED)
    )


def test_base_reaches_the_splice():
    keys = placed()
    assert not torch.allclose(
        splice_keys(keys, 4, 7, base=500000.0), splice_keys(keys, 4, 7)
    )


# --- spans -----------------------------------------------------------------


@pytest.mark.parametrize("span", [(7, 4), (-1, 3)])
def test_malformed_spans_are_rejected(span):
    with pytest.raises(SpliceError, match="bad span"):
        splice_keys(placed(), *span)


def test_a_span_past_the_end_is_rejected():
    with pytest.raises(SpliceError, match="span ends at 99, cache holds 12"):
        splice_keys(placed(), 4, 99)
    with pytest.raises(SpliceError, match="cache holds"):
        splice_values(tensor(), 4, 99)


# --- layers and stacks -----------------------------------------------------


def test_a_layer_needs_as_many_keys_as_values():
    with pytest.raises(SpliceError, match="12 keys but 8 values"):
        Layer(keys=placed(12), values=tensor(8))


def test_splicing_a_layer_does_both_halves():
    out = splice_layer(layer(), 4, 7)
    assert out.length == SEQ - 3
    assert torch.equal(out.values[..., 4:, :], tensor(SEQ, seed=1)[..., 7:, :])


def test_every_layer_gets_the_same_cut():
    cache = [layer() for _ in range(4)]
    out = splice_cache(cache, 4, 7)
    assert [la.length for la in out] == [SEQ - 3] * 4


def test_an_empty_cache_is_rejected():
    with pytest.raises(SpliceError, match="no layers"):
        splice_cache([], 0, 1)


def test_layers_of_different_lengths_are_rejected():
    with pytest.raises(SpliceError, match="layers disagree on length"):
        splice_cache([layer(12), layer(8)], 0, 1)


def test_dtype_survives_the_splice():
    la = Layer(keys=placed().to(torch.bfloat16), values=tensor().to(torch.bfloat16))
    out = splice_layer(la, 4, 7)
    assert out.keys.dtype is torch.bfloat16
    assert out.values.dtype is torch.bfloat16
