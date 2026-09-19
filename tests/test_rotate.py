import pytest
import torch

from scm.rotate import (
    HALF_SPLIT,
    INTERLEAVED,
    apply_rope,
    inv_freq,
    relative_l2,
    rotate,
)

DIM = 64


def keys(seq: int = 8, dim: int = DIM, seed: int = 0) -> torch.Tensor:
    return torch.randn(2, seq, dim, generator=torch.manual_seed(seed))


def at(*positions: int) -> torch.Tensor:
    return torch.tensor(positions, dtype=torch.long)


@pytest.fixture(params=[INTERLEAVED, HALF_SPLIT])
def layout(request):
    return request.param


# --- the property the whole splice rests on --------------------------------


def test_rotating_matches_prefilling_at_the_new_position(layout):
    """R(delta) on a key at p equals RoPE at p+delta. This is the contract."""
    k = keys()
    start, delta = at(*range(100, 108)), -46
    rotated = rotate(apply_rope(k, start, layout), delta, layout)
    honest = apply_rope(k, start + delta, layout)
    assert relative_l2(rotated, honest) < 1e-6


@pytest.mark.parametrize("delta", [-6794, -200, -1, 1, 200, 8836])
def test_contract_holds_across_the_range_leyline_swept(delta, layout):
    """Their App. Q sweeps positions to 8836 and |delta| to 6794, fp32 under 1e-3."""
    k = keys(seq=4)
    start = at(9000, 9001, 9002, 9003)
    rotated = rotate(apply_rope(k, start, layout), delta, layout)
    assert relative_l2(rotated, apply_rope(k, start + delta, layout)) < 1e-3


def test_error_grows_with_absolute_position_and_stays_under_the_fp32_claim():
    """float32 holds the angle, so far-out keys drift. Agent contexts run long."""
    k = keys(seq=4)

    def drift(start: int) -> float:
        p = at(*range(start, start + 4))
        return relative_l2(
            rotate(apply_rope(k, p, INTERLEAVED), -46, INTERLEAVED),
            apply_rope(k, p + (-46), INTERLEAVED),
        )

    near, mid, far = drift(0), drift(9_000), drift(60_000)
    assert near < mid < far < 1e-3


def test_rotations_compose(layout):
    k = apply_rope(keys(), at(*range(50, 58)), layout)
    once = rotate(k, -30, layout)
    twice = rotate(rotate(k, -10, layout), -20, layout)
    assert relative_l2(twice, once) < 1e-6


def test_rotating_by_zero_changes_nothing(layout):
    k = apply_rope(keys(), at(*range(8)), layout)
    assert torch.equal(rotate(k, 0, layout), k)


def test_rotation_preserves_length(layout):
    """It is a rotation, so every key keeps its norm."""
    k = apply_rope(keys(), at(*range(8)), layout)
    before = torch.linalg.vector_norm(k, dim=-1)
    after = torch.linalg.vector_norm(rotate(k, -137, layout), dim=-1)
    assert torch.allclose(before, after, atol=1e-5)


def test_rotation_does_not_depend_on_where_the_key_started(layout):
    """One angle serves every shifted key, which is why the splice is cheap."""
    k = keys(seq=1)
    near = rotate(apply_rope(k, at(10), layout), -5, layout)
    far = rotate(apply_rope(k, at(9000), layout), -5, layout)
    assert relative_l2(near, apply_rope(k, at(5), layout)) < 1e-6
    assert relative_l2(far, apply_rope(k, at(8995), layout)) < 1e-3


# --- layout is load-bearing ------------------------------------------------


def test_the_two_layouts_disagree():
    """Picking the wrong one is silent, so prove they are not interchangeable."""
    k = apply_rope(keys(), at(*range(8)), INTERLEAVED)
    assert relative_l2(rotate(k, -40, HALF_SPLIT), rotate(k, -40, INTERLEAVED)) > 0.1


def test_rotating_under_the_wrong_layout_breaks_the_contract():
    k, start, delta = keys(), at(*range(100, 108)), -46
    placed = apply_rope(k, start, INTERLEAVED)
    honest = apply_rope(k, start + delta, INTERLEAVED)
    assert relative_l2(rotate(placed, delta, HALF_SPLIT), honest) > 0.1


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match="layout must be one of"):
        rotate(keys(), 1, "sideways")


def test_odd_dim_is_rejected():
    with pytest.raises(ValueError, match="must be even"):
        rotate(torch.randn(2, 4, 7), 1)


def test_position_count_must_match_key_count():
    with pytest.raises(ValueError, match="3 positions for 8 keys"):
        apply_rope(keys(), at(0, 1, 2))


# --- precision floor Leyline reports ---------------------------------------


def test_bf16_storage_costs_about_a_percent_and_fp32_does_not():
    """Their App. Q: ~1-3% per-entry error in bf16, under 1e-3 in fp32."""
    k, start, delta = keys(seq=16), at(*range(500, 516)), -46
    honest = apply_rope(k, start + delta, INTERLEAVED)

    fp32 = rotate(apply_rope(k, start, INTERLEAVED), delta, INTERLEAVED)
    stored = apply_rope(k, start, INTERLEAVED).to(torch.bfloat16)
    bf16 = rotate(stored, delta, INTERLEAVED).to(torch.float32)

    assert relative_l2(fp32, honest) < 1e-3
    assert 0.001 < relative_l2(bf16, honest) < 0.05


@pytest.mark.parametrize("delta", [-4000, -46, 46, 4000])
def test_the_bf16_floor_does_not_depend_on_delta(delta):
    """They report the floor flat in delta; a delta-dependent one means a bug."""
    k = keys(seq=16)
    start = at(*range(5000, 5016))
    stored = apply_rope(k, start, INTERLEAVED).to(torch.bfloat16)
    error = relative_l2(
        rotate(stored, delta, INTERLEAVED).to(torch.float32),
        apply_rope(k, start + delta, INTERLEAVED),
    )
    assert 0.001 < error < 0.05


# --- helpers ---------------------------------------------------------------


def test_inv_freq_decreases_across_pairs():
    freqs = inv_freq(DIM)
    assert freqs.shape == (DIM // 2,)
    assert freqs[0] == pytest.approx(1.0)
    assert torch.all(freqs[1:] < freqs[:-1])


def test_base_changes_the_rotation():
    k = apply_rope(keys(), at(*range(8)), INTERLEAVED)
    assert relative_l2(
        rotate(k, -40, INTERLEAVED, base=500000.0),
        rotate(k, -40, INTERLEAVED, base=10000.0),
    ) > 0.01


def test_relative_l2_is_zero_for_identical_tensors():
    k = keys()
    assert relative_l2(k, k) == 0.0


def test_relative_l2_rejects_an_all_zero_reference():
    with pytest.raises(ValueError, match="all zeros"):
        relative_l2(torch.ones(4), torch.zeros(4))


def test_dtype_survives_a_rotation():
    k = apply_rope(keys(), at(*range(8)), INTERLEAVED).to(torch.bfloat16)
    assert rotate(k, -10, INTERLEAVED).dtype is torch.bfloat16


# --- explicit frequencies --------------------------------------------------


def test_explicit_freqs_reproduce_the_default():
    k = apply_rope(keys(), at(*range(8)), INTERLEAVED)
    from scm.rotate import inv_freq as default_freqs
    assert torch.equal(
        rotate(k, -40, INTERLEAVED, freqs=default_freqs(DIM)),
        rotate(k, -40, INTERLEAVED),
    )


def test_scaled_freqs_rotate_differently():
    """A YaRN ladder is not the plain one, so the rotation must differ."""
    from scm.rotate import inv_freq as default_freqs
    k = apply_rope(keys(), at(*range(8)), INTERLEAVED)
    bent = default_freqs(DIM) * 0.84
    assert relative_l2(rotate(k, -40, INTERLEAVED, freqs=bent),
                       rotate(k, -40, INTERLEAVED)) > 0.01


def test_the_wrong_number_of_freqs_is_rejected():
    with pytest.raises(ValueError, match="frequencies for a 64-wide key"):
        rotate(keys(), -1, INTERLEAVED, freqs=torch.ones(64))


def test_the_contract_holds_under_a_bent_ladder():
    """Rotation still lands where a prefill would, whatever the ladder."""
    from scm.rotate import inv_freq as default_freqs
    bent = default_freqs(DIM) * 0.84
    raw, start, delta = keys(), at(*range(100, 108)), -46
    angle_start = start.float()[:, None] * bent
    angle_end = (start + delta).float()[:, None] * bent
    from scm.rotate import _spin
    placed = _spin(raw, angle_start, INTERLEAVED)
    honest = _spin(raw, angle_end, INTERLEAVED)
    assert relative_l2(rotate(placed, delta, INTERLEAVED, freqs=bent), honest) < 1e-6
