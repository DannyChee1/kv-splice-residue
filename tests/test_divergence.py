import pytest

torch = pytest.importorskip("torch")

from scm.divergence import Decay, decay, shadow_length  # noqa: E402
from scm.splice import Layer  # noqa: E402


def layer(values: list[float], seq: int | None = None) -> Layer:
    """One entry per value, the value scaling how far it sits from the reference."""
    seq = seq or len(values)
    keys = torch.ones(1, 1, seq, 4)
    vals = torch.ones(1, 1, seq, 4)
    for i, v in enumerate(values):
        vals[0, 0, i] = 1.0 + v
    return Layer(keys=keys, values=vals)


def flat(seq: int) -> Layer:
    return Layer(keys=torch.ones(1, 1, seq, 4), values=torch.ones(1, 1, seq, 4))


# --- settles_at ------------------------------------------------------------


def test_a_gap_that_fades_settles_once_it_stays_below_epsilon():
    d = Decay(layer=0, distances=(0.5, 0.2, 0.05, 0.01, 0.001))
    assert d.settles_at(0.1) == 2
    assert d.settles_at(0.02) == 3


def test_a_gap_that_never_fades_never_settles():
    assert Decay(layer=0, distances=(0.5, 0.5, 0.5)).settles_at(0.1) is None


def test_a_gap_already_inside_epsilon_settles_immediately():
    assert Decay(layer=0, distances=(0.01, 0.01)).settles_at(0.1) == 0


def test_a_late_spike_pushes_the_settling_point_out():
    """Dipping under epsilon is not enough; it has to stay under."""
    assert Decay(layer=0, distances=(0.5, 0.01, 0.9, 0.01)).settles_at(0.1) == 3


def test_epsilon_zero_needs_an_exact_match():
    assert Decay(layer=0, distances=(0.0, 0.0)).settles_at(0.0) == 0
    assert Decay(layer=0, distances=(0.0, 1e-9)).settles_at(0.0) is None


def test_a_negative_epsilon_is_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        Decay(layer=0, distances=(0.1,)).settles_at(-0.1)


def test_peak_reports_the_largest_gap():
    assert Decay(layer=0, distances=(0.1, 0.9, 0.2)).peak == 0.9
    assert Decay(layer=0, distances=()).peak == 0.0


# --- decay -----------------------------------------------------------------


def test_identical_caches_show_no_gap():
    curves = decay([flat(6)], [flat(6)], cut_at=2)
    assert curves[0].distances == (0.0,) * 4


def test_the_gap_is_measured_from_the_cut_onward():
    curves = decay([layer([0, 0, 0, 5])], [flat(4)], cut_at=2)
    assert len(curves[0].distances) == 2
    assert curves[0].distances[0] == 0.0
    assert curves[0].distances[1] > 0


def test_every_layer_gets_its_own_curve():
    curves = decay([flat(4), layer([0, 0, 0, 5])], [flat(4), flat(4)], cut_at=0)
    assert [c.layer for c in curves] == [0, 1]
    assert curves[0].peak == 0.0
    assert curves[1].peak > 0


def test_keys_can_be_compared_instead_of_values():
    spliced = Layer(keys=torch.zeros(1, 1, 3, 4), values=torch.ones(1, 1, 3, 4))
    honest = flat(3)
    assert decay([spliced], [honest], 0, use_values=True)[0].peak == 0.0
    assert decay([spliced], [honest], 0, use_values=False)[0].peak > 0


def test_mismatched_stacks_are_rejected():
    with pytest.raises(ValueError, match="2 spliced layers, 1 honest"):
        decay([flat(4), flat(4)], [flat(4)], cut_at=0)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="layer 0: 4 spliced entries, 6 honest"):
        decay([flat(4)], [flat(6)], cut_at=0)


def test_an_empty_stack_is_rejected():
    with pytest.raises(ValueError, match="no layers"):
        decay([], [], cut_at=0)


def test_a_cut_past_the_end_is_rejected():
    with pytest.raises(ValueError, match="cut at 9 but the cache holds 4"):
        decay([flat(4)], [flat(4)], cut_at=9)


# --- shadow_length ---------------------------------------------------------


def test_the_shadow_runs_until_the_slowest_layer_settles():
    curves = [
        Decay(layer=0, distances=(0.5, 0.01, 0.01)),
        Decay(layer=1, distances=(0.5, 0.5, 0.01)),
    ]
    assert shadow_length(curves, 0.1) == 2


def test_a_layer_that_never_settles_means_no_stopping_point():
    curves = [
        Decay(layer=0, distances=(0.01, 0.01)),
        Decay(layer=1, distances=(0.9, 0.9)),
    ]
    assert shadow_length(curves, 0.1) is None


def test_a_looser_epsilon_never_lengthens_the_shadow():
    curves = [Decay(layer=0, distances=(0.5, 0.2, 0.05, 0.005))]
    assert shadow_length(curves, 0.1) <= shadow_length(curves, 0.01)
