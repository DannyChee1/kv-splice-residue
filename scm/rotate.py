"""RoPE composes, so re-anchoring a cached key is a single rotation by delta.

Layout is an explicit argument because picking the wrong one fails silently.
"""

from __future__ import annotations

import torch

INTERLEAVED = "interleaved"
HALF_SPLIT = "half_split"
LAYOUTS = (INTERLEAVED, HALF_SPLIT)


def _check(dim: int, layout: str) -> None:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if dim % 2:
        raise ValueError(f"rotary dim must be even, got {dim}")


def inv_freq(dim: int, base: float = 10000.0, device=None) -> torch.Tensor:
    """Angular frequency per rotated pair."""
    steps = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    return 1.0 / (base ** (steps / dim))


def _pairs(x: torch.Tensor, layout: str) -> tuple[torch.Tensor, torch.Tensor]:
    if layout == INTERLEAVED:
        return x[..., 0::2], x[..., 1::2]
    half = x.shape[-1] // 2
    return x[..., :half], x[..., half:]


def _unpair(even: torch.Tensor, odd: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == HALF_SPLIT:
        return torch.cat((even, odd), dim=-1)
    out = torch.stack((even, odd), dim=-1)
    return out.flatten(-2)


def _spin(x: torch.Tensor, angle: torch.Tensor, layout: str) -> torch.Tensor:
    """Rotate each pair by its own angle, in float32 whatever came in."""
    dtype = x.dtype
    even, odd = _pairs(x.to(torch.float32), layout)
    cos, sin = torch.cos(angle), torch.sin(angle)
    return _unpair(even * cos - odd * sin, even * sin + odd * cos, layout).to(dtype)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    layout: str = INTERLEAVED,
    base: float = 10000.0,
) -> torch.Tensor:
    """Place keys at absolute positions, the way a prefill would. Reference only."""
    _check(x.shape[-1], layout)
    if positions.shape[0] != x.shape[-2]:
        raise ValueError(
            f"got {positions.shape[0]} positions for {x.shape[-2]} keys"
        )
    angle = positions.to(torch.float32)[:, None] * inv_freq(
        x.shape[-1], base, x.device
    )
    return _spin(x, angle, layout)


def rotate(
    x: torch.Tensor,
    delta: int,
    layout: str = INTERLEAVED,
    base: float = 10000.0,
    freqs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Real models need `freqs`: YaRN and friends bend the ladder away from
    `1 / base ** (2i/d)`, so rebuilding it from a base rotates by the wrong angle.
    """
    _check(x.shape[-1], layout)
    if freqs is None:
        freqs = inv_freq(x.shape[-1], base, x.device)
    elif freqs.shape[-1] != x.shape[-1] // 2:
        raise ValueError(
            f"got {freqs.shape[-1]} frequencies for a {x.shape[-1]}-wide key"
        )
    return _spin(x, float(delta) * freqs.to(x.device), layout)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Leyline reports rel-L2, so we do too."""
    scale = torch.linalg.vector_norm(expected.to(torch.float32))
    if scale == 0:
        raise ValueError("expected tensor is all zeros, rel-L2 undefined")
    diff = torch.linalg.vector_norm((actual - expected).to(torch.float32))
    return (diff / scale).item()
