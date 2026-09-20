"""Gate 3: how far past a cut does its influence reach?

Reports how long a shadow would have to run per epsilon, which is the cost of
being exact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scm.divergence import compare, decay, shadow_length
from scm.probes import SUITE, pad, token_span
from scm.runner import load, prefill, read_cache
from scm.splice import splice_cache

EPSILONS = (0.0, 0.001, 0.01, 0.05, 0.1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--padding", type=int, default=40)
    ap.add_argument("--layout", default=None)
    ap.add_argument("--rope-dim", type=int, default=None)
    ap.add_argument("--remote-code", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/decay.json"))
    args = ap.parse_args()

    target = load(args.model, args.dtype, args.device, args.layout,
                  args.rope_dim, args.remote_code)

    rows = []
    for base in SUITE:
        probe = pad(base, before=args.padding, after=args.padding)
        start, end = token_span(probe, target.tokenizer)
        spliced = splice_cache(
            read_cache(prefill(target, target.tokenizer.encode(probe.full))),
            start, end, target.layout, target.rope_dim, target.base,
            target.mla, target.freqs,
        )
        honest = read_cache(prefill(target, target.tokenizer.encode(probe.edited)))

        # Look at whichever tensor the rotation never reaches, since that is
        # where anything left over has to live. On MLA that is the keys slot,
        # which holds the position-free latent; elsewhere it is the values.
        curves = decay(spliced, honest, cut_at=start, use_values=not target.mla)

        # Baseline: how far apart are the unedited and honestly edited caches at
        # the same content? If this is as large as the splice's own gap, the
        # architecture scatters under any perturbation and the splice is not
        # what destroyed the information.
        full_cache = read_cache(prefill(target, target.tokenizer.encode(probe.full)))
        plain = compare(full_cache, honest, end, start, use_values=not target.mla)

        row = {
            "probe": base.name,
            "measured": "latent" if target.mla else "values",
            "downstream_tokens": len(curves[0].distances),
            "peak_by_layer": [c.peak for c in curves],
            "baseline_by_layer": [c.peak for c in plain],
            "shadow_length": {
                str(eps): shadow_length(curves, eps) for eps in EPSILONS
            },
        }
        rows.append(row)
        splice_peak = max(row["peak_by_layer"])
        base_peak = max(row["baseline_by_layer"])
        print(f"{base.name:<26} splice={splice_peak:.3f}  deletion-alone={base_peak:.3f}"
              f"  ratio={splice_peak / base_peak if base_peak else float('inf'):.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": args.model, "epsilons": list(EPSILONS), "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
