"""Gate 3 entry point: measure how fast a deletion's influence fades.

    python -m experiments.decay --model <hf-name> --out results/decay.json

Reports, per epsilon, how many tokens a shadow track would have to run before the
main track's entries are within tolerance. That number is the cost of exactness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scm.divergence import decay, shadow_length
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
    ap.add_argument("--out", type=Path, default=Path("results/decay.json"))
    args = ap.parse_args()

    target = load(args.model, args.dtype, args.device, args.layout, args.rope_dim)

    rows = []
    for base in SUITE:
        probe = pad(base, before=args.padding, after=args.padding)
        start, end = token_span(probe, target.tokenizer)
        spliced = splice_cache(
            read_cache(prefill(target, target.tokenizer.encode(probe.full))),
            start, end, target.layout, target.rope_dim, target.base,
        )
        honest = read_cache(prefill(target, target.tokenizer.encode(probe.edited)))
        curves = decay(spliced, honest, cut_at=start)

        row = {
            "probe": base.name,
            "downstream_tokens": len(curves[0].distances),
            "peak_by_layer": [c.peak for c in curves],
            "shadow_length": {
                str(eps): shadow_length(curves, eps) for eps in EPSILONS
            },
        }
        rows.append(row)
        print(f"{base.name:<26} downstream={row['downstream_tokens']:<5} "
              f"shadow@0.01={row['shadow_length']['0.01']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": args.model, "epsilons": list(EPSILONS), "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
