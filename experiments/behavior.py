"""Does the leftover change what the model says, or only the cache?

`recoverable` near 0 means the splice recalls no better than an honest
reprefill; near 1 means it still knows a fact whose tokens are gone.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scm.recall import SUITE
from scm.runner import load, run_recall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layout", default=None)
    ap.add_argument("--rope-dim", type=int, default=None)
    ap.add_argument("--remote-code", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/behavior.json"))
    args = ap.parse_args()

    target = load(args.model, args.dtype, args.device, args.layout,
                  args.rope_dim, args.remote_code)

    rows = []
    print(f"{'probe':<20}{'full':>9}{'reprefill':>11}{'spliced':>9}"
          f"{'helped':>9}{'recovered':>11}{'L0':>10}")
    for probe in SUITE:
        row = run_recall(target, probe)
        rows.append(row)
        rec = row["recoverable"]
        shown = "n/a" if rec is None else f"{rec:.2f}"
        print(f"{row['probe']:<20}{row['full']:>9.3f}{row['reprefill']:>11.3f}"
              f"{row['spliced']:>9.3f}{row['fact_helped']:>9.3f}{shown:>11}"
              f"{row['layer0_error']:>10.1e}")

    usable = [r for r in rows if r["recoverable"] is not None]
    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "probes": len(rows),
        "usable": len(usable),
        "mean_recoverable": statistics.mean(r["recoverable"] for r in usable)
        if usable else None,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    if not usable:
        print("\nthe fact never helped, so the probes cannot answer this")
    else:
        mean = summary["mean_recoverable"]
        print(f"\nmean recovered {mean:.2f} over {len(usable)}/{len(rows)} probes")
        if mean < 0.15:
            print("   the splice recalls no better than an honest reprefill: the "
                  "leftover does not reach behaviour")
        elif mean > 0.5:
            print("   the splice still knows a fact whose tokens are gone")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
