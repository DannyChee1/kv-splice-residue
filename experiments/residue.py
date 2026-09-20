"""Gate 1: does the leftover from a positional splice change what the model says?

Runs every probe at several padding depths, three paths each, and reports which
reference the spliced cache tracked. The number that matters is how often it
tracked `reprefill`: Leyline reports ~0 on Moonlight, and SCM should be ~1 by
construction, since it recomputes under the edited context.

    python -m experiments.residue --model <hf-name> --out results/residue.json

Use --model deepseek-ai/DeepSeek-V2-Lite as the control. Its two references
barely diverge, so almost every trial is uninformative; if that run reports many
informative trials, something is wrong before any result is believable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scm.agreement import summarize
from scm.probes import SUITE, pad, token_length
from scm.runner import layer_zero_error, load, run_probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32",
                    help="fp32 by default; bf16 storage swamps the effect")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--padding", type=int, nargs="+", default=[0, 10, 40])
    ap.add_argument("--layout", default=None)
    ap.add_argument("--rope-dim", type=int, default=None)
    ap.add_argument("--remote-code", action="store_true",
                    help="use the repo's own modeling file; off by default")
    ap.add_argument("--out", type=Path, default=Path("results/residue.json"))
    args = ap.parse_args()

    target = load(args.model, args.dtype, args.device, args.layout,
                  args.rope_dim, args.remote_code)

    rows, trials = [], []
    for depth in args.padding:
        for base in SUITE:
            probe = pad(base, before=depth, after=depth) if depth else base
            trial = run_probe(target, probe, steps=args.steps)
            trials.append(trial)
            rows.append({
                "probe": base.name,
                "harm": base.harm,
                "padding": depth,
                "prompt_tokens": token_length(probe, target.tokenizer),
                "diverged": trial.references_diverge,
                "verdict": trial.verdict,
                "prefix_lengths": trial.prefix_lengths(),
                "kl": trial.divergences(),
                "layer0_error": layer_zero_error(target, probe),
            })
            print(f"{base.name:<26} pad={depth:<4} "
                  f"{'diverged' if trial.references_diverge else 'flat':<9} "
                  f"{trial.verdict}")

    report = summarize(trials)
    payload = {
        "model": args.model,
        "dtype": args.dtype,
        "steps": args.steps,
        "trials": report.trials,
        "informative": report.informative,
        "verdicts": report.verdicts,
        "mean_prefix": report.mean_prefix,
        "tracks_reprefill": report.tracks_reprefill,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"\ninformative {report.informative}/{report.trials}"
          f"   verdicts {report.verdicts}"
          f"   tracks_reprefill {report.tracks_reprefill:.2f}")
    worst = max((r["layer0_error"] for r in rows), default=0.0)
    if worst > 1e-3:
        print(f"WARNING layer-0 error reached {worst:.2e}; the splice is off, "
              "so treat every verdict above as unproven")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
