"""Gate 2: how much prefill does a real agent run actually repeat?

Reads collected traces, simulates a prefix cache with no capacity limit, and
reports the share of computed tokens that were repeats. Unlimited capacity makes
the answer a floor: a real engine evicts and shares the GPU, which only pushes
the number up.

    python -m experiments.opportunity --traces data/traces --tokenizer <hf-name>

No GPU needed. The tokenizer is only used to count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scm.prefixsim import simulate, summarize
from scm.traces import iter_traces


SHAPE = ("system", "user", "assistant", "tool", "tool", "assistant")


def plain(messages):
    return "".join(f"<|{m.role}|>{m.content}\n" for m in messages)


def chat_renderer(tokenizer):
    """Pick one renderer for the whole run, and say which was picked.

    Most chat templates refuse the shape agent traces actually have, since roles
    never alternate. Choosing per call would mix two renderings in one run and
    invent cache misses, so the choice is made once here and anything that fails
    afterwards is a real error, not something to paper over.
    """
    probe = [{"role": role, "content": "x"} for role in SHAPE]
    try:
        tokenizer.apply_chat_template(probe, tokenize=False)
    except Exception as exc:
        print(f"note: {type(exc).__name__} from the chat template on an agent-shaped "
              f"conversation, so counting with a generic one instead")
        return plain

    def render(messages):
        payload = [{"role": m.role, "content": m.content} for m in messages]
        return tokenizer.apply_chat_template(payload, tokenize=False)

    return render


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, default=Path("data/traces"))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--block-size", type=int, default=1,
                    help="1 is the friendliest cache; 16 matches vLLM")
    ap.add_argument("--out", type=Path, default=Path("results/opportunity.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    render = chat_renderer(tokenizer)

    stats, rows = [], []
    for trace in iter_traces(args.traces):
        result = simulate(trace, render, tokenizer, block_size=args.block_size)
        stats.append(result)
        rows.append({
            "program_id": trace.program_id,
            "turns": len(trace.turns),
            "prompt_tokens": result.prompt_tokens,
            "computed_tokens": result.computed_tokens,
            "reused_tokens": result.reused_tokens,
            "reprefill_share": result.reprefill_share,
            "reused_by_decider": result.reused_by_decider(),
        })
        print(f"{trace.program_id:<34} turns={len(trace.turns):<4} "
              f"reprefill={result.reprefill_share:.1%}")

    if not stats:
        raise SystemExit(f"no traces in {args.traces}")

    overall = summarize(stats)
    overall["block_size"] = args.block_size
    overall["tokenizer"] = args.tokenizer
    overall["rows"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(overall, indent=2))

    print(f"\nreprefill share {overall['reprefill_share']:.1%} "
          f"of {overall['computed_tokens']:,} computed tokens")
    print(f"harness-attributable {overall['harness_attributable']:,} tokens")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
