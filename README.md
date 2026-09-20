# kv-splice-residue

Does cutting a span out of a KV cache actually forget it?

Leyline (2606.01065) deletes the span's cache entries and rotates everything
after it, rather than re-prefilling. They call the result "positional, not
informational", meaning the surviving entries were computed while the span was
still there and nobody goes back to fix them. If that matters, the span is gone
from the prompt but still contributing to output.

## Setup

Moonlight-16B-A3B-Instruct (`deepseek_v3`, MLA), with DeepSeek-V2-Lite as a
control. Every probe runs three ways: `full` keeps the span, `reprefill` drops
it and prefills again from scratch, and `spliced` cuts it from the cache and
re-anchors what follows.

Layer 0 is built from token embeddings alone, so the spliced and reprefilled
caches have to match there exactly. They do, at `layer0_error = 0.0e+00` on
every run.

## Results

**Which one does the splice act like?** Leyline reports 0/17. Here it was 7 of 14, about 5x closer to re-prefill in KL (0.29 vs 1.64 mean).

**Does the gap fade with distance?** No. Over 1616 downstream tokens it never
drops below any epsilon we tried, including 0.1. Peaks near 1.4 rel-L2, which is
what two unrelated vectors give you.

| probe | fact worth (nats) | recovered |
|---|---|---|
| database-host | 2.03 | -0.03 |
| error-code | 5.59 | 0.12 |
| pinned-version | 1.62 | -0.02 |
| assigned-reviewer | 2.78 | 0.17 |
| ticket-number | 4.53 | 0.08 |

`recovered = (spliced - reprefill) / (full - reprefill)`. **Mean 0.06**, two
negative. The splice forgets about as well as a re-prefill.

## Resources

| | |
|---|---|
| Leyline (2606.01065) | reactive splice, approximate or falls back to re-prefill |
| Continuum (2511.02230) | TTL retention of caches that already exist |
| An Internet for the KV Cache (2608.01526) | CDN-style placement and prefetch |
| CXL-SpecKV (2512.11920) | predicted prefetch across a memory tier, read-only |

298 tests, every decision point mutation-checked.
