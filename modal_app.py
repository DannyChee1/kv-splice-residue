"""Run the experiments on Modal.

Weights are fetched on a CPU worker into a Volume, so the GPU never pays for
the download. Run the DeepSeek-V2-Lite control before trusting Moonlight.
"""

import modal

CACHE = "/cache"
OUT = "/results"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # tiktoken/blobfile are for Moonlight's tokenizer; sentencepiece for others.
    .pip_install("torch", "transformers>=5.0", "accelerate", "hf_transfer",
                 "tiktoken", "blobfile", "sentencepiece", "protobuf")
    .env({"HF_HOME": CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("scm", "experiments")
)

app = modal.App("scm-residue", image=image)
weights = modal.Volume.from_name("scm-weights", create_if_missing=True)
results = modal.Volume.from_name("scm-results", create_if_missing=True)

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@app.function(volumes={CACHE: weights}, timeout=3600, cpu=4)
def fetch(model: str) -> str:
    """Pull weights on a CPU worker. Same bytes, a fraction of the price."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(model, ignore_patterns=["*.pth", "*.msgpack", "*.h5"])
    weights.commit()
    return path


@app.function(
    gpu="H100",
    volumes={CACHE: weights, OUT: results},
    timeout=3600,
    memory=65536,
)
def residue(
    model: str,
    padding: list[int],
    steps: int = 128,
    dtype: str = "bfloat16",
) -> dict:
    """The three-path run."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    tag = model.split("/")[-1]
    out = Path(OUT) / f"residue-{tag}.json"
    subprocess.run(
        [sys.executable, "-u", "-m", "experiments.residue",
         "--model", model, "--device", "cuda", "--dtype", dtype,
         "--steps", str(steps), "--out", str(out),
         "--padding", *[str(p) for p in padding]],
        check=True,
    )
    results.commit()
    return json.loads(out.read_text())


@app.function(
    gpu="H100",
    volumes={CACHE: weights, OUT: results},
    timeout=3600,
    memory=65536,
)
def decay(model: str, padding: int = 40, dtype: str = "bfloat16") -> dict:
    """Measure the leftover directly, instead of inferring it from decoded tokens."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    tag = model.split("/")[-1]
    out = Path(OUT) / f"decay-{tag}.json"
    subprocess.run(
        [sys.executable, "-u", "-m", "experiments.decay",
         "--model", model, "--device", "cuda", "--dtype", dtype,
         "--padding", str(padding), "--out", str(out)],
        check=True,
    )
    results.commit()
    return json.loads(out.read_text())


@app.function(
    gpu="H100",
    volumes={CACHE: weights, OUT: results},
    timeout=3600,
    memory=65536,
)
def behavior(model: str, dtype: str = "bfloat16") -> dict:
    """Does the leftover reach behaviour, or only the cache?"""
    import json
    import subprocess
    import sys
    from pathlib import Path

    tag = model.split("/")[-1]
    out = Path(OUT) / f"behavior-{tag}.json"
    subprocess.run(
        [sys.executable, "-u", "-m", "experiments.behavior",
         "--model", model, "--device", "cuda", "--dtype", dtype,
         "--out", str(out)],
        check=True,
    )
    results.commit()
    return json.loads(out.read_text())


@app.function(gpu="H100", volumes={CACHE: weights, OUT: results}, timeout=900)
def smoke() -> dict:
    """Prove the GPU path works on a model that costs nothing to run."""
    return residue.local(TINY, padding=[0, 2], steps=8, dtype="float32")


@app.local_entrypoint()
def main(
    action: str = "smoke",
    model: str = TINY,
    steps: int = 128,
    dtype: str = "bfloat16",
    padding: str = "0,10,40",
):
    depths = [int(p) for p in padding.split(",")]

    if action == "fetch":
        print(f"fetched to {fetch.remote(model)}")
        return

    if action == "behavior":
        report = behavior.remote(model, dtype)
        mean = report["mean_recoverable"]
        print(f"\nmodel {report['model']}")
        print(f"usable probes {report['usable']}/{report['probes']}")
        if mean is None:
            print("the fact never helped, so the probes cannot answer this")
        else:
            print(f"mean recovered {mean:.2f}")
        return

    if action == "decay":
        report = decay.remote(model, depths[0], dtype)
        print(f"\nmodel {report['model']}   epsilons {report['epsilons']}")
        print(f"\n{'probe':<26}{'splice':>9}{'deletion':>10}{'ratio':>8}")
        for row in report["rows"]:
            splice = max(row["peak_by_layer"])
            plain = max(row["baseline_by_layer"])
            ratio = splice / plain if plain else float("inf")
            print(f"{row['probe']:<26}{splice:>9.3f}{plain:>10.3f}{ratio:>8.2f}")

        splices = [max(r["peak_by_layer"]) for r in report["rows"]]
        plains = [max(r["baseline_by_layer"]) for r in report["rows"]]
        worst, base = max(splices), max(plains)
        print(f"\nsplice {worst:.3f}   deletion alone {base:.3f}")
        if base > 0 and worst / base < 1.5:
            print("   an honest deletion scatters the cache just as hard, so the "
                  "splice is not what destroyed anything")
        elif worst > 10 * max(base, 1e-9):
            print("   the splice destroys far more than the deletion does")
        return

    if action == "smoke":
        report = smoke.remote()
    elif action == "residue":
        report = residue.remote(model, depths, steps, dtype)
    else:
        raise SystemExit(f"unknown action {action!r}")

    print(f"\nmodel        {report['model']}")
    print(f"informative  {report['informative']}/{report['trials']}")
    print(f"verdicts     {report['verdicts']}")
    print(f"tracks_rp    {report['tracks_reprefill']:.2f}")
    worst = max((r["layer0_error"] for r in report["rows"]), default=0.0)
    print(f"layer0 worst {worst:.2e}"
          + ("   <- splice is off, verdicts unproven" if worst > 1e-3 else "   ok"))
