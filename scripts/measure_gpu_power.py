"""Measure the GPU's power draw during the course's two kinds of GPU work, for Lecture 28.

Run once, on an otherwise idle GPU, from the repository root::

    pixi run -e aml-env python scripts/measure_gpu_power.py

Writes ``data/derived/gpu_power/<gpu>_<date>.csv`` (one row per sample: seconds from the start,
watts, utilization, memory, phase) and a ``.json`` with the GPU, the driver, the power limit, and
each phase's start, end, and throughput. The figure script of Lecture 28 reads these files; the
measurement is data, so that the lecture's numbers do not change from run to run. The script
refuses to overwrite a record of the same GPU and date: move it aside first.

Phases (``nvidia-smi`` samples every 0.5 s throughout):

- idle, 30 s;
- training, 120 s: ResNet-18 (no pretrained weights) on synthetic 224 x 224 images, batches of 64,
  SGD, float32 with TF32 off for matrix products and convolutions (the course's figure scripts
  differ: L16, L18, L20, and L22 leave cuDNN's TF32 on);
- idle, 30 s;
- inference, 120 s: Qwen3.5-2B in bfloat16, greedy generation, at Lecture 25's pinned revision
  (loaded before the phase starts, from the local cache only);
- idle, 30 s.

The synthetic images are enough: the arithmetic, not the pixels, sets the power.
"""

from __future__ import annotations

import os

# Read once, when huggingface_hub is first imported (by transformers, below): the model comes
# from the local cache only, and no usage reports are sent.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import json
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import torch
import torchvision
import transformers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "gpu_power"
IDLE_S, WORK_S, SAMPLE_MS = 30, 120, 500
BATCH = 64
QWEN = ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc")  # Lecture 25's revision
PROMPT = (
    "Explain, step by step, how a pressure reducing valve sets the pressure in a water network."
)


def gpu_info() -> dict[str, str | float]:
    """Name, driver, and power limit of GPU 0, from ``nvidia-smi``.

    Returns:
        ``name``, ``driver``, and ``power_limit_w``.

    Raises:
        subprocess.CalledProcessError: If ``nvidia-smi`` fails.
    """
    q = "name,driver_version,power.limit"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    name, driver, limit = (s.strip() for s in out.split(","))
    return {"name": name, "driver": driver, "power_limit_w": float(limit)}


def idle(seconds: float) -> dict[str, float]:
    """Do nothing on the GPU for ``seconds``.

    Args:
        seconds: How long to wait.

    Returns:
        An empty record (the phase has no throughput).
    """
    time.sleep(seconds)
    return {}


def training(seconds: float) -> dict[str, float]:
    """ResNet-18 forward and backward passes on synthetic images until ``seconds`` pass.

    Args:
        seconds: How long to train.

    Returns:
        ``images_per_s`` and ``steps``.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(28)
    model = torchvision.models.resnet18(weights=None).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(BATCH, 3, 224, 224, device="cuda")
    y = torch.randint(0, 1000, (BATCH,), device="cuda")
    steps, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        if steps % 10 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    del model, opt, x, y
    torch.cuda.empty_cache()
    return {"images_per_s": BATCH * steps / elapsed, "steps": steps}


def load_qwen() -> tuple[transformers.PreTrainedTokenizerBase, transformers.PreTrainedModel]:
    """Qwen3.5-2B in bfloat16 on the GPU, from the local Hugging Face cache only.

    Returns:
        The tokenizer and the model, in evaluation mode.

    Raises:
        OSError: If the pinned revision is not in the local cache.
    """
    model_id, revision = QWEN
    tok = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, local_files_only=True
    )
    return tok, model.to("cuda").eval()


def inference(
    seconds: float,
    tok: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
) -> dict[str, float]:
    """Greedy generation, 256 new tokens at a time, until ``seconds`` pass.

    Args:
        seconds: How long to generate.
        tok: The tokenizer.
        model: The model, on the GPU.

    Returns:
        ``tokens_per_s`` and ``tokens``.
    """
    msgs = [{"role": "user", "content": PROMPT}]
    enc = tok.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    ids = enc["input_ids"].to("cuda")
    tokens, t0 = 0, time.time()
    with torch.no_grad():
        while time.time() - t0 < seconds:
            out = model.generate(ids, max_new_tokens=256, do_sample=False)
            tokens += out.shape[1] - ids.shape[1]
    torch.cuda.synchronize()
    return {"tokens_per_s": tokens / (time.time() - t0), "tokens": tokens}


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a ``.part`` file and a rename.

    Args:
        path: The file to write.
        text: Its contents.
    """
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    part.replace(path)


def main() -> None:
    """Sample the GPU's power through the five phases and write the CSV and its metadata.

    Raises:
        FileExistsError: If a record of this GPU and date exists (Lecture 28 reads it by name).
    """
    OUT.mkdir(parents=True, exist_ok=True)
    info = gpu_info()
    slug = (str(info["name"]).lower().replace("nvidia", "").replace("geforce", "").strip()).replace(
        " ", ""
    )
    stem = OUT / f"{slug}_{datetime.now(UTC):%Y-%m-%d}"
    targets = [stem.with_suffix(".csv"), stem.with_suffix(".json")]
    if any(p.exists() for p in targets):
        raise FileExistsError(f"{stem}.csv or .json exists: move it aside to measure again")
    raw = stem.with_suffix(".nvsmi.txt")
    query = "--query-gpu=timestamp,power.draw,utilization.gpu,memory.used"
    with raw.open("w") as fh:
        sampler = subprocess.Popen(
            ["nvidia-smi", query, "--format=csv,noheader,nounits", f"-lms={SAMPLE_MS}", "-i", "0"],
            stdout=fh,
        )
        phases = []
        t_start = time.time()

        def run(name: str, fn: Callable[..., dict[str, float]], *args: object) -> None:
            t0 = time.time()
            extra = fn(*args)
            phases.append(
                {"phase": name, "start_s": t0 - t_start, "end_s": time.time() - t_start, **extra}
            )

        try:
            run("idle", idle, IDLE_S)
            run("training", training, WORK_S)
            run("idle", idle, IDLE_S)
            tok, model = load_qwen()
            run("idle", idle, IDLE_S)
            run("inference", inference, WORK_S, tok, model)
            del model
            torch.cuda.empty_cache()
            run("idle", idle, IDLE_S)
        finally:
            sampler.terminate()
            sampler.wait(timeout=10)
    rows = []
    for line in raw.read_text().splitlines():
        stamp, watts, util, mem = (s.strip() for s in line.split(","))
        t = datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").astimezone().timestamp() - t_start
        phase = next((p["phase"] for p in phases if p["start_s"] <= t <= p["end_s"]), "between")
        rows.append(f"{t:.3f},{float(watts):.2f},{int(util)},{int(mem)},{phase}")
    write_atomic(targets[0], "t_s,watts,util_pct,mem_mib,phase\n" + "\n".join(rows) + "\n")
    meta = {
        **info,
        "sample_ms": SAMPLE_MS,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "qwen": {"model": QWEN[0], "revision": QWEN[1]},
        "batch": BATCH,
        "phases": phases,
        "date_utc": f"{datetime.now(UTC):%Y-%m-%d}",
    }
    write_atomic(targets[1], json.dumps(meta, indent=2) + "\n")
    raw.unlink()
    print(f"wrote {targets[0]} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
