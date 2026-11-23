"""The command line, seeding, data paths, and output writers shared by the figure scripts.

Every figure script writes ``numbers.json`` (each number its slides quote) with
``write_numbers``, and the LaTeX macros its TikZ diagrams read, ``numbers.tex``, with
``write_tex_macros``. Datasets and simulators used by more than one lecture have their own
modules: ``concrete.py``, ``bridges.py``, ``buildings.py``, ``ltown.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np

DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data"
MINUS: str = "\N{MINUS SIGN}"  # typeset minus for slides and diagrams; "-" is a hyphen


# ---------------------------------------------------------------------------------------------
# Command line, seeding, and inputs


def parse_out_dir(description: str) -> Path:
    """Parse the ``--out`` directory argument shared by all figure scripts and create it.

    Args:
        description: Help text for the script.

    Returns:
        The output directory (it exists on return).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory for the PNG figures, numbers.json, numbers.tex",
    )
    out: Path = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    return out


def seed_all(seed: int) -> None:
    """Seed Python's, NumPy's, and, if the script imported it, PyTorch's global generators.

    Scripts also seed their own ``np.random.default_rng`` and scikit-learn ``random_state``; the
    global seeds cover any library that draws from the global generators. String hashing is
    fixed only when ``PYTHONHASHSEED`` is set before Python starts, as the Makefile does; a
    script run by hand without it gets a warning.

    Args:
        seed: The seed for every generator.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (the legacy global generator other libraries draw from)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)  # seeds the CPU and every CUDA device
    if os.environ.get("PYTHONHASHSEED") != "0":
        warnings.warn(
            "PYTHONHASHSEED is not 0: run the script through the Makefile, or set it before "
            "Python starts, for reproducible string hashing",
            stacklevel=2,
        )


def require(path: Path, fetch_script: str) -> Path:
    """Return ``path`` if it exists; otherwise say which script downloads it.

    Args:
        path: A file under ``data/``.
        fetch_script: The script that writes it, e.g. ``"fetch_nbi.sh"``.

    Returns:
        ``path``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `bash scripts/{fetch_script}`")
    return path


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square difference of two arrays of the same shape.

    Args:
        a: Predictions (or any array).
        b: Targets, the same shape as ``a``.

    Returns:
        sqrt(mean((a - b)^2)).

    Raises:
        ValueError: If the shapes differ: (n,) against (n, 1) would otherwise broadcast to an
            n x n difference and return a meaningless number.
    """
    if np.shape(a) != np.shape(b):
        raise ValueError(f"rmse of arrays of different shapes: {np.shape(a)} and {np.shape(b)}")
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ---------------------------------------------------------------------------------------------
# Rounding and outputs


def rnd(v: float, nd: int) -> float:
    """Round a number for ``numbers.json`` once, where it is produced.

    Args:
        v: The value (a Python or NumPy number).
        nd: Decimal places.

    Returns:
        A Python float; ``-0.0`` becomes ``0.0`` so no slide shows a signed zero.
    """
    return round(float(v), nd) + 0.0


def pct(v: float, nd: int = 1) -> float:
    """A share as a percentage, rounded.

    Args:
        v: A share in [0, 1].
        nd: Decimal places of the percentage.

    Returns:
        100 v, rounded to ``nd`` places.
    """
    return rnd(100 * float(v), nd)


def jsonable(obj: object) -> object:
    """Convert a nested result to plain Python for ``json``.

    Keys starting with ``_`` are private (kept for later computations, not quoted) and are
    dropped; NumPy scalars become Python numbers; tuples become lists; ``-0.0`` becomes ``0.0``.

    Args:
        obj: Nested dicts, lists, tuples, and scalars.

    Returns:
        The same structure in built-in types.

    Raises:
        TypeError: On a NumPy array. Round and convert it where the number is produced, so the
            precision a slide quotes is visible there.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        raise TypeError(f"array of shape {obj.shape} in numbers.json; convert it at its source")
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj + 0.0  # -0.0 + 0.0 is 0.0; every other float is unchanged
    return obj


def write_numbers(numbers: Mapping[str, object], out_dir: Path) -> Path:
    """Write ``numbers.json``: every number the lecture's slides quote.

    Args:
        numbers: The lecture's results (see ``jsonable`` for what is allowed).
        out_dir: The figure directory.

    Returns:
        The path written.

    Raises:
        ValueError: If a number is NaN or infinite (JSON has no such values).
    """
    path = out_dir / "numbers.json"
    text = json.dumps(jsonable(dict(numbers)), indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def tex_num(x: float, spec: str = "") -> str:
    """Format a number for a slide or a diagram, with a typeset minus sign.

    Args:
        x: The value, already at full precision (round only here, once).
        spec: A format spec, e.g. ``".2f"``, ``"+.1f"``, ``",.0f"``.

    Returns:
        The formatted number; a negative one starts with U+2212.
    """
    return format(x, spec).replace("-", MINUS)


def write_tex_macros(macros: Mapping[str, str], out_dir: Path, source: str) -> Path:
    r"""Write ``numbers.tex``: one ``\newcommand`` per macro, for the lecture's diagrams.

    Values are written as given: format numbers with ``tex_num``. Hyphens in text values
    stay hyphens (a road named I-55, a range 1950-1979).

    Args:
        macros: Macro name (without the backslash) to its LaTeX text.
        out_dir: The figure directory.
        source: The figure script's file name, for the header comment.

    Returns:
        The path written.
    """
    lines = [f"% Generated by figures/{source}. Do not edit."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    path = out_dir / "numbers.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
