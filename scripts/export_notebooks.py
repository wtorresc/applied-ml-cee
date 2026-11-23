"""Export every marimo notebook in ``notebooks/`` to a Jupyter ``.ipynb`` in ``ipynb/``.

Run via ``pixi run -e aml-env export-ipynb``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"
EXPORT_DIR = ROOT / "ipynb"
# marimo from this interpreter, not whichever `marimo` is first on PATH.
MARIMO = [sys.executable, "-m", "marimo"]
ESTIMATORS = ROOT / "lectures" / "tools" / "check_estimators.py"

PYTHON3_KERNELSPEC = {
    "display_name": "Python 3 (aml-env)",
    "language": "python",
    "name": "python3",
}


def _patch_kernelspec(ipynb_path: Path) -> None:
    """Inject a ``python3`` kernelspec so the notebook opens cleanly in Jupyter.

    marimo's exporter leaves ``metadata.kernelspec`` empty, which makes Jupyter
    prompt the recipient to pick a kernel. Setting it to the standard ``python3``
    name avoids that. The notebook is read and written as UTF-8 and its characters
    (−, ×, Greek) are written as they are, not as ASCII escape sequences.

    Args:
        ipynb_path: The exported notebook, patched in place.
    """
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    nb.setdefault("metadata", {})["kernelspec"] = PYTHON3_KERNELSPEC
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    ipynb_path.write_text(text, encoding="utf-8")


def check_one(src: Path) -> bool:
    """Run ``marimo check`` and the estimator check; return False on any finding.

    The export pipeline (``marimo export ipynb``) is more permissive than the
    interactive editor's static check — it will happily emit ``.ipynb`` files
    from notebooks that have duplicate-name errors and would break in the
    editor. Run this *before* export so the same notebooks that build also
    open cleanly.
    ``lectures/tools/check_estimators.py`` then fails a notebook that builds an
    estimator without the parameter defining its method, as the
    Makefile does for the figure scripts.

    Args:
        src: Path to the marimo ``.py`` notebook.

    Returns:
        True if both checks passed.
    """
    checks = (
        ("CHECK FAIL", [*MARIMO, "check", str(src)]),
        ("ESTIMATORS", [sys.executable, str(ESTIMATORS), str(src)]),
    )
    for label, cmd in checks:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        # Exit code 1 means a finding (or a crash of the checker); marimo's warnings exit 0.
        if result.returncode != 0:
            print(f"{label:<10}  {src.name}", file=sys.stderr)
            for line in (result.stdout + result.stderr).splitlines():
                print(f"            {line}", file=sys.stderr)
            return False
    return True


def export_one(src: Path, dst: Path) -> bool:
    """Check then export a single marimo notebook to ``.ipynb``.

    Args:
        src: Path to the marimo ``.py`` notebook.
        dst: Destination path for the generated ``.ipynb``.

    Returns:
        True on success, False if ``marimo check``, the estimator check
        (``check_one``), or ``marimo export`` reports an error.
    """
    if not check_one(src):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [*MARIMO, "export", "ipynb", str(src), "-o", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"FAIL  {src.name}\n      {result.stderr.strip()}", file=sys.stderr)
        return False
    _patch_kernelspec(dst)
    print(f"OK    {src.name}  ->  {dst.resolve().relative_to(ROOT)}")
    return True


def main() -> int:
    """Check and export every ``cee_*.py`` notebook.

    Returns:
        0 if all succeed (or there is nothing to export), 1 without a ``notebooks/``
        directory, 2 if any check or export failed.
    """
    if not NOTEBOOKS_DIR.exists():
        print(f"No notebooks/ directory at {NOTEBOOKS_DIR}", file=sys.stderr)
        return 1
    sources = sorted(NOTEBOOKS_DIR.glob("cee_*.py"))
    if not sources:
        print("No notebooks to export.")
        return 0
    failed = 0
    for src in sources:
        dst = EXPORT_DIR / f"{src.stem}.ipynb"
        if not export_one(src, dst):
            failed += 1
    if failed:
        print(f"\n{failed} of {len(sources)} export(s) failed.", file=sys.stderr)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
