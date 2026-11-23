"""Fail when an estimator is built without the parameter that defines its method.

A library default can differ from the method a
slide names.

scikit-learn's ``RandomForestRegressor()`` tries every input at every split (``max_features=1.0``):
that is bagging, not a random forest; ``LinearSVC()`` minimizes the squared hinge, not the hinge;
``HistGradientBoosting*()`` turns early stopping on by itself above 10,000 rows
(``early_stopping="auto"``), so the number of trees depends on the data's size; PyTorch's
``AdamW()`` decays weights by 0.01 unless told otherwise; scikit-learn's ``LogisticRegression()``
is penalized (L2, ``C=1.0``), so a slide's "logistic regression" may be ridge-penalized without
saying so. Each call must state the parameter, so the code says which method it runs.

A call that passes ``**name`` is checked against the keys of the dict literal (or ``dict(...)``
call) assigned to ``name`` in the same file, as marimo notebooks do with a settings cell; if
the file assigns no such dict, the call is trusted.

Each file named is checked together with the local modules it imports (``import ltown`` finds
``ltown.py`` next to the file), so a figure script's data modules are checked with it.

Usage: python tools/check_estimators.py FILE.py [FILE.py ...]   (exit 1 on a finding)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Constructor name -> keyword arguments that define the method it runs.
REQUIRED: dict[str, tuple[str, ...]] = {
    "RandomForestRegressor": ("max_features",),
    "RandomForestClassifier": ("max_features",),
    "ExtraTreesRegressor": ("max_features",),
    "ExtraTreesClassifier": ("max_features",),
    "LinearSVC": ("loss",),
    "HistGradientBoostingRegressor": ("early_stopping",),
    "HistGradientBoostingClassifier": ("early_stopping",),
    "AdamW": ("weight_decay",),
    "LogisticRegression": ("C",),  # keyword-only in scikit-learn 1.8
}
# Parameters that may also be given by position: name -> {parameter: index}.
POSITION: dict[str, dict[str, int]] = {
    "LinearSVC": {"loss": 1},
    "AdamW": {"weight_decay": 4},  # AdamW(params, lr, betas, eps, weight_decay, ...)
}


def dict_keys(tree: ast.Module) -> dict[str, set[str] | None]:
    """Keys of the dicts assigned to each name: a literal ``{"k": v}`` or ``dict(k=v)``.

    Args:
        tree: The parsed file.

    Returns:
        Name -> the keys every such assignment gives (their intersection), or None when one
        of them has keys that are not plain strings (``**other``, a computed key).
    """
    out: dict[str, set[str] | None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        keys: set[str] | None
        if isinstance(value, ast.Dict):
            strings = [
                k.value
                for k in value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            ]
            keys = set(strings) if len(strings) == len(value.keys) else None
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "dict"
            and not value.args
        ):
            named = [k.arg for k in value.keywords]
            keys = {k for k in named if k} if None not in named else None
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                prev = out.get(t.id, keys)
                out[t.id] = None if prev is None or keys is None else prev & keys
    return out


def given(node: ast.Call, dicts: dict[str, set[str] | None]) -> set[str] | None:
    """The keyword arguments a call states, or None when a ``**`` mapping cannot be read."""
    out = set()
    for k in node.keywords:
        if k.arg is not None:
            out.add(k.arg)
        elif isinstance(k.value, ast.Dict) and all(
            isinstance(key, ast.Constant) and isinstance(key.value, str) for key in k.value.keys
        ):
            out |= {
                key.value
                for key in k.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
        elif isinstance(k.value, ast.Name) and dicts.get(k.value.id) is not None:
            out |= dicts[k.value.id] or set()
        else:
            return None  # **kwargs from elsewhere: the parameters may be there; trust the caller
    return out


def findings(path: Path) -> list[str]:
    """Calls in ``path`` that leave a defining parameter to the library default."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dicts = dict_keys(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in REQUIRED:
            continue
        stated = given(node, dicts)
        if stated is None:
            continue
        stated |= {p for p, i in POSITION.get(name, {}).items() if i < len(node.args)}
        missing = [k for k in REQUIRED[name] if k not in stated]
        if missing:
            out.append(f"{path}:{node.lineno}: {name}(...) without {', '.join(missing)}")
    return out


def with_local_imports(path: Path) -> list[Path]:
    """``path`` and every module it imports, directly or not, that sits in its directory."""
    out: list[Path] = []
    todo = [path]
    while todo:
        p = todo.pop(0)
        if p in out:
            continue
        out.append(p)
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"), filename=str(p))):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            todo += [m for m in (path.parent / f"{n}.py" for n in names) if m.exists()]
    return out


def main() -> None:
    """Check every file named on the command line, with the local modules it imports.

    A file that does not parse is a finding too (exit 1), named with the parser's message.
    """
    files: list[Path] = []
    problems: list[str] = []
    for arg in sys.argv[1:]:
        try:
            files += [f for f in with_local_imports(Path(arg)) if f not in files]
        except SyntaxError as e:
            problems.append(f"{e.filename}:{e.lineno}: cannot parse ({e.msg})")
    for f in files:
        problems += findings(f)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
