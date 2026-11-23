"""List every number in a deck that no value in its numbers.json renders to.

Every number on a slide comes from the figure script's ``numbers.json``. The decks
used to be filled from templates with «key» markers; now the decks are their own
source, so this tool checks the link the templates used to carry. A number counts as found when
some leaf of numbers.json, formatted the way slides format numbers (fixed decimals, a share as a
percentage or a value already in percent, thousands separators, scientific), gives the same
text.

**Found is not checked.** The tool finds orphans, not wrong links: on a large numbers.json almost
every short number renders from some leaf (on L21's 1,377 leaves, 98% of the two-decimal values
0.10-0.99 and 98% of the integer percentages do), so a wrong number usually still counts as
found. ``--keys`` prints, for each found number, the keys that render to it: read the key names
against the sentence.

**Re-rounding.** A number explained only by leaves that sit exactly on a rounding tie
at the slide's precision (2.75 shown as "2.8", 0.675 as "68%") may have been rounded twice: the
raw value could round the other way. The summary line counts these; ``--ties`` lists each with
its keys. Write such a slide number from the raw value, or store it at the slide's precision.

Numbers that are not data are skipped, each for a stated reason, so none is hidden: the summary
line counts them per reason and ``--skipped`` lists every one with its line.

- ``small``: years (four digits with no thousands comma, 1900-2100, or 1700-2100 in parentheses,
  "(1807)") and integers up to ``SMALL``, which usually count things on the page ("three rooms",
  "5 seeds"). "2,000" is a number, not a year.
- ``reference``: a lecture, section, part, figure, table, or regulation number ("Lecture 12",
  "23 CFR 655"), and the cells of a table column headed "Lecture". After a plural word, the rest
  of its range or list too ("Lectures 5–22", "Lectures 12 and 14", "Lectures 2, 5, 10"): over a
  comma only a plain integer, and never a percentage, so "in Lecture 5, 57% of tests" checks 57%.
- ``name``: a number that is part of a name: a model ("ResNet-18", "ViT-B/16", "Qwen2.5-1.5B",
  "PRV-3"); a room, junction, or seed, with the rest of its list after a plural ("room 110",
  "rooms 108 and 110", "seeds 0 to 4"); an identifier with a leading zero ("03338780").
- ``formula``: an exponent written after a caret ("h^1.5", "frequency^0.75").
- ``clock``: the two-digit minutes of a time of day ("35" in "15:35"; the hour is still checked).
- ``bibliography``: the References slides (volumes, issues, pages).
- ``constant``: a value that the figure script, or a data module it imports from ``figures/``,
  defines at module level (an UPPER_CASE name other than a seed, also unpacked from a tuple:
  ``A, B = (16, 8)``), as a literal or as the value of its expression ("P_SERVICE = 35 * PSI_M"
  gives 35 and 24.6). A literal matches only as written (0.05 as "0.05" or "5%", not "5.0%"); a
  computed value as slides round it. The style modules (aml_style, common, title_art) are not
  read. This is the one reason that can coincide with a data number (a cited "400 million
  images" where the script has a 400): read the constants that ``--skipped`` lists once per deck.

Code blocks, inline code, and image paths are not read; titles, subtitles, alt text, tables,
and notes are. Use ``--all`` to check every number, with no skips and no script constants.

Usage:
    python tools/deck_numbers.py DECK.md [DECK.md ...] [--skipped] [--keys] [--ties] [--all]
    python tools/deck_numbers.py DECK.md NUMBERS.json [--skipped] [--keys] [--ties] [--all]
(exit 1 if any number of any deck is missing; ties alone do not fail)
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from decimal import Decimal
from pathlib import Path

SMALL = 12  # integers up to this are skipped unless --all (counts like "three rooms", "5 seeds")
NUMBER = re.compile(
    r"(?<![\w.])[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.])[-−]?\d+(?:\.\d+)?(?:e[-−+]?\d+)?%?"
)
REFERENCE = re.compile(
    r"\b(?:Lectures?|lectures?|Sections?|Parts?|Figures?|Tables?|Chapters?|CS|CEE|ECE|CFR|L)\s*$"
    r"|§\s*$"
)
LABEL = re.compile(r"\b(?:[Rr]ooms?|[Jj]unctions?|[Ss]eeds?)\s+$")
# Between two numbers of one range or list: "5–22", "0 to 4", "12 and 14", "108 or 110".
CONTINUES = re.compile(r"^\s*(?:[–—-]|to|,?\s*and|,?\s*or)\s*$")
LIST_COMMA = re.compile(r"^\s*,\s*$")  # "2, 5, 10": only plain integers continue over a comma
CLOCK = re.compile(r"\b\d{1,2}:$")  # the hour of "15:35"; only two-digit minutes follow
YEAR_IN_PARENS = re.compile(r"\((?:1[7-9]\d\d|20\d\d|2100)[a-z]?[),;]")
STYLE_MODULES = {"aml_style", "common", "title_art"}  # no lecture data in them
BINARY: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
HALF = Decimal("0.5")
Chain = tuple[str, bool]  # the reason a number passes on, and whether a comma may continue it
NO_CHAIN: Chain = ("", False)


def keyed_leaves(obj: object, path: str = "") -> list[tuple[str, float]]:
    """Every finite number in a JSON value, with its key path.

    Args:
        obj: A parsed JSON value.
        path: The path of ``obj`` itself ("" at the top), e.g. ``searches.random``.

    Returns:
        (path, value) for each number, e.g. ``("searches.random.regret_max", 2.75)``; list
        elements are written ``key[i]``. Booleans, strings, nulls, and non-finite numbers are
        left out.
    """
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [(path or "(top)", float(obj))] if math.isfinite(obj) else []
    if isinstance(obj, dict):
        return [x for k, v in obj.items() for x in keyed_leaves(v, f"{path}.{k}" if path else k)]
    if isinstance(obj, list):
        return [x for i, v in enumerate(obj) for x in keyed_leaves(v, f"{path}[{i}]")]
    return []


def leaves(obj: object) -> list[float]:
    """Every finite number in a JSON value."""
    return [v for _, v in keyed_leaves(obj)]


def on_tie(stored: Decimal, decimals: int) -> bool:
    """Whether showing ``stored`` at ``decimals`` places rounds away a trailing exact half.

    Args:
        stored: The value as numbers.json stores it, exactly (``Decimal(repr(x))``), already
            scaled to the unit shown (times 100 for a share shown as a percentage).
        decimals: The decimal places the slide shows.

    Returns:
        True for 2.75 at one place or 67.5 at none: the raw value behind such a stored value
        may round either way.
    """
    return abs(stored).scaleb(decimals) % 1 == HALF


def forms(x: float) -> dict[str, bool]:
    """The texts a slide may show for one value, each with whether it shows a rounding tie.

    Args:
        x: One leaf of numbers.json.

    Returns:
        Text -> True when every way this value gives that text re-rounds a tie (``on_tie``);
        fixed-point texts with and without thousands separators, as a share in percent
        (0.168 -> "16.8%"), as a value already in percent (``common.pct``: 16.8 -> "16.8%"), and
        scientific ("3e-04", "3e-4"). A U+2212 minus sign is accepted for "-".
    """
    out: dict[str, bool] = {}

    def add(text: str, tie: bool) -> None:
        for t in {text, text.replace("-", "−")}:
            out[t] = out.get(t, True) and tie

    exact = Decimal(repr(x))
    for d in range(6):
        for v, stored, unit in ((x, exact, ""), (100 * x, 100 * exact, "%"), (x, exact, "%")):
            tie = on_tie(stored, d)
            add(f"{v:.{d}f}{unit}", tie)
            add(f"{v:,.{d}f}{unit}", tie)
    for d in range(3):
        s = f"{x:.{d}e}"
        add(s, False)
        short = re.sub(
            r"e([-+])0*(\d)", lambda m: f"e{'-' if m.group(1) == '-' else ''}{m.group(2)}", s
        )
        add(short, False)
    return out


def renderings(x: float) -> set[str]:
    """The texts a slide may show for one value (the keys of ``forms``)."""
    return set(forms(x))


def exact(x: float) -> set[str]:
    """The texts that show a script's literal as written, with no padding: 0.05 as "0.05" or "5%".

    A literal is matched only this way, so a constant 0.05 does not explain a "5.0%" on a slide.
    """
    out = set()
    for v, unit in ((x, ""), (100 * x, "%")):
        s = f"{v:.10f}".rstrip("0").rstrip(".")
        decimals = len(s.split(".")[1]) if "." in s else 0
        out |= {s + unit, f"{v:,.{decimals}f}{unit}"}
    mantissa, power = f"{x:.0e}".split("e")
    if float(f"{x:.0e}") == x:
        out.add(f"{mantissa}e{int(power)}")  # 3e-4
    return {t.replace("-", "−") for t in out} | out


def rendered(values: Iterable[float], form: Callable[[float], set[str]] = renderings) -> set[str]:
    """The union of ``form`` (by default ``renderings``) over ``values``."""
    out: set[str] = set()
    for x in values:
        out |= form(x)
    return out


def index(table: list[tuple[str, float]]) -> dict[str, list[tuple[str, float, bool]]]:
    """Text -> the leaves that render to it.

    Args:
        table: (key path, value) for every leaf, from ``keyed_leaves``.

    Returns:
        For each text a slide may show: (key path, value, whether that leaf shows it only by
        re-rounding a tie) for every leaf that gives the text.
    """
    out: dict[str, list[tuple[str, float, bool]]] = {}
    for key, x in table:
        for text, tie in forms(x).items():
            out.setdefault(text, []).append((key, x, tie))
    return out


def value_of(node: ast.expr, env: dict[str, float]) -> float | None:
    """The number an expression of literals and known constants evaluates to, else None."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            return None
        return float(node.value)
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        v = value_of(node.operand, env)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        return binary_value(node, env)
    return None


def binary_value(node: ast.BinOp, env: dict[str, float]) -> float | None:
    """The value of ``a op b`` when both sides evaluate and the result is finite, else None."""
    a, b = value_of(node.left, env), value_of(node.right, env)
    v = None
    if a is not None and b is not None:
        try:
            v = float(BINARY[type(node.op)](a, b))
        except (ArithmeticError, ValueError):
            v = None  # 1 / 0, 10.0 ** 400: not a number a slide shows
    return v if v is not None and math.isfinite(v) else None


def is_literal(node: ast.expr) -> bool:
    """A number written as is: ``35``, ``-0.5``."""
    if isinstance(node, ast.UnaryOp):
        node = node.operand
    return isinstance(node, ast.Constant)


def numbers_of(node: ast.expr, env: dict[str, float]) -> list[tuple[float, bool]]:
    """The numbers one constant's definition shows, each with whether it was computed.

    Args:
        node: The right-hand side of the definition.
        env: The module's scalar constants defined so far.

    Returns:
        (value, computed) for: the expression's value when it evaluates to a number (computed
        unless it is a literal); the elements of a tuple, list, set, or dict, each the same way;
        and every numeric literal inside any other expression (35 in ``35 * PSI_M``, 0.01 in
        ``np.arange(-3.0, 3.0, 0.01)``), but no intermediate product.
    """
    v = value_of(node, env)
    out = [] if v is None else [(v, not is_literal(node))]
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        for e in node.elts:
            out += numbers_of(e, env)
    elif isinstance(node, ast.Dict):
        for e in node.values:
            out += numbers_of(e, env)
    elif v is None or not is_literal(node):
        for sub in ast.walk(node):
            if sub is not node and isinstance(sub, ast.Constant):
                lit = value_of(sub, env)
                out += [] if lit is None else [(lit, False)]
    return out


def definitions(stmt: ast.stmt) -> list[tuple[list[str], ast.expr]]:
    """The (names, value) pairs one module-level statement defines.

    Args:
        stmt: A statement of the module body.

    Returns:
        ``A = B = 3`` gives (["A", "B"], 3); ``A, B = (16, 8)`` gives (["A"], 16) and
        (["B"], 8); anything else that is not an assignment, or that assigns to an attribute or
        a subscript, gives nothing.
    """
    if isinstance(stmt, ast.Assign):
        targets, value = stmt.targets, stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        targets, value = [stmt.target], stmt.value
    else:
        return []
    out: list[tuple[list[str], ast.expr]] = []
    names = [t.id for t in targets if isinstance(t, ast.Name)]
    if names:
        out.append((names, value))
    for t in targets:
        if (
            isinstance(t, ast.Tuple)
            and isinstance(value, ast.Tuple | ast.List)
            and len(t.elts) == len(value.elts)
        ):
            pairs = zip(t.elts, value.elts, strict=True)
            out += [([e.id], v) for e, v in pairs if isinstance(e, ast.Name)]
    return out


def module_constants(path: Path) -> tuple[list[tuple[float, bool]], set[str]]:
    """Numbers a module defines at module level under UPPER_CASE names, and the modules it imports.

    Args:
        path: A Python source file.

    Returns:
        (value, computed) for every numeric literal in those definitions and for the value of
        each evaluable expression among them; and the top-level names of the modules the file
        imports.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    env: dict[str, float] = {}
    values: list[tuple[float, bool]] = []
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.split(".")[0])
    for stmt in tree.body:
        for names, value in definitions(stmt):
            if not all(re.fullmatch(r"_?[A-Z][A-Z0-9_]*", n) for n in names):
                continue
            if any("SEED" in n for n in names):  # a seed is never a quantity on a slide
                continue
            values += numbers_of(value, env)
            top = value_of(value, env)
            if top is not None:
                env.update(dict.fromkeys(names, top))
    return values, imports


def script_constants(script: Path) -> list[tuple[float, bool]]:
    """Constants of a figure script and of the data modules it imports from its directory."""
    values: list[tuple[float, bool]] = []
    seen: set[Path] = set()
    todo = [script]
    while todo:
        path = todo.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        found, imports = module_constants(path)
        values += found
        todo += [script.parent / f"{m}.py" for m in sorted(imports - STYLE_MODULES)]
    return values


def table_columns(line: str) -> list[str]:
    """The cells of a Markdown table row, stripped and lower-cased."""
    return [c.strip().lower() for c in line.strip().strip("|").split("|")]


def continues(chain: Chain, gap: str, token: str) -> bool:
    """Whether a number continues the range or list of the number before it.

    Args:
        chain: What the previous number passes on (``context_reason``).
        gap: The text between the two numbers.
        token: This number as written.

    Returns:
        True after a reason that passes on, over "–", "to", "and", or "or", or over a comma
        when the list was opened by a plural word and the number is a plain integer. Never for
        a percentage: "in Lecture 5, 57% of tests" checks the 57%.
    """
    reason, comma = chain
    if not reason or token.endswith("%"):
        return False
    if CONTINUES.match(gap):
        return True
    return comma and LIST_COMMA.match(gap) is not None and re.fullmatch(r"\d+", token) is not None


def context_reason(
    text: str, m: re.Match[str], *, title: str, column: str, gap: str, chain: Chain
) -> tuple[str, Chain]:
    """Why one number of a line is not data, read from its context.

    Args:
        text: The line, without link targets and inline code.
        m: The number's match in ``text``.
        title: The slide's title.
        column: The header of the table column the number sits in ("" outside a table).
        gap: The text since the previous number of the line.
        chain: What the previous number passes on to the rest of its range or list.

    Returns:
        The reason ("" when the context says nothing), and what this number passes on: a reason
        and whether a comma may continue it. Only a plural word ("Lectures", "rooms", "seeds")
        opens a range or list; "Lecture 5, 57" does not.
    """
    before = text[: m.start()]
    word = before.split()[-1] if before and not before[-1].isspace() else ""
    model = word.endswith(("-", "/")) and re.search(r"[A-Z]", word) is not None
    ref, label = REFERENCE.search(before), LABEL.search(before)
    if title.startswith("References"):
        return "bibliography", NO_CHAIN
    if ref is not None or column in {"lecture", "lectures"}:
        plural = ref is not None and ref.group(0).rstrip().endswith("s")
        return "reference", ("reference", True) if plural else NO_CHAIN
    if label is not None:
        plural = label.group(0).rstrip().endswith("s")
        return "name", ("name", True) if plural else NO_CHAIN
    if continues(chain, gap, m.group(0)):
        return chain[0], chain
    rules = (  # (applies, reason)
        (model or re.fullmatch(r"0\d{3,}", m.group(0)) is not None, "name"),
        (before.endswith("^"), "formula"),
        (
            CLOCK.search(before) is not None and re.fullmatch(r"[0-5]\d", m.group(0)) is not None,
            "clock",
        ),
        (m.start() > 0 and YEAR_IN_PARENS.match(text, m.start() - 1) is not None, "small"),
    )
    for applies, reason in rules:
        if applies:
            return reason, NO_CHAIN
    return "", NO_CHAIN


RANGE_PERCENT = re.compile(r"\s*[–-]\s*[-−]?\d[\d,]*(?:\.\d+)?%")  # the rest of a range "71–92%"


def numbers_in(deck: str) -> list[tuple[int, str, str, str]]:
    """(line, number, reason to skip or "", context) for every number a deck shows.

    Args:
        deck: The deck's Markdown.

    Returns:
        One entry per number outside code blocks, inline code, and image or link targets; the
        reason is a context rule of the module docstring ("reference", "name", "formula",
        "clock", "bibliography", or "small" for a year in parentheses) or "" when the context
        says nothing.
    """
    out = []
    in_code = False
    title = ""
    header: list[str] | None = None  # the current table's header cells
    previous_table_line = False
    for i, line in enumerate(deck.splitlines(), start=1):
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or line.startswith(("author:", "lang:")):
            continue
        if line.startswith("## "):
            title = line[3:].strip()
        text = re.sub(r"\]\([^)]*\)", "]", line)  # drop image and link targets
        text = re.sub(r"`[^`]*`", "", text)  # inline code
        is_table = text.lstrip().startswith("|")
        if is_table and not previous_table_line:
            header = table_columns(text)
        previous_table_line = is_table
        body_row = is_table and header is not None and header != table_columns(text)
        last_end, chain = 0, NO_CHAIN
        for m in NUMBER.finditer(text):
            col = text[: m.start()].count("|") - 1
            column = header[col] if body_row and header and 0 <= col < len(header) else ""
            reason, chain = context_reason(
                text, m, title=title, column=column, gap=text[last_end : m.start()], chain=chain
            )
            last_end = m.end()
            context = text[max(0, m.start() - 40) : m.end() + 25].strip()
            token = m.group(0)
            if not token.endswith("%") and RANGE_PERCENT.match(text, m.end()):
                token += "%"  # "71–92%": the percent sign belongs to both ends
            out.append((i, token, reason, context))
    return out


def small(token: str) -> bool:
    """Years (four digits, no thousands comma, 1900-2100) and integers up to ``SMALL``."""
    bare = token.replace("−", "-").rstrip("%")
    if token.endswith("%") or "." in bare or "e" in bare:
        return False
    if re.fullmatch(r"\d{4}", bare):
        return 1900 <= int(bare) <= 2100
    return abs(int(bare.replace(",", ""))) <= SMALL


def keys_text(hits: list[tuple[str, float, bool]], limit: int = 4) -> str:
    """At most ``limit`` "key = value" pairs, and how many more there are."""
    shown = ", ".join(f"{k} = {v:g}" for k, v, _ in hits[:limit])
    return shown + (f" (+{len(hits) - limit} more)" if len(hits) > limit else "")


def classified(
    deck: Path, constants: set[str], *, check_all: bool
) -> list[tuple[int, str, str, str]]:
    """Every number of a deck with the reason it would be skipped if no leaf renders to it.

    Args:
        deck: The deck's Markdown.
        constants: The texts the script's constants give (``exact`` for literals).
        check_all: No reasons at all (``--all``).

    Returns:
        (line, number, reason or "", context): the context rule's reason, else "small", else
        "constant".
    """
    out = []
    for line, token, reason, context in numbers_in(deck.read_text(encoding="utf-8")):
        why = "" if check_all else reason
        if not check_all and not why:
            if small(token):
                why = "small"
            elif token in constants or token.replace("−", "-") in constants:
                why = "constant"
        out.append((line, token, why, context))
    return out


def check(deck: Path, nums: Path, *, check_all: bool, options: set[str]) -> int:
    """Print one deck's missing numbers; return how many there are.

    Args:
        deck: The deck's Markdown.
        nums: Its numbers.json.
        check_all: Check every number, with no skips and no script constants (``--all``).
        options: The listing flags given: ``--skipped``, ``--keys``, ``--ties``.

    Returns:
        The count of numbers that no leaf renders to and no rule skips.
    """
    if not nums.exists():
        print(f"{deck}: {nums} does not exist (run the figure script)")
        return 1
    known = index(keyed_leaves(json.loads(nums.read_text(encoding="utf-8"))))
    script = deck.parent / "figures" / f"{deck.stem}.py"
    found = [] if check_all else script_constants(script)
    # A literal matches as written; a computed value (35 psi in metres) as slides round it.
    constants = rendered((x for x, c in found if not c), exact) | rendered(x for x, c in found if c)
    missing, skipped, ties, linked = [], [], [], []
    for line, token, why, context in classified(deck, constants, check_all=check_all):
        hits = known.get(token) or known.get(token.replace("−", "-"))
        if hits:
            linked.append((line, token, hits, context))
            if not why and all(tie for _, _, tie in hits):
                ties.append((line, token, hits, context))
        elif why:
            skipped.append((line, token, why, context))
        else:
            missing.append((line, token, context))
    report(f"{deck}", nums.name, missing, skipped, len(ties))
    listed = (("--ties", "tie", ties), ("--keys", "keys", linked))
    for flag, label, rows in listed:
        if flag in options:
            for line, token, hits, context in rows:
                print(f"  {label}, line {line}: {token} <- {keys_text(hits)}  | {context}")
    if "--skipped" in options:
        for line, token, reason, context in skipped:
            print(f"  skipped, {reason}, line {line}: {token}  | {context}")
    return len(missing)


def report(
    name: str,
    table: str,
    missing: list[tuple[int, str, str]],
    skipped: list[tuple[int, str, str, str]],
    ties: int,
) -> None:
    """Print a deck's summary line, then each missing number.

    Args:
        name: The deck's path as given.
        table: The numbers.json file's name.
        missing: (line, number, context) for each number nothing explains.
        skipped: (line, number, reason, context) for each number a rule skipped.
        ties: How many numbers were found only by re-rounding a value on a tie.
    """
    counts = Counter(r for _, _, r, _ in skipped)
    tally = ", ".join(f"{counts[r]} {r}" for r in sorted(counts))
    print(
        f"{name}: {len(missing)} numbers not found in {table}"
        + (f" (skipped: {tally})" if tally else "")
        + (f"; {ties} found only by re-rounding a tie (--ties)" if ties else "")
    )
    for line, token, context in missing:
        print(f"  line {line}: {token}  | {context}")


def main() -> int:
    """Check each deck given."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if unknown := flags - {"--all", "--skipped", "--keys", "--ties"}:
        sys.exit(f"unknown option(s): {', '.join(sorted(unknown))}")
    decks = [Path(a) for a in args if not a.endswith(".json")]
    tables = [Path(a) for a in args if a.endswith(".json")]
    if not decks or (tables and len(decks) != 1) or len(tables) > 1:
        sys.exit(__doc__)
    missing = 0
    for deck in decks:
        nums = tables[0] if tables else deck.parent / "build" / "fig" / deck.stem / "numbers.json"
        missing += check(deck, nums, check_all="--all" in flags, options=flags)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
