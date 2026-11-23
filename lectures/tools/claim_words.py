"""List every sentence of a deck that makes a comparison, a causal claim, or a universal claim.

A title or note that says a model "wins", that
something happens "because", or that a pattern holds "in every room" needs a number in the
figure script's ``numbers.json`` for that sentence, and a comparison between trained models
needs its seeds and gap (a small gap from one seed is "close"). Reading for these
by eye missed some; this tool prints each one with its slide title so none is skipped.

The words are matched with their inflections ("outperforms", "learned", "beaten", "improves")
and with the synonyms a deck uses for the same claim ("close", "halves", "cannot", "nothing",
"comes from", "as good as"). "learning" is left out: it names methods ("machine learning",
"Q-learning"). So are "each" and "no", which are on nearly every slide.

Sentences are read whole: each hard-wrapped paragraph, list item, and speaker-notes paragraph is
joined into one line before it is split into sentences at ".", "!" and "?" (not at ":" or ";",
so the number after a colon stays with its claim), so a claim wrapped across lines ("limited
/ by", "so / that") is found and printed with its subject. Titles and table rows stay one line
each. In code blocks only the comments are read.

Usage: python tools/claim_words.py DECK.md [DECK.md ...]   (prints the hits; exit 0)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Regular-expression alternatives, matched as whole words in lower-cased text.
WORDS: dict[str, tuple[str, ...]] = {
    "comparison": (
        r"wins?", r"winning", r"won", r"beat(?:s|en|ing)?", r"pass(?:es|ed)", r"ahead",
        r"better", r"worse", r"outperform\w*", r"best", r"worst", r"more accurate",
        r"faster", r"slower", r"than", r"match(?:es|ed|ing)?", r"twice", r"lower", r"higher",
        r"close(?!-)", r"closer", r"less", r"fewer", r"smaller", r"larger", r"cheaper",
        r"halve[sd]", r"double[sd]", r"same as", r"as good as", r"improv(?:es|ed|ing|ement)",
        r"ties", r"tied", r"rivals?",
    ),
    "cause": (
        r"because", r"since", r"so", r"due to", r"explain(?:s|ed|ing)?", r"drive[sn]?",
        r"drags", r"know(?:s)?", r"knew", r"learn(?:s|ed|t)?", r"prefer(?:s|red)?",
        r"confuse[sd]", r"hurts?", r"limited by", r"the reason", r"why", r"comes? from",
        r"leads? to", r"results? in", r"thanks to", r"therefore", r"thus", r"hence",
        r"cause[sd]",
    ),
    "universal": (
        r"every", r"everything", r"everywhere", r"all", r"always", r"never", r"none",
        r"nothing", r"only", r"most", r"any", r"cannot", r"can't", r"guarantee[sd]?",
        r"whatever",
    ),
}  # fmt: skip
PATTERNS = {kind: re.compile(r"\b(?:" + "|".join(words) + r")\b") for kind, words in WORDS.items()}
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# Abbreviations whose full stop does not end a sentence (protected while splitting).
ABBREV = re.compile(r"\b(e\.g|i\.e|et al|vs|cf|approx|fig|eq|sec|ref)\.(?=\s)", re.IGNORECASE)
COMMENT = re.compile(r"(?:^|\s)#\s?(.*)$")  # a code line's comment (Python and shell)
WIDTH = 220  # characters printed per sentence


def units(text: str) -> list[str]:
    """Titles and table rows whole; each paragraph, list item, or code comment one line.

    Args:
        text: One slide's Markdown (its title line, body, and notes).

    Returns:
        The units in order. A blank line, a fenced-div line (``::: notes``, ``:::``), a title,
        a table row, an image, or a new list item ends the paragraph before it. In a code block
        only comments are kept, each as its own unit, without its "#".
    """
    out: list[str] = []
    buf: list[str] = []
    in_code = False

    def flush() -> None:
        if buf:
            out.append(" ".join(buf))
            buf.clear()

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            flush()
            in_code = not in_code
            continue
        if in_code:
            m = COMMENT.search(s)
            if m and m.group(1).strip():
                out.append(m.group(1).strip())
            continue
        if not s or s.startswith(":::") or s == "---":
            flush()
        elif s.startswith(("#", "|")):
            flush()
            out.append(s)
        else:
            if LIST_ITEM.match(line) or s.startswith("!["):
                flush()
            buf.append(s)
    flush()
    return out


def sentences(text: str) -> list[str]:
    """Split each unit at sentence ends (. ! ?), keeping titles and table rows whole."""
    out = []
    for unit in units(text):
        if unit.startswith(("#", "|")):
            out.append(unit)
            continue
        marked = ABBREV.sub(lambda m: m.group(0) + "\0", LIST_ITEM.sub("", unit))
        out.extend(s.replace("\0", "") for s in re.split(r"(?<=[.!?])\s+", marked) if s)
    return out


def excerpt(s: str, start: int) -> str:
    """At most ``WIDTH`` characters of ``s``, around the claim word at ``start`` when it is long."""
    if len(s) <= WIDTH:
        return s
    lo = max(0, min(start - WIDTH // 3, len(s) - WIDTH))
    return ("…" if lo else "") + s[lo : lo + WIDTH] + ("…" if lo + WIDTH < len(s) else "")


def hits(path: Path) -> list[tuple[str, str, str]]:
    """(slide title, kind, sentence) for every sentence holding a listed word."""
    title, out = "", []
    for block in re.split(r"(?m)^(## .*)$", path.read_text(encoding="utf-8")):
        if block.startswith("## "):
            title = block[3:].strip()
        for raw in sentences(block):
            s = re.sub(r"\s+", " ", raw).strip()
            low = s.lower()
            for kind, pattern in PATTERNS.items():
                m = pattern.search(low)
                if m:
                    out.append((title, kind, excerpt(s, m.start())))
                    break
    return out


def main() -> int:
    """Print the hits of each deck given."""
    for arg in sys.argv[1:]:
        found = hits(Path(arg))
        print(f"{arg}: {len(found)} sentences to check against numbers.json")
        for title, kind, s in found:
            print(f"  [{kind}] {title} :: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
