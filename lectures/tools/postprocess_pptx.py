"""Post-process a pandoc-generated .pptx for accessibility and portability.

1. **Alt text.** Pandoc writes an image's file path into the picture's ``descr`` attribute and
   shows the Markdown image text as a visible caption. We move the image text into ``descr``
   (what screen readers announce) and remove the caption box. Write images as
   ``![Full alt-text sentence](cee_lNN_topic/figure.png)``.
2. **Equation fallback.** Pandoc wraps every text box that contains an equation in
   ``mc:AlternateContent`` with only an Office-2010 math ``mc:Choice`` (``Requires="a14"``).
   PowerPoint renders the native, editable equation; LibreOffice, Google Slides, and our PDF
   build need an ``mc:Fallback``. As PowerPoint itself does, we make the fallback a picture of
   the text box, typeset by LuaLaTeX in the slide fonts (``math_fallback.py``).
3. **Code and tables.** Code is set to 13 pt with no gap between lines (pandoc uses 18 pt
   plus paragraph spacing, which overflows slides); table text is set to 16 pt. Each code
   block gets its own grey box at the top of its text area, its lines at an exact pitch so
   PowerPoint and LibreOffice agree on the box's height; the slide's other text moves below
   the box. Code must come first in its text area, a listing may have at most
   ``CODE_MAX_LINES`` lines of ``CODE_MAX_CHARS`` characters, and its longest line must fit the
   width of its text area (the build fails otherwise). A paragraph that only holds inline code
   is text, not a code block.
4. **Picture size.** Pandoc shrinks every picture to leave room for the caption that step 1
   removes, so a figure drawn at slide size lands at about 85% and its 14 pt text at 12 pt. We
   scale each picture back up to fill its content area on the slide layout (aspect kept,
   centered) and fail if the enlarged picture would cover text.
5. **Display math.** A ``$$...$$`` paragraph loses its bullet and is centered under the bullet
   it belongs to.
6. **Titles.** With ``--max-title-chars N``, fail if a slide title is longer than N characters
   (the ``cee_*`` decks use 40: lectures/README.md, "Titles are short claims").
7. **Title art.** With ``--title-art ART.png``, the title slide's text is narrowed to the left and
   the picture fills the right 4 in, with a one-line credit; its alt text is read from
   ``ART.txt``. Each lecture's figure script draws its art (``figures/title_art.py``). The
   title is narrowed before its equations, if any, are typeset, so their fallback matches it.

The deck is written to ``OUT.pptx.part`` and renamed when complete.

Usage::

    python tools/postprocess_pptx.py [--max-title-chars N] [--title-art ART.png]
        RAW.pptx SOURCE.md OUT.pptx
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from math_fallback import MathBox, body_levels, box_latex, fallback_picture, render, title_size

NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_A14 = "http://schemas.microsoft.com/office/drawing/2010/main"
EMU_PER_IN = 914400
# Title art: the right 4 in of the 10 x 5.625 in title slide; the text keeps the left 5.25 in.
ART_X, ART_W, ART_H = int(6.0 * EMU_PER_IN), int(4.0 * EMU_PER_IN), int(5.625 * EMU_PER_IN)
TITLE_TEXT_W = int(5.25 * EMU_PER_IN)
ART_CREDIT = "Art drawn in code from this lecture's figures, after Fernando de Szyszlo"
CODE_FONT = "Atkinson Hyperlegible Mono"
CODE_SIZE = 1300
# Code box: every code colour (template/aml-code.theme) keeps >= 4.5:1 contrast on CODE_FILL.
CODE_FILL, CODE_EDGE = "F2F2F2", "C8C8C8"
CODE_PITCH = 1560  # line pitch, hundredths of a point: 1.2 x CODE_SIZE
CODE_INSET_X, CODE_INSET_Y = int(0.12 * EMU_PER_IN), int(0.05 * EMU_PER_IN)
CODE_GAP = int(0.06 * EMU_PER_IN)  # box to the text below (which adds its own 0.05 in inset)
CODE_MAX_LINES, CODE_MAX_CHARS = 16, 75  # what fits a slide at CODE_SIZE (lectures/README.md)
# Advance width of every glyph of Atkinson Hyperlegible Mono, in em (632 of 1000 font units,
# Regular and Bold): 75 characters at 13 pt take 8.56 in, plus the insets 8.80 in of the 9 in.
CODE_ADVANCE_EM = 0.632
EMU_PER_PT = 12700
TABLE_SIZE = 1600

_SUB = str.maketrans(
    "0123456789+-=()aeoxhklmnpstijruvβγρφχ", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜᵢⱼᵣᵤᵥᵦᵧᵨᵩᵪ"
)
_SUP = str.maketrans("0123456789+-=()niT⊤*′", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵀᵀ*′")
_SUB_OK = set("0123456789+-=()aeoxhklmnpstijruvβγρφχ")
_SUP_OK = set("0123456789+-=()niT⊤*′")


def _q(tag: str) -> str:
    return f"{{{NS_M}}}{tag}"


def _script(text: str, sub: bool) -> str:
    """Unicode sub/superscript when every character has one, else ``_(...)``/``^(...)``."""
    ok, table, mark = (_SUB_OK, _SUB, "_") if sub else (_SUP_OK, _SUP, "^")
    if text and all(c in ok for c in text):
        return text.translate(table)
    return f"{mark}{text}" if len(text) == 1 else f"{mark}({text})"


def _child(el: ET.Element, tag: str) -> ET.Element | None:
    return el.find(_q(tag))


def linear(el: ET.Element | None) -> str:  # noqa: PLR0911, PLR0912 (one branch per tag)
    """Render an OMML element as linear Unicode text (approximate, for fallbacks only).

    Args:
        el: OMML element (``m:oMath``, ``m:f``, ...), or ``None``.

    Returns:
        Linear text, e.g. ``‖x − x′‖₂² = 2(1 − s)``.
    """
    if el is None:
        return ""
    tag = el.tag.split("}")[-1]
    kids = lambda: "".join(linear(c) for c in el)  # noqa: E731
    if tag == "t":
        return el.text or ""
    if tag == "r":
        return "".join(t.text or "" for t in el.iter(_q("t")))
    if tag in {"sSub", "sSup", "sSubSup", "sPre"}:
        base = linear(_child(el, "e"))
        sub, sup = linear(_child(el, "sub")), linear(_child(el, "sup"))
        return base + (_script(sub, True) if sub else "") + (_script(sup, False) if sup else "")
    if tag == "f":
        num, den = linear(_child(el, "num")), linear(_child(el, "den"))
        wrap = lambda s: s if len(s) <= 1 else f"({s})"  # noqa: E731
        return f"{wrap(num)}/{wrap(den)}"
    if tag == "rad":
        return f"√({linear(_child(el, 'e'))})"
    if tag == "nary":
        pr = _child(el, "naryPr")
        chr_el = pr.find(_q("chr")) if pr is not None else None
        op = chr_el.get(_q("val")) if chr_el is not None else "∫"
        sub, sup = linear(_child(el, "sub")), linear(_child(el, "sup"))
        return (
            op
            + (_script(sub, True) if sub else "")
            + (_script(sup, False) if sup else "")
            + " "
            + linear(_child(el, "e"))
        )
    if tag == "d":
        pr = _child(el, "dPr")
        get = lambda name, default: (  # noqa: E731
            pr.find(_q(name)).get(_q("val"), default)
            if pr is not None and pr.find(_q(name)) is not None
            else default
        )
        # OMML defaults (as in math_fallback.omml_latex): ( and ), separated by |.
        beg, end, sep = get("begChr", "("), get("endChr", ")"), get("sepChr", "|")
        return beg + sep.join(linear(e) for e in el.findall(_q("e"))) + end
    if tag in {"limLow", "limUpp"}:
        lim = linear(_child(el, "lim"))
        return linear(_child(el, "e")) + (_script(lim, tag == "limLow") if lim else "")
    if tag == "func":
        return linear(_child(el, "fName")) + " " + linear(_child(el, "e"))
    if tag == "acc":
        pr = _child(el, "accPr")
        chr_el = pr.find(_q("chr")) if pr is not None else None
        accent = chr_el.get(_q("val")) if chr_el is not None else "̂"
        return linear(_child(el, "e")) + accent
    if tag == "bar":
        return linear(_child(el, "e")) + "̅"
    if tag == "eqArr":
        return "; ".join(linear(e) for e in el.findall(_q("e")))
    if tag == "m":
        rows = [" ".join(linear(e) for e in mr.findall(_q("e"))) for mr in el.findall(_q("mr"))]
        return "[" + "; ".join(rows) + "]"
    if tag.endswith("Pr") or tag == "ctrlPr":
        return ""
    return kids()


def _math_run(fragment: str) -> str:
    """Replace one ``<a14:m>...</a14:m>`` fragment with a plain DrawingML run."""
    doc = f'<root xmlns:a14="{NS_A14}" xmlns:m="{NS_M}" xmlns:a="{NS_A}">{fragment}</root>'
    root = ET.fromstring(doc)
    text = " ".join(linear(om) for om in root.iter(_q("oMath"))).strip()
    return f'<a:r><a:rPr lang="en-US" dirty="0"/><a:t>{html.escape(text, quote=False)}</a:t></a:r>'


def check_no_math_in_tables(xml: str, where: str) -> None:
    """Fail on equations inside tables: LibreOffice and Google Slides drop such tables.

    Raises:
        ValueError: If a table cell contains an equation.
    """
    for tbl in re.findall(r"<a:tbl>.*?</a:tbl>", xml, re.S):
        if "<a14:m>" in tbl:
            raise ValueError(f"{where}: equation inside a table; write that cell as plain text")


def image_alt_texts(markdown: str) -> dict[str, str]:
    """Map image path -> alt text from ``![alt](path)`` in the Markdown source."""
    alts: dict[str, str] = {}
    for raw_alt, path in re.findall(r"!\[([^\]]*)\]\(([^)\s]+)[^)]*\)", markdown):
        alt = " ".join(raw_alt.split())
        if not alt:
            raise ValueError(f"image {path} has no alt text")
        alts[path] = alt
    return alts


_BR = r"<a:br\b[^>]*/>|<a:br\b[^>]*>.*?</a:br>"


def _lines(para: str) -> list[str]:
    """A paragraph's text, one string per line (split at ``a:br`` and at newlines in ``a:t``)."""
    parts = re.split(_BR, para, flags=re.S)
    text = "\n".join(
        "".join(html.unescape(t) for t in re.findall(r"<a:t>([^<]*)</a:t>", part)) for part in parts
    )
    return text.split("\n")


def _shape_text(sp: str) -> str:
    """A shape's visible text: runs joined as written, paragraphs and line breaks by a space."""
    paras = re.findall(r"<a:p>.*?</a:p>", sp, re.S)
    return " ".join(" ".join(line for para in paras for line in _lines(para)).split())


def fix_images(xml: str, alts: dict[str, str]) -> tuple[str, int]:
    """Set picture ``descr`` to the alt text and drop caption boxes repeating it.

    Args:
        xml: Slide XML.
        alts: Image path -> alt text.

    Returns:
        New XML and the number of pictures fixed.

    Raises:
        KeyError: If a picture's path is missing from the Markdown alt-text map.
    """
    fixed = 0
    used: set[str] = set()

    def repl(m: re.Match[str]) -> str:
        nonlocal fixed
        path = html.unescape(m.group(1))
        if path not in alts:
            raise KeyError(f"no alt text for image {path!r}")
        fixed += 1
        used.add(alts[path])
        return f'descr="{html.escape(alts[path], quote=True)}"'

    xml = re.sub(r'descr="([^"]*)"', repl, xml)
    for alt in used:
        for sp in re.findall(r"<p:sp>(?:(?!<p:sp>).)*?</p:sp>", xml, re.S):
            if 'txBox="1"' in sp and _canon(_shape_text(sp)) == _canon(alt):
                xml = xml.replace(sp, "", 1)
    return xml, fixed


def _canon(text: str) -> str:
    """Undo pandoc's smart typography so a caption can be matched to its Markdown alt text.

    Pandoc turns ``'`` and ``"`` into curly quotes, ``--``/``---`` into en/em dashes, and
    ``...`` into an ellipsis in the visible caption, but not in the alt-text map.
    """
    for old, new in (
        ("---", "—"),
        ("--", "–"),
        ("...", "…"),
        ("‘", "'"),
        ("’", "'"),
        ("“", '"'),
        ("”", '"'),
    ):
        text = text.replace(old, new)
    return " ".join(text.split())


def _resize_runs(xml: str, size: int) -> str:
    """Set ``sz`` on every run property element in ``xml``."""

    def repl(m: re.Match[str]) -> str:
        rpr = m.group(0)
        rpr = re.sub(r'\ssz="\d+"', "", rpr, count=1)
        return rpr.replace("<a:rPr", f'<a:rPr sz="{size}"', 1)

    # Self-closing <a:rPr .../> first, so a match never spans two runs.
    return re.sub(r"<a:rPr\b(?:[^>]*?/>|[^>]*>.*?</a:rPr>)", repl, xml, flags=re.S)


def _is_code(para: str) -> bool:
    """A code-block paragraph: every run in ``CODE_FONT``, laid out as pandoc sets code.

    Pandoc writes a code block as one paragraph with ``indent="0"``, no ``marL``, and no
    bullet. A bullet or a paragraph that only holds inline code has a bullet or ``marL="0"``
    and stays text. A paragraph of two or more lines, all in ``CODE_FONT``, is code either way.
    """
    runs = re.findall(r"<a:r>.*?</a:r>", para, re.S)
    if not runs or not all(f'typeface="{CODE_FONT}"' in r for r in runs):
        return False
    ppr = re.match(r"<a:p><a:pPr\b([^>]*?)/?>", para)
    attrs = ppr[1] if ppr else ""
    block = 'indent="0"' in attrs and "marL=" not in attrs and "<a:buNone" in para
    return block or len(_lines(para)) >= 2


def _style_code_block_lines(xml: str) -> str:
    """Code-block lines (paragraphs whose runs are all monospace): ``CODE_SIZE``, no gap.

    Inline code inside a sentence keeps the sentence's size.
    """
    zero = '<a:spcBef><a:spcPts val="0"/></a:spcBef>'

    def repl(m: re.Match[str]) -> str:
        para = m.group(0)
        if not _is_code(para):
            return para
        para = _resize_runs(para, CODE_SIZE)
        if re.search(r"<a:pPr\b[^>]*/>", para):
            return re.sub(r"<a:pPr\b([^>]*?)\s*/>", rf"<a:pPr\1>{zero}</a:pPr>", para, count=1)
        if "<a:pPr" in para:
            return re.sub(
                r"(<a:pPr\b[^>]*>(?:<a:lnSpc>.*?</a:lnSpc>)?)",
                rf"\1{zero}",
                para,
                count=1,
                flags=re.S,
            )
        return para.replace("<a:p>", f"<a:p><a:pPr>{zero}</a:pPr>", 1)

    return re.sub(r"<a:p>.*?</a:p>", repl, xml, flags=re.S)


def style_code_and_tables(xml: str) -> str:
    """Code-block lines at ``CODE_SIZE`` with no gaps; table text at ``TABLE_SIZE``."""
    xml = _style_code_block_lines(xml)
    return re.sub(
        r"<a:tbl>.*?</a:tbl>", lambda m: _resize_runs(m.group(0), TABLE_SIZE), xml, flags=re.S
    )


Rect = tuple[int, int, int, int]  # x, y, cx, cy in EMU
Placeholders = dict[tuple[str, str], Rect | None]  # (type, idx) -> rectangle
_NON_CONTENT_PH = {"title", "ctrTitle", "subTitle", "dt", "ftr", "sldNum"}


def _code_para(para: str) -> str:
    """A code paragraph for the code box: no left margin, lines at ``CODE_PITCH``."""
    pitch = f'<a:lnSpc><a:spcPts val="{CODE_PITCH}"/></a:lnSpc>'
    m = re.match(r"<a:p><a:pPr\b([^>]*?)(/?)>", para)
    if m is None:
        return para.replace("<a:p>", f'<a:p><a:pPr marL="0" indent="0">{pitch}</a:pPr>', 1)
    attrs = re.sub(r'\s(?:marL|indent)="[^"]*"', "", m[1]) + ' marL="0" indent="0"'
    body = para[m.end() :]
    if m[2]:  # <a:pPr .../> had no children
        return f"<a:p><a:pPr{attrs}>{pitch}</a:pPr>{body}"
    body = re.sub(r"^<a:lnSpc>.*?</a:lnSpc>", "", body, count=1, flags=re.S)
    return f"<a:p><a:pPr{attrs}>{pitch}{body}"


_SP = r"<p:sp>(?:(?!<p:sp>).)*?</p:sp>"  # one shape, never spanning into the next


def box_code(xml: str, layout_xml: str, master_xml: str, where: str) -> tuple[str, int]:
    """Put each code block in a grey box at the top of its text area; other text goes below.

    Args:
        xml: Slide XML, after ``style_code_and_tables``.
        layout_xml: The slide's layout XML.
        master_xml: The layout's master XML.
        where: Slide name for error messages.

    Returns:
        New XML and the number of code boxes.

    Raises:
        ValueError: If text comes before the code in its text area, the text area's position
            is unknown, the box would not fit in the area, or the text below cannot be moved.
    """
    layout, master = _placeholders(layout_xml), _placeholders(master_xml)
    first_id = max((int(i) for i in re.findall(r'<p:cNvPr id="(\d+)"', xml)), default=1) + 1
    next_id = first_id
    out: list[str] = []
    end = 0
    for m in re.finditer(rf"<mc:AlternateContent\b.*?</mc:AlternateContent>|{_SP}", xml, re.S):
        out.append(xml[end : m.start()])
        end = m.end()
        boxed = _box_shape(m.group(0), layout, master, where, next_id)
        if boxed is None:
            out.append(m.group(0))
        else:
            out.append(boxed)
            next_id += 1
    out.append(xml[end:])
    return "".join(out), next_id - first_id


def _box_shape(
    whole: str, layout: Placeholders, master: Placeholders, where: str, shape_id: int
) -> str | None:
    """A shape (or its ``mc:AlternateContent``) with its code in a code box; None if no code.

    A text area with an equation sits in ``mc:AlternateContent``, whose fallback picture
    re-typesets everything inside the wrapper: the code box goes outside it, the text stays in.
    """
    if not whole.startswith("<mc:AlternateContent"):
        parts = _split_code(whole, layout, master, where, shape_id)
        return None if parts is None else parts[0] + parts[1]
    inner = re.search(_SP, whole, re.S)
    if inner is None:
        return None
    parts = _split_code(inner.group(0), layout, master, where, shape_id)
    if parts is None:
        return None
    code, rest = parts
    return code + (whole.replace(inner.group(0), rest, 1) if rest else "")


def _split_code(
    sp: str, layout: Placeholders, master: Placeholders, where: str, shape_id: int
) -> tuple[str, str] | None:
    """(code box, the text area without its code, or "") for a shape; None if it has no code.

    Raises:
        ValueError: If text comes before or between code blocks, the text area has no
            position, or the code does not fit in it (lines or width).
    """
    paras = re.findall(r"<a:p>.*?</a:p>", sp, re.S)
    flags = [_is_code(para) for para in paras]
    if not any(flags):
        return None
    n_code = flags.index(False) if False in flags else len(flags)
    if any(flags[n_code:]):
        raise ValueError(
            f"{where}: text comes before or between code blocks; put the code first "
            "on the slide and the text after it"
        )
    key = _ph_key(sp)
    rect = _rect(sp) or (_resolve(key, layout, master) if key else None)
    if rect is None:
        raise ValueError(f"{where}: a code block in a text area with no position")
    x, y, cx, cy = rect
    lines = [line for para in paras[:n_code] for line in _lines(para)]
    h = round(len(lines) * CODE_PITCH / 100 * EMU_PER_PT) + 2 * CODE_INSET_Y
    if h + (CODE_GAP if n_code < len(paras) else 0) > cy:
        raise ValueError(f"{where}: {len(lines)} code lines do not fit in the text area")
    longest = max(len(line) for line in lines)
    w = round(longest * CODE_SIZE / 100 * CODE_ADVANCE_EM * EMU_PER_PT) + 2 * CODE_INSET_X
    if w > cx:
        raise ValueError(
            f"{where}: a {longest}-character code line is wider than its text area "
            f"({cx / EMU_PER_IN:.2f} in); give the code a slide of its own"
        )
    code = _code_box_xml(shape_id, (x, y, cx, h), paras[:n_code])
    if n_code == len(paras):
        return code, ""
    y2 = y + h + CODE_GAP
    return code, _move_below(sp, paras[:n_code], (x, y2, cx, y + cy - y2), where)


def _code_box_xml(shape_id: int, rect: Rect, paras: list[str]) -> str:
    """A grey text box at ``rect`` holding the code paragraphs, lines at ``CODE_PITCH``."""
    x, y, cx, h = rect
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="Code {shape_id}"/>'
        '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{h}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'<a:solidFill><a:srgbClr val="{CODE_FILL}"/></a:solidFill>'
        f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{CODE_EDGE}"/></a:solidFill></a:ln>'
        "</p:spPr><p:txBody>"
        f'<a:bodyPr wrap="none" lIns="{CODE_INSET_X}" tIns="{CODE_INSET_Y}" '
        f'rIns="{CODE_INSET_X}" bIns="{CODE_INSET_Y}" anchor="t"><a:noAutofit/></a:bodyPr>'
        f"<a:lstStyle/>{''.join(_code_para(para) for para in paras)}"
        "</p:txBody></p:sp>"
    )


def _move_below(sp: str, code: list[str], rect: Rect, where: str) -> str:
    """The text area ``sp`` without its ``code`` paragraphs, moved to ``rect``.

    Raises:
        ValueError: If the shape has neither an empty ``p:spPr`` nor an ``a:xfrm`` to replace.
    """
    rest = sp
    for para in code:
        rest = rest.replace(para, "", 1)
    x, y, cx, cy = rect
    xfrm = f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
    if re.search(r"<p:spPr\s*/>", rest):
        return re.sub(r"<p:spPr\s*/>", f"<p:spPr>{xfrm}</p:spPr>", rest, count=1)
    rest, n = re.subn(r"<a:xfrm\b.*?</a:xfrm>", xfrm, rest, count=1, flags=re.S)
    if n != 1:
        raise ValueError(f"{where}: cannot move the text below its code box")
    return rest


def _rect(fragment: str) -> Rect | None:
    """First ``a:off``/``a:ext`` pair in a shape, or None if it inherits its position."""
    off = re.search(r'<a:off x="(-?\d+)" y="(-?\d+)"\s*/>', fragment)
    ext = re.search(r'<a:ext cx="(\d+)" cy="(\d+)"\s*/>', fragment)
    if off is None or ext is None:
        return None
    return int(off[1]), int(off[2]), int(ext[1]), int(ext[2])


def _ph_key(sp: str) -> tuple[str, str] | None:
    """Placeholder (type, idx) of a shape; type defaults to ``obj`` as in OOXML."""
    ph = re.search(r"<p:ph\b([^>]*?)/?>", sp)
    if ph is None:
        return None
    typ = re.search(r'type="([^"]+)"', ph[1])
    idx = re.search(r'idx="([^"]+)"', ph[1])
    return (typ[1] if typ else "obj", idx[1] if idx else "0")


def _placeholders(xml: str) -> dict[tuple[str, str], Rect | None]:
    """Placeholder key -> explicit rectangle (None when inherited) for a layout or master."""
    return {
        key: _rect(sp)
        for sp in re.findall(r"<p:sp>.*?</p:sp>", xml, re.S)
        if (key := _ph_key(sp)) is not None
    }


def _resolve(key: tuple[str, str], layout: dict, master: dict) -> Rect | None:
    """Rectangle of a placeholder, following layout -> master inheritance."""
    if layout.get(key) is not None:
        return layout[key]
    typ = key[0]
    master_typ = "body" if typ in {"obj", "body"} else typ
    for (mtyp, _), rect in master.items():
        if mtyp == master_typ and rect is not None:
            return rect
    return None


def _target(rels: str, kind: str) -> str:
    """The single relationship target of ``kind`` (e.g. ``slideLayout``) in a .rels file."""
    targets = re.findall(rf'Target="\.\./{kind}s/({kind}\d+\.xml)"', rels)
    if len(targets) != 1:
        raise ValueError(f"expected one {kind} relationship, found {targets}")
    return f"ppt/{kind}s/{targets[0]}"


def fit_pictures(xml: str, layout_xml: str, master_xml: str, where: str) -> tuple[str, int]:
    """Scale each picture to fill the layout content area it sits in.

    Args:
        xml: Slide XML.
        layout_xml: The slide's layout XML.
        master_xml: The layout's master XML.
        where: Slide name for error messages.

    Returns:
        New XML and the number of pictures resized.

    Raises:
        ValueError: If a picture is in no content area, or the enlarged picture covers text.
    """
    layout, master = _placeholders(layout_xml), _placeholders(master_xml)
    regions = [
        r
        for key in layout
        if key[0] not in _NON_CONTENT_PH
        if (r := _resolve(key, layout, master)) is not None
    ]
    text_rects = []
    for sp in re.findall(r"<p:sp>(?:(?!<p:sp>).)*?</p:sp>", xml, re.S):
        key = _ph_key(sp)
        if _shape_text(sp).strip() and (key is None or key[0] not in _NON_CONTENT_PH):
            rect = _rect(sp) or (_resolve(key, layout, master) if key else None)
            if rect is not None:
                text_rects.append(rect)

    resized = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal resized
        pic = m.group(0)
        old = _rect(pic)
        if old is None:
            raise ValueError(f"{where}: picture without a position")
        x, _, cx, cy = old
        centre = x + cx // 2
        inside = [r for r in regions if r[0] <= centre <= r[0] + r[2]]
        if not inside:
            raise ValueError(f"{where}: picture at x={x} lies in no content area")
        rx, ry, rcx, rcy = min(inside, key=lambda r: r[2] * r[3])
        scale = min(rcx / cx, rcy / cy)
        ncx, ncy = round(cx * scale), round(cy * scale)
        new = (rx + (rcx - ncx) // 2, ry + (rcy - ncy) // 2, ncx, ncy)
        for tx, ty, tcx, tcy in text_rects:
            if new[0] < tx + tcx and tx < new[0] + ncx and new[1] < ty + tcy and ty < new[1] + ncy:
                raise ValueError(
                    f"{where}: enlarged picture would cover text; give the "
                    "figure its own slide or column"
                )
        resized += 1
        pic = re.sub(
            r'<a:off x="-?\d+" y="-?\d+"\s*/>', f'<a:off x="{new[0]}" y="{new[1]}" />', pic, count=1
        )
        return re.sub(
            r'<a:ext cx="\d+" cy="\d+"\s*/>', f'<a:ext cx="{ncx}" cy="{ncy}" />', pic, count=1
        )

    return re.sub(r"<p:pic>.*?</p:pic>", repl, xml, flags=re.S), resized


_IMAGE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
_SLIDE = re.compile(r"ppt/slides/(slide\d+\.xml)")
_SLIDE_RELS = re.compile(r"ppt/slides/_rels/(slide\d+\.xml)\.rels")


@dataclass
class _Fixed:
    """A deck's slides after ``_fix_slides``, with what the rest of the build needs."""

    slides: dict[str, str] = field(default_factory=dict)  # fixed XML by slide file name
    boxes: list[MathBox] = field(default_factory=list)  # text boxes with equations
    art_slide: str | None = None  # the slide that received the title art
    n_img: int = 0  # pictures given alt text
    n_fit: int = 0  # pictures resized to fill their area
    n_code: int = 0  # code boxes


def _fix_slides(
    zin: zipfile.ZipFile, alts: dict[str, str], markdown: Path, art_alt: str | None = None
) -> _Fixed:
    """Fix every slide's XML and collect the text boxes that contain equations.

    Args:
        zin: The pandoc deck, open for reading.
        alts: Alt text by image path, from the Markdown source.
        markdown: The Markdown source (named in error messages).
        art_alt: Alt text of the title art, or None for a deck without it. The art goes on the
            first slide with a ``ctrTitle``, before its equation boxes are collected.

    Returns:
        The fixed slides, the equation boxes, the title-art slide, and the counts.
    """
    deck = _Fixed()
    for name in zin.namelist():
        m = _SLIDE.fullmatch(name)
        if not m:
            continue
        where = f"{markdown.name}: {name}"
        xml = zin.read(name).decode("utf-8")
        check_no_math_in_tables(xml, where)
        xml, k = fix_images(xml, alts)
        deck.n_img += k
        layout_name = _target(zin.read(f"ppt/slides/_rels/{m[1]}.rels").decode(), "slideLayout")
        layout_rels = layout_name.replace("slideLayouts/", "slideLayouts/_rels/") + ".rels"
        master_name = _target(zin.read(layout_rels).decode(), "slideMaster")
        layout, master = zin.read(layout_name).decode(), zin.read(master_name).decode()
        if "<p:pic>" in xml:
            xml, k = fit_pictures(xml, layout, master, where)
            deck.n_fit += k
        xml = unbullet_display_math(style_code_and_tables(xml))
        xml, k = box_code(xml, layout, master, where)
        deck.n_code += k
        if art_alt is not None and deck.art_slide is None and 'type="ctrTitle"' in xml:
            xml = add_title_art(xml, art_alt, where)
            deck.art_slide = m[1]
        deck.boxes += math_boxes(xml, m[1], layout, master, where)
        deck.slides[m[1]] = xml
    return deck


def add_title_art(xml: str, alt: str, where: str) -> str:
    """Narrow the title slide's title and subtitle, and add the art and its credit on the right.

    Raises:
        ValueError: If the title or subtitle has no explicit position to narrow.
    """
    for ph in ("ctrTitle", "subTitle"):
        blocks = [
            b for b in re.findall(r"<p:sp>.*?</p:sp>", xml, flags=re.S) if f'type="{ph}"' in b
        ]
        if len(blocks) != 1 or not re.search(r'<a:ext cx="\d+"', blocks[0]):
            raise ValueError(f"{where}: no positioned {ph} placeholder to narrow for the title art")
        narrow = re.sub(r'<a:ext cx="\d+"', f'<a:ext cx="{TITLE_TEXT_W}"', blocks[0], count=1)
        xml = xml.replace(blocks[0], narrow)
    credit_h = int(0.28 * EMU_PER_IN)
    shapes = (
        '<p:pic><p:nvPicPr><p:cNvPr id="90" name="Title art" '
        f'descr="{html.escape(alt, quote=True)}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/>'
        '</p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rIdArt"/><a:stretch>'
        f'<a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="{ART_X}" y="0"/>'
        f'<a:ext cx="{ART_W}" cy="{ART_H}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/>'
        "</a:prstGeom></p:spPr></p:pic>"
        '<p:sp><p:nvSpPr><p:cNvPr id="91" name="Title art credit"/><p:cNvSpPr txBox="1"/>'
        f'<p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{ART_X}" y="{ART_H - credit_h}"/>'
        f'<a:ext cx="{ART_W}" cy="{credit_h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/>'
        '</a:prstGeom><a:noFill/></p:spPr><p:txBody><a:bodyPr wrap="square" lIns="91440" '
        'rIns="91440" anchor="ctr"/><a:lstStyle/><a:p><a:pPr algn="r"/><a:r><a:rPr lang="en-US" '
        f'sz="700"><a:solidFill><a:srgbClr val="D9D9D9"/></a:solidFill></a:rPr><a:t>{ART_CREDIT}'
        "</a:t></a:r></a:p></p:txBody></p:sp>"
    )
    if xml.count("</p:spTree>") != 1:
        raise ValueError(f"{where}: expected one shape tree on the title slide")
    return xml.replace("</p:spTree>", shapes + "</p:spTree>")


def _add_fallbacks(xml: str, pairs: list[tuple[MathBox, Path]], where: str) -> str:
    """Give each equation box on one slide its typeset picture as ``mc:Fallback``."""
    for i, (box, png) in enumerate(pairs, start=1):
        if xml.count(box.block) != 1:
            raise ValueError(f"{where}: equation box not unique")
        fb = fallback_picture(box, f"rIdMath{i}", png)
        xml = xml.replace(
            box.block, box.block.replace("</mc:AlternateContent>", fb + "</mc:AlternateContent>")
        )
    return xml


def _write_deck(
    zin: zipfile.ZipFile,
    out: Path,
    slides: dict[str, str],
    per_slide: dict[str, list[tuple[MathBox, Path]]],
    markdown: Path,
    *,
    art: tuple[str | None, Path | None] = (None, None),
) -> None:
    """Write the fixed deck, with the fallback pictures, the title art, and their relationships."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            slide = _SLIDE.fullmatch(item.filename)
            rel = _SLIDE_RELS.fullmatch(item.filename)
            if slide:
                where = f"{markdown.name}: {slide[1]}"
                data = _add_fallbacks(slides[slide[1]], per_slide.get(slide[1], []), where)
                data = data.encode("utf-8")
            elif rel and (rel[1] in per_slide or rel[1] == art[0]):
                extra = "".join(
                    f'<Relationship Id="rIdMath{i}" Type="{_IMAGE_REL}" '
                    f'Target="../media/math-{rel[1][:-4]}-{i}.png"/>'
                    for i in range(1, len(per_slide.get(rel[1], [])) + 1)
                )
                if rel[1] == art[0]:
                    extra += (
                        f'<Relationship Id="rIdArt" Type="{_IMAGE_REL}" '
                        'Target="../media/title-art.png"/>'
                    )
                data = (
                    data.decode("utf-8")
                    .replace("</Relationships>", extra + "</Relationships>")
                    .encode("utf-8")
                )
            elif item.filename == "[Content_Types].xml":
                types = data.decode("utf-8")
                if 'Extension="png"' not in types:
                    png_type = '<Default Extension="png" ContentType="image/png"/>'
                    types = types.replace("<Default ", png_type + "<Default ", 1)
                data = types.encode("utf-8")
            zout.writestr(item, data)
        for slide_name, pairs in per_slide.items():
            for i, (_, png) in enumerate(pairs, start=1):
                zout.write(png, f"ppt/media/math-{slide_name[:-4]}-{i}.png")
        if art[0] is not None and art[1] is not None:
            zout.write(art[1], "ppt/media/title-art.png")


def process(raw: Path, markdown: Path, out: Path, title_art: Path | None = None) -> None:
    """Write the post-processed copy of ``raw`` to ``out``.

    The deck is written to a temporary file next to ``out`` and moved into place only when
    complete, so a failure never leaves a truncated deck that looks newer than its sources.

    Args:
        raw: Pandoc output.
        markdown: The Markdown source (for image alt texts).
        out: Destination path.
        title_art: A picture for the title slide, with its alt text in the ``.txt`` beside it.

    Raises:
        ValueError: If the deck has no title slide to put the art on.
    """
    alts = image_alt_texts(markdown.read_text(encoding="utf-8"))
    art_alt = None
    if title_art is not None:
        art_alt = title_art.with_suffix(".txt").read_text(encoding="utf-8").strip()
    tmp = out.with_name(out.name + ".part")
    with zipfile.ZipFile(raw) as zin:
        deck = _fix_slides(zin, alts, markdown, art_alt)
        if title_art is not None and deck.art_slide is None:
            raise ValueError(f"{markdown.name}: no title slide for the title art")
        boxes = deck.boxes
        # Typeset all equation boxes of the deck in one LuaLaTeX run.
        pngs = render(boxes, out.parent / "mathfb" / markdown.stem) if boxes else []
        per_slide: dict[str, list[tuple[MathBox, Path]]] = {}
        for box, png in zip(boxes, pngs, strict=True):
            per_slide.setdefault(box.slide, []).append((box, png))
        try:
            art = (deck.art_slide, title_art)
            _write_deck(zin, tmp, deck.slides, per_slide, markdown, art=art)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(out)
    print(
        f"{out}: {deck.n_img} images with alt text ({deck.n_fit} resized to fill their area), "
        f"{len(boxes)} equation boxes with typeset fallbacks, {deck.n_code} code boxes",
        file=sys.stderr,
    )


def unbullet_display_math(xml: str) -> str:
    """Drop the bullet from paragraphs that hold only a displayed equation.

    Pandoc gives a ``$$...$$`` line its own paragraph at the list's level, so PowerPoint draws
    a bullet in front of the centered equation. The paragraph keeps its level's margin.

    Raises:
        ValueError: If such a paragraph has paragraph properties this function does not expect.
    """

    def repl(m: re.Match[str]) -> str:
        p = m.group(0)
        if "<m:oMathPara" not in p:
            return p
        outside = re.sub(r"<a14:m>.*?</a14:m>", "", p, flags=re.S)
        if "".join(re.findall(r"<a:t>([^<]*)</a:t>", outside)).strip():
            return p  # text shares the paragraph: keep its bullet
        if re.search(r"<a:pPr\b[^>]*>(?:(?!</a:pPr>).)*<a:buNone", p, re.S):
            return p  # already unbulleted (a plain paragraph)
        ppr = re.search(r"<a:pPr\b([^>]*?)\s*/>", p)
        if ppr is None:
            raise ValueError("displayed-equation paragraph with unexpected properties")
        attrs = re.sub(r'\s(indent|marL)="[^"]*"', "", ppr[1])
        return p.replace(ppr[0], f'<a:pPr{attrs} indent="0"><a:buNone /></a:pPr>', 1)

    return re.sub(r"<a:p>.*?</a:p>", repl, xml, flags=re.S)


def _match_ph(key: tuple[str, str], layout_xml: str) -> str | None:
    """The layout placeholder ``p:sp`` that a slide placeholder inherits from."""
    for sp in re.findall(r"<p:sp>.*?</p:sp>", layout_xml, re.S):
        other = _ph_key(sp)
        if other is None:
            continue
        if (key[1] != "0" and other[1] == key[1]) or (key[1] == "0" and other[0] == key[0]):
            return sp
    return None


def math_boxes(xml: str, slide: str, layout: str, master: str, where: str) -> list[MathBox]:
    """Text boxes on a slide whose equations need a typeset fallback picture.

    Args:
        xml: Slide XML (after the other fixes).
        slide: Slide file name, e.g. ``slide7.xml``.
        layout: The slide's layout XML.
        master: The layout's master XML.
        where: Slide name for error messages.

    Returns:
        One ``MathBox`` per ``mc:AlternateContent`` that has no fallback yet.

    Raises:
        ValueError: If a box's position cannot be resolved.
    """
    lay_ph, mas_ph = _placeholders(layout), _placeholders(master)
    found = []
    for block in re.findall(r"<mc:AlternateContent\b.*?</mc:AlternateContent>", xml, re.S):
        if "<mc:Fallback" in block or "<a14:m>" not in block:
            continue
        sp = re.search(r"<mc:Choice[^>]*>(.*)</mc:Choice>", block, re.S).group(1)
        key = _ph_key(sp)
        rect = _rect(sp) or (_resolve(key, lay_ph, mas_ph) if key else None)
        if rect is None:
            raise ValueError(f"{where}: cannot place an equation box")
        title = key is not None and key[0] in {"title", "ctrTitle"}
        lay_sp = _match_ph(key, layout) if key else None
        if title:
            size = re.search(r'<a:defRPr[^>]*\bsz="(\d+)"', lay_sp or "")
            latex = box_latex(sp, {}, int(size[1]) if size else title_size(master))
        else:
            latex = box_latex(sp, body_levels(master, lay_sp), None)
        linear_sp = re.sub(r"<a14:m>.*?</a14:m>", lambda f: _math_run(f.group(0)), sp, flags=re.S)
        cnv = re.search(r'<p:cNvPr id="(\d+)" name="([^"]*)"', sp)
        found.append(
            MathBox(
                slide=slide,
                block=block,
                latex=latex,
                rect=rect,
                title=title,
                alt=_shape_text(linear_sp),
                shape_id=cnv[1],
                shape_name=cnv[2],
            )
        )
    return found


def check_titles(markdown: Path, limit: int) -> None:
    """Fail if a slide title is longer than ``limit`` characters.

    Titles are short claims that fit on one line (lectures/README.md, "Titles are short claims").

    Raises:
        ValueError: Listing every title over the limit.
    """
    long, fenced = [], False
    for line in markdown.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("## ") and len(line[3:].strip()) > limit:
            long.append(f"{len(line[3:].strip())}: {line[3:].strip()}")
    if long:
        raise ValueError(f"{markdown}: titles over {limit} characters:\n  " + "\n  ".join(long))


def check_code(markdown: Path) -> None:
    """Fail if a code listing has more than ``CODE_MAX_LINES`` lines or a longer line.

    Raises:
        ValueError: Listing every code block over a limit, by its first line.
    """
    text = markdown.read_text(encoding="utf-8")
    bad = []
    for block in re.findall(r"^```[^\n]*\n(.*?)^```", text, re.S | re.M):
        lines = block.rstrip("\n").split("\n")
        longest = max(len(line) for line in lines)
        if len(lines) > CODE_MAX_LINES or longest > CODE_MAX_CHARS:
            bad.append(f"{len(lines)} lines, longest {longest}: {lines[0].strip()}")
    if bad:
        raise ValueError(
            f"{markdown}: code over {CODE_MAX_LINES} lines or {CODE_MAX_CHARS} characters:\n  "
            + "\n  ".join(bad)
        )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path)
    parser.add_argument("markdown", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--max-title-chars",
        type=int,
        default=None,
        help="fail if a slide title is longer than this",
    )
    parser.add_argument(
        "--title-art",
        type=Path,
        default=None,
        help="picture for the right of the title slide (alt text in the .txt beside it)",
    )
    args = parser.parse_args()
    if args.max_title_chars is not None:
        check_titles(args.markdown, args.max_title_chars)
    check_code(args.markdown)
    process(args.raw, args.markdown, args.out, args.title_art)


if __name__ == "__main__":
    main()
