"""Typeset fallback pictures for slide text boxes that contain equations.

PowerPoint stores equations as Office 2010 math (``a14:m``) inside ``mc:AlternateContent``.
Apps that cannot draw it (LibreOffice, Google Slides, and therefore our PDF build) show the
``mc:Fallback`` instead. PowerPoint itself puts a picture of the rendered text box there; we do
the same. Each text box with an equation is typeset by LuaLaTeX in the slide fonts (Atkinson
Hyperlegible Next for text, Latin Modern Math for math) at the box's width, with the template's
bullets, indents, sizes, and spacing, and the PNG becomes the fallback. PowerPoint keeps showing
the native, editable equations.
"""

from __future__ import annotations

import re
import struct
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

EMU_PER_IN = 914400
DPI = 300
INSET_TB = 45720  # template text-box top/bottom inset (0.05 in); left/right insets are 0

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}

PREAMBLE = r"""\documentclass[multi=shape,border=0pt]{standalone}
\usepackage{amsmath}
\usepackage{fontspec}
\usepackage{unicode-math}
\directlua{luaotfload.add_fallback("aml", {"latinmodern-math.otf:mode=harf;"})}
\setmainfont{AtkinsonHyperlegibleNext}[Extension=.otf, UprightFont=*-Regular,
  BoldFont=*-Bold, ItalicFont=*-RegularItalic, BoldItalicFont=*-BoldItalic,
  RawFeature={fallback=aml}]
\setmonofont{AtkinsonHyperlegibleMono}[Extension=.otf, UprightFont=*-Regular,
  BoldFont=*-Bold, RawFeature={fallback=aml}]
\setmathfont{latinmodern-math.otf}
\usepackage{xcolor}
\definecolor{ink}{HTML}{1A1A1A}
\definecolor{navy}{HTML}{1B3A5C}
\newenvironment{shape}[1]{\begin{minipage}[t]{#1}\color{ink}\raggedright
  \hyphenpenalty=10000 \exhyphenpenalty=10000 \parskip=0pt
  \abovedisplayskip=2pt \belowdisplayskip=2pt}{\end{minipage}}
\begin{document}
"""


def _m(tag: str) -> str:
    return f"{{{NS['m']}}}{tag}"


def _a(tag: str) -> str:
    return f"{{{NS['a']}}}{tag}"


def _val(el: ET.Element | None, child: str, default: str | None = None) -> str | None:
    """``m:val`` of an OMML property child, e.g. ``chr`` of ``m:naryPr``."""
    if el is None:
        return default
    c = el.find(_m(child))
    if c is None:
        return default
    return c.get(_m("val"), default)


# ---------------------------------------------------------------------------------------------
# OMML -> LaTeX (unicode-math)

_MATH_ESCAPE = {
    "{": r"\{",
    "}": r"\}",
    "#": r"\#",
    "%": r"\%",
    "$": r"\$",
    "&": r"\&",
    "_": r"\_",
    "~": r"\sim ",
    "^": r"\hat{}",
    "\\": r"\backslash ",
}
_MATH_SPACE = {
    "\u200b": "",
    "\u2003": r"\quad ",
    "\u2002": r"\enspace ",
    "\u2009": r"\,",
    "\u205f": r"\:",
    "\u2004": r"\;",
    "\u2005": r"\:",
    "\u2006": r"\,",
    "\u00a0": "~",
    " ": r"\ ",
}
_NARY = {
    "∑": r"\sum",
    "∏": r"\prod",
    "∐": r"\coprod",
    "∫": r"\int",
    "∬": r"\iint",
    "∭": r"\iiint",
    "∮": r"\oint",
    "⋃": r"\bigcup",
    "⋂": r"\bigcap",
    "⋁": r"\bigvee",
    "⋀": r"\bigwedge",
    "⨁": r"\bigoplus",
    "⨂": r"\bigotimes",
}
_ACCENT = {
    "\u0302": "hat",
    "\u0303": "tilde",
    "\u0304": "bar",
    "\u203e": "bar",
    "\u0305": "bar",
    "\u0307": "dot",
    "\u0308": "ddot",
    "\u20d7": "vec",
    "\u2192": "vec",
    "\u0306": "breve",
    "\u030c": "check",
    "\u0301": "acute",
    "\u0300": "grave",
    "^": "hat",
    "~": "tilde",
}
_WIDE = {"hat": r"\widehat", "tilde": r"\widetilde", "bar": r"\overline", "vec": r"\overrightarrow"}
_DELIM = {
    "": ".",
    "(": "(",
    ")": ")",
    "[": "[",
    "]": "]",
    "{": r"\{",
    "}": r"\}",
    "|": "|",
    "‖": r"\Vert",
    "⟨": r"\langle",
    "⟩": r"\rangle",
    "〈": r"\langle",
    "〉": r"\rangle",
    "⌊": r"\lfloor",
    "⌋": r"\rfloor",
    "⌈": r"\lceil",
    "⌉": r"\rceil",
    "/": "/",
}
_SCR = {
    "script": r"\symcal",
    "fraktur": r"\symfrak",
    "double-struck": r"\symbb",
    "sans-serif": r"\symsf",
    "monospace": r"\symtt",
}


def _math_text(text: str, in_eqarr: bool) -> str:
    out = []
    for ch in text:
        if ch == "&" and in_eqarr:
            out.append("&")
        elif ch in _MATH_SPACE:
            out.append(_MATH_SPACE[ch])
        else:
            out.append(_MATH_ESCAPE.get(ch, ch))
    return "".join(out)


def _run(el: ET.Element, in_eqarr: bool) -> str:  # noqa: PLR0911 (one branch per style)
    text = "".join(t.text or "" for t in el.iter(_m("t")))
    pr = el.find(_m("rPr"))
    if pr is not None and pr.find(_m("nor")) is not None:
        return r"\text{" + latex_escape(text) + "}"
    body = _math_text(text, in_eqarr)
    if not body.strip():
        return body
    sty = _val(pr, "sty")
    scr = _val(pr, "scr", "roman")
    has_letter = any(c.isalpha() for c in text)
    if scr in _SCR:
        cmd = _SCR[scr]
        if sty in {"b", "bi"}:
            cmd = {"script": r"\symbfcal", "fraktur": r"\symbffrak"}.get(scr, cmd)
        return f"{cmd}{{{body}}}"
    if sty == "b":
        return rf"\symbf{{{body}}}"
    if sty == "bi":
        return rf"\symbfit{{{body}}}"
    if sty == "p" and has_letter:
        return rf"\symup{{{body}}}"
    return body


def omml_latex(el: ET.Element | None, in_eqarr: bool = False) -> str:
    """Convert an OMML element to LaTeX math for ``unicode-math``.

    Args:
        el: An OMML element (``m:oMath``, ``m:f``, ...) or ``None``.
        in_eqarr: Inside an equation array, where ``&`` marks alignment.

    Returns:
        LaTeX math-mode source.

    Raises:
        ValueError: On an OMML construct this converter does not know.
    """
    if el is None:
        return ""
    tag = el.tag.split("}")[-1]
    if tag.endswith("Pr") or tag in {"ctrlPr", "argPr"}:
        return ""
    if tag in _CONTAINERS:
        return _kids(el, in_eqarr)
    convert = _OMML.get(tag)
    if convert is None:
        raise ValueError(f"OMML element m:{tag} is not supported by the fallback renderer")
    return convert(el, in_eqarr)


# Elements that are just their children, in order.
_CONTAINERS = {"oMath", "e", "num", "den", "sub", "sup", "lim", "deg", "fName", "mr"}


def _sub(el: ET.Element, name: str, in_eqarr: bool) -> str:
    """LaTeX of the child ``m:<name>`` of ``el`` ("" if it has none)."""
    return omml_latex(el.find(_m(name)), in_eqarr)


def _kids(el: ET.Element, in_eqarr: bool) -> str:
    """LaTeX of all children of ``el``, concatenated."""
    return "".join(omml_latex(c, in_eqarr) for c in el)


def _on(pr: ET.Element | None, name: str) -> bool:
    """An OMML on/off property (``degHide``, ``subHide``, ...); off when absent."""
    return _val(pr, name, "off") in {"on", "1", "true"}


def _scripts(el: ET.Element, base: str, in_eqarr: bool) -> str:
    """``base`` with ``el``'s ``m:sub`` and ``m:sup``."""
    lo, hi = _sub(el, "sub", in_eqarr).strip(), _sub(el, "sup", in_eqarr).strip()
    if hi and set(hi) <= set("′″‴"):
        # unicode-math raises primes itself; f′_c, not f_c^{′} (which would be raised twice).
        return base + hi + (f"_{{{lo}}}" if lo else "")
    return base + (f"_{{{lo}}}" if lo else "") + (f"^{{{hi}}}" if hi else "")


def _script(el: ET.Element, in_eqarr: bool) -> str:
    """``m:sSub``, ``m:sSup``, ``m:sSubSup``."""
    return _scripts(el, f"{{{_sub(el, 'e', in_eqarr)}}}", in_eqarr)


def _pre_script(el: ET.Element, in_eqarr: bool) -> str:
    """``m:sPre``: scripts before the base."""
    return _scripts(el, "{}", in_eqarr) + f"{{{_sub(el, 'e', in_eqarr)}}}"


def _fraction(el: ET.Element, in_eqarr: bool) -> str:
    """``m:f``, by its ``type``: bar (default), lin, skw, noBar."""
    kind = _val(el.find(_m("fPr")), "type", "bar")
    num, den = _sub(el, "num", in_eqarr), _sub(el, "den", in_eqarr)
    if kind == "lin":
        return f"{num}/{den}"
    if kind == "skw":
        return f"{{}}^{{{num}}}\\!/\\!{{}}_{{{den}}}"
    if kind == "noBar":
        return rf"\genfrac{{}}{{}}{{0pt}}{{}}{{{num}}}{{{den}}}"
    return rf"\frac{{{num}}}{{{den}}}"


def _radical(el: ET.Element, in_eqarr: bool) -> str:
    """``m:rad``: a square root, or an n-th root when it shows a degree."""
    deg = _sub(el, "deg", in_eqarr).strip()
    body = _sub(el, "e", in_eqarr)
    if _on(el.find(_m("radPr")), "degHide") or not deg:
        return rf"\sqrt{{{body}}}"
    return rf"\sqrt[{deg}]{{{body}}}"


def _nary(el: ET.Element, in_eqarr: bool) -> str:
    """``m:nary``: a sum, product, or integral (the OMML default) with its limits."""
    pr = el.find(_m("naryPr"))
    op = _val(pr, "chr", "∫")
    cmd = _NARY.get(op, op)
    lo = "" if _on(pr, "subHide") else _sub(el, "sub", in_eqarr).strip()
    hi = "" if _on(pr, "supHide") else _sub(el, "sup", in_eqarr).strip()
    body = _sub(el, "e", in_eqarr)
    return cmd + (f"_{{{lo}}}" if lo else "") + (f"^{{{hi}}}" if hi else "") + f"{{{body}}}"


def _delimiter(el: ET.Element, in_eqarr: bool) -> str:
    """``m:d``: its parts between ``begChr`` and ``endChr``, separated by ``sepChr``."""
    pr = el.find(_m("dPr"))
    beg, end = _val(pr, "begChr", "("), _val(pr, "endChr", ")")
    sep = _val(pr, "sepChr", "|")
    parts = [omml_latex(e, in_eqarr) for e in el.findall(_m("e"))]
    joiner = r" \middle| " if sep == "|" else _math_text(sep, in_eqarr) + " "
    return rf"\left{_DELIM.get(beg, beg)} " + joiner.join(parts) + rf" \right{_DELIM.get(end, end)}"


def _function(el: ET.Element, in_eqarr: bool) -> str:
    r"""``m:func``: an operator name (``\operatorname`` when it is plain text) and its argument."""
    name = el.find(_m("fName"))
    if name is not None and all(c.tag == _m("r") for c in name):
        fname = "".join(t.text or "" for t in name.iter(_m("t")))
        head = rf"\operatorname{{{fname}}}"
    else:
        head = rf"\mathop{{{_sub(el, 'fName', in_eqarr)}}}"
    return f"{head}{{{_sub(el, 'e', in_eqarr)}}}"


def _lim_low(el: ET.Element, in_eqarr: bool) -> str:
    r"""``m:limLow``: a limit below (``\lim_{x \to 0}``)."""
    return rf"\mathop{{{_sub(el, 'e', in_eqarr)}}}_{{{_sub(el, 'lim', in_eqarr)}}}"


def _lim_upp(el: ET.Element, in_eqarr: bool) -> str:
    """``m:limUpp``: a limit above."""
    return rf"\overset{{{_sub(el, 'lim', in_eqarr)}}}{{{_sub(el, 'e', in_eqarr)}}}"


def _accent(el: ET.Element, in_eqarr: bool) -> str:
    """``m:acc``: hat, tilde, bar, ...; the wide form over more than one symbol.

    Raises:
        ValueError: On an accent character this converter does not know.
    """
    char = _val(el.find(_m("accPr")), "chr", "\u0302")
    name = _ACCENT.get(char)
    if name is None:
        raise ValueError(f"unknown OMML accent {char!r}")
    body = _sub(el, "e", in_eqarr)
    wide = len(re.sub(r"\\[a-zA-Z]+|[{}\s]", "", body)) > 1
    cmd = _WIDE.get(name, "\\" + name) if wide else "\\" + name
    return f"{cmd}{{{body}}}"


def _bar(el: ET.Element, in_eqarr: bool) -> str:
    """``m:bar``: a line under (the OMML default) or over its argument."""
    pos = _val(el.find(_m("barPr")), "pos", "bot")
    return (r"\overline{" if pos == "top" else r"\underline{") + _sub(el, "e", in_eqarr) + "}"


def _group_chr(el: ET.Element, in_eqarr: bool) -> str:
    """``m:groupChr``: a brace under (the OMML default) or over its argument."""
    pr = el.find(_m("groupChrPr"))
    top = _val(pr, "pos", "bot") == "top" or _val(pr, "chr", "⏟") == "⏞"
    return (r"\overbrace{" if top else r"\underbrace{") + _sub(el, "e", in_eqarr) + "}"


def _eq_array(el: ET.Element, _in_eqarr: bool) -> str:
    """``m:eqArr``: rows, aligned at ``&`` when any row has one."""
    rows = [omml_latex(e, True) for e in el.findall(_m("e"))]
    env = "aligned" if any("&" in r for r in rows) else "gathered"
    return rf"\begin{{{env}}}" + r" \\ ".join(rows) + rf"\end{{{env}}}"


def _matrix(el: ET.Element, in_eqarr: bool) -> str:
    """``m:m``: a matrix, row by row."""
    rows = [
        " & ".join(omml_latex(e, in_eqarr) for e in mr.findall(_m("e")))
        for mr in el.findall(_m("mr"))
    ]
    return r"\begin{matrix}" + r" \\ ".join(rows) + r"\end{matrix}"


_OMML: dict[str, Callable[[ET.Element, bool], str]] = {
    "r": _run,
    "sSub": _script,
    "sSup": _script,
    "sSubSup": _script,
    "sPre": _pre_script,
    "f": _fraction,
    "rad": _radical,
    "nary": _nary,
    "d": _delimiter,
    "func": _function,
    "limLow": _lim_low,
    "limUpp": _lim_upp,
    "acc": _accent,
    "bar": _bar,
    "groupChr": _group_chr,
    "eqArr": _eq_array,
    "m": _matrix,
    "borderBox": lambda el, eq: rf"\boxed{{{_sub(el, 'e', eq)}}}",
    "box": lambda el, eq: _sub(el, "e", eq),
    "phant": lambda el, eq: rf"\phantom{{{_sub(el, 'e', eq)}}}",
}


# ---------------------------------------------------------------------------------------------
# Paragraph styles from the master and layout


@dataclass(frozen=True)
class Level:
    """Paragraph style of one outline level (lengths in EMU, size in hundredths of a point)."""

    mar_l: int
    indent: int
    size: int
    bullet: str | None
    spc_bef: float  # points


def _overlay(base: Level, ppr: ET.Element | None) -> Level:
    """Apply the attributes and children of an ``a:lvlNpPr``/``a:pPr`` on top of ``base``."""
    if ppr is None:
        return base
    mar_l = int(ppr.get("marL", base.mar_l))
    indent = int(ppr.get("indent", base.indent))
    size, bullet, spc = base.size, base.bullet, base.spc_bef
    defr = ppr.find(_a("defRPr"))
    if defr is not None and defr.get("sz"):
        size = int(defr.get("sz"))
    if ppr.find(_a("buNone")) is not None:
        bullet = None
    elif ppr.find(_a("buChar")) is not None:
        bullet = ppr.find(_a("buChar")).get("char")
    elif ppr.find(_a("buAutoNum")) is not None:
        bullet = "#"
    pts = ppr.find(f"{_a('spcBef')}/{_a('spcPts')}")
    if pts is not None:
        spc = int(pts.get("val")) / 100
    return Level(mar_l, indent, size, bullet, spc)


def body_levels(master_xml: str, layout_sp: str | None) -> dict[int, Level]:
    """Outline-level styles for a body placeholder: master ``bodyStyle`` + layout overrides.

    Args:
        master_xml: Slide master XML.
        layout_sp: The matching layout placeholder ``p:sp`` XML, if any.

    Returns:
        Level index (0-based) -> style.
    """
    style = re.search(r"<p:bodyStyle>.*?</p:bodyStyle>", master_xml, re.S)
    if style is None:
        raise ValueError("slide master has no bodyStyle")
    root = _parse(style.group(0))
    levels: dict[int, Level] = {}
    base = Level(0, 0, 1800, None, 0.0)
    for i in range(9):
        levels[i] = _overlay(base, root.find(f".//{_a(f'lvl{i + 1}pPr')}"))
    if layout_sp:
        lst = _parse(layout_sp).find(f".//{_a('lstStyle')}")
        if lst is not None:
            for i in range(9):
                levels[i] = _overlay(levels[i], lst.find(_a(f"lvl{i + 1}pPr")))
    return levels


def title_size(master_xml: str) -> int:
    """Title font size (hundredths of a point) from the master ``titleStyle``."""
    m = re.search(r"<p:titleStyle>.*?<a:defRPr[^>]*\bsz=\"(\d+)\"", master_xml, re.S)
    if m is None:
        raise ValueError("slide master titleStyle has no size")
    return int(m.group(1))


def _parse(fragment: str) -> ET.Element:
    decl = " ".join(f'xmlns:{k}="{v}"' for k, v in NS.items())
    return ET.fromstring(f"<root {decl}>{fragment}</root>")


# ---------------------------------------------------------------------------------------------
# Text box -> LaTeX

_TEXT_ESCAPE = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    """Escape text for LaTeX text mode."""
    return "".join(_TEXT_ESCAPE.get(c, c) for c in text)


_DISPLAY = r"\par{\centering\leavevmode$\displaystyle "
_DISPLAY_END = r"$\par}"


def _len(emu: int) -> str:
    return f"{emu / EMU_PER_IN:.4f}in"


def _pt(hundredths: int, factor: float = 1.0) -> str:
    return f"{hundredths / 100 * factor:.2f}pt"


def _text_run(r: ET.Element) -> tuple[str, int | None]:
    rpr = r.find(_a("rPr"))
    text = latex_escape("".join(t.text or "" for t in r.iter(_a("t"))))
    size = None
    if rpr is not None:
        latin = rpr.find(_a("latin"))
        if latin is not None and "Mono" in (latin.get("typeface") or ""):
            text = rf"\texttt{{{text.replace(' ', '~')}}}"  # keep code indentation
        if rpr.get("b") in {"1", "true"}:
            text = rf"\textbf{{{text}}}"
        if rpr.get("i") in {"1", "true"}:
            text = rf"\textit{{{text}}}"
        if rpr.get("sz"):
            size = int(rpr.get("sz"))
    return text, size


def _paragraph_parts(p: ET.Element) -> tuple[list[str], list[int]]:
    """LaTeX for each run, line break, and equation of one ``a:p``, and the run font sizes."""
    parts: list[str] = []
    sizes: list[int] = []
    for child in p:
        tag = child.tag.split("}")[-1]
        if tag in {"r", "fld"}:
            text, size = _text_run(child)
            if size:
                sizes.append(size)
                text = rf"{{\fontsize{{{_pt(size)}}}{{{_pt(size, 1.2)}}}\selectfont {text}}}"
            parts.append(text)
        elif tag == "br":
            # TeX discards glue after a line break; the empty box keeps the leading
            # spaces of the next line (code indentation).
            parts.append(r"\newline\mbox{}")
        elif tag == "m":  # a14:m
            para = child.find(_m("oMathPara"))
            if para is not None:
                eq = " ".join(omml_latex(om) for om in para.iter(_m("oMath")))
                parts.append(_DISPLAY + eq + _DISPLAY_END)
            else:
                parts.append("$" + "".join(omml_latex(om) for om in child.iter(_m("oMath"))) + "$")
    return parts, sizes


def _number(ppr: ET.Element | None, lvl: int, counters: dict[int, int]) -> int | None:
    """Advance the auto-numbering ``counters`` for a paragraph; return its number, if any."""
    num = ppr.find(_a("buAutoNum")) if ppr is not None else None
    for deeper in [k for k in counters if k > lvl]:
        del counters[deeper]
    if num is None:
        counters.pop(lvl, None)
        return None
    counters[lvl] = counters.get(lvl, int(num.get("startAt", "1")) - 1) + 1
    return counters[lvl]


def _paragraph_latex(
    parts: list[str], size: int, style: Level, number: int | None, title: bool
) -> str:
    """Set one paragraph: a centered display equation, or text with its bullet or number."""
    lead = 0.9 * 1.2 if title else 1.2
    shown = [x for x in parts if x.strip()]
    if all(x.startswith(_DISPLAY) for x in shown):
        # A displayed equation on its own: centered in the text area, no bullet line.
        eqs = "".join(x[len(_DISPLAY) : -len(_DISPLAY_END)] for x in shown)
        return (
            rf"{{\fontsize{{{_pt(size)}}}{{{_pt(size, lead)}}}\selectfont"
            rf"\leftskip={_len(style.mar_l)} plus 1fil\rightskip=0pt plus 1fil"
            rf"\parfillskip=0pt\parindent=0pt\noindent$\displaystyle {eqs}$\par}}"
        )
    marker = ""
    if number is not None:
        marker = rf"\textcolor{{navy}}{{{number}.}}"
    elif style.bullet:
        marker = rf"\textcolor{{navy}}{{{latex_escape(style.bullet)}}}"
    hang = -style.indent
    box = rf"\makebox[{_len(hang)}][l]{{{marker}}}" if hang > 0 else marker
    weight = r"\bfseries\color{navy}" if title else ""
    return (
        rf"{{\fontsize{{{_pt(size)}}}{{{_pt(size, lead)}}}\selectfont{weight}"
        rf"\leftskip={_len(style.mar_l)}\relax\parindent={_len(style.indent)}\relax"
        rf"\indent{{}}{box}{''.join(parts)}\par}}"
    )


def box_latex(sp: str, levels: dict[int, Level], title_sz: int | None) -> str:
    """Typeset one text box (``p:sp`` XML) as the body of a ``shape`` environment.

    Args:
        sp: The text box from pandoc's ``mc:Choice``.
        levels: Outline-level styles (ignored for titles).
        title_sz: Title font size in hundredths of a point, or None for a body box.

    Returns:
        LaTeX source for the inside of the box.
    """
    out: list[str] = []
    counters: dict[int, int] = {}
    for p in _parse(sp).iter(_a("p")):
        ppr = p.find(_a("pPr"))
        lvl = int(ppr.get("lvl", "0")) if ppr is not None else 0
        if title_sz is not None:
            style = Level(0, 0, title_sz, None, 0.0)
        else:
            style = _overlay(levels[lvl], ppr)
        number = _number(ppr, lvl, counters)
        parts, sizes = _paragraph_parts(p)
        if not "".join(parts).strip():
            continue
        if out and style.spc_bef:
            out.append(rf"\vspace{{{style.spc_bef:.1f}pt}}")
        out.append(
            _paragraph_latex(parts, max([style.size, *sizes]), style, number, title_sz is not None)
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# Rendering and the fallback picture


@dataclass
class MathBox:
    """One text box that gets a typeset fallback picture."""

    slide: str  # e.g. "slide7.xml"
    block: str  # the original mc:AlternateContent XML
    latex: str  # inside of the shape environment
    rect: tuple[int, int, int, int]
    title: bool
    alt: str
    shape_id: str
    shape_name: str


def render(boxes: list[MathBox], workdir: Path) -> list[Path]:
    """Typeset every box in one LuaLaTeX run and rasterize one PNG per box.

    Args:
        boxes: Boxes in output order.
        workdir: Scratch directory (created if missing).

    Returns:
        PNG paths, one per box, in order.

    Raises:
        RuntimeError: If LuaLaTeX fails or reports missing glyphs, or the page count is wrong.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    for old in workdir.glob("box-*.png"):
        old.unlink()
    pages = [
        rf"\begin{{shape}}{{{_len(b.rect[2])}}}" + "\n" + b.latex + "\n" + r"\end{shape}"
        for b in boxes
    ]
    tex = workdir / "boxes.tex"
    tex.write_text(PREAMBLE + "\n\n".join(pages) + "\n\\end{document}\n", encoding="utf-8")
    run = subprocess.run(
        ["lualatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    log = (workdir / "boxes.log").read_text(encoding="utf-8", errors="replace")
    if run.returncode != 0:
        tail = "\n".join(log.splitlines()[-25:])
        raise RuntimeError(f"LuaLaTeX failed on {tex}:\n{tail}")
    missing = sorted(set(re.findall(r"Missing character: There is no (\S+)", log)))
    if missing:
        raise RuntimeError(f"{tex}: glyphs missing from the fonts: {' '.join(missing)}")
    subprocess.run(
        ["pdftocairo", "-r", str(DPI), "-png", "-transp", "boxes.pdf", "box"],
        cwd=workdir,
        check=True,
    )
    pngs = sorted(workdir.glob("box-*.png"))
    if len(pngs) != len(boxes):
        raise RuntimeError(f"{tex}: expected {len(boxes)} pages, got {len(pngs)}")
    return pngs


def png_size(path: Path) -> tuple[int, int]:
    """Width and height in pixels of a PNG file."""
    head = path.read_bytes()[:24]
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    return struct.unpack(">II", head[16:24])


def fallback_picture(box: MathBox, rid: str, png: Path) -> str:
    """The ``mc:Fallback`` XML: the typeset box placed where the text box sits.

    Args:
        box: The text box.
        rid: Relationship id of the PNG in the slide's rels.
        png: The rendered PNG.

    Returns:
        ``<mc:Fallback>...</mc:Fallback>``.
    """
    w_px, h_px = png_size(png)
    x, y, cx, _ = box.rect
    cy = round(h_px / w_px * cx)
    top = y + box.rect[3] - INSET_TB - cy if box.title else y + INSET_TB
    alt = box.alt.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    return (
        "<mc:Fallback><p:pic><p:nvPicPr>"
        f'<p:cNvPr id="{box.shape_id}" name="{box.shape_name}" descr="{alt}" />'
        '<p:cNvPicPr><a:picLocks noChangeAspect="1" /></p:cNvPicPr><p:nvPr /></p:nvPicPr>'
        f'<p:blipFill><a:blip r:embed="{rid}" /><a:stretch><a:fillRect /></a:stretch>'
        "</p:blipFill><p:spPr>"
        f'<a:xfrm><a:off x="{x}" y="{top}" /><a:ext cx="{cx}" cy="{cy}" /></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst /></a:prstGeom></p:spPr></p:pic></mc:Fallback>'
    )
