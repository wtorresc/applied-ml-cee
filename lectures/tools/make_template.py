"""Build the PowerPoint template (``template/aml-reference.pptx``) and code colors.

Starts from pandoc's default ``reference.pptx`` (16:9, 10 x 5.625 in) and applies the lecture
style: white background, Atkinson Hyperlegible Next, navy left-aligned titles with a short blue
bar, 20 pt body text (about 27 pt on a 13.33 in slide), a footer, and slide numbers. Pandoc
selects slide layouts by name, so only layout geometry and styles change, never names.

Run once, or whenever the style changes::

    python tools/make_template.py --out template/aml-reference.pptx

PowerPoint can also edit the result directly (View > Slide Master); if you do, stop
regenerating it with this script, or your edits will be overwritten.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

FONT = "Atkinson Hyperlegible Next"
FOOTER = "Wilfredo Torres Calderon · Applied Machine Learning"
CREDITS = (
    "Original slides, figures, and code. Some slides inspired by Derek Hoiem's CS 441 "
    "Applied Machine Learning, University of Illinois Urbana-Champaign."
)
MUTED = "595959"

EMU_PER_IN = 914400
# Geometry (EMU) shared by layouts.
LEFT = int(0.5 * EMU_PER_IN)
WIDTH = int(9.0 * EMU_PER_IN)
# Title slide: text on the left; the lecture's title art fills the right 4 in (postprocess_pptx)
TITLE_TEXT_W = int(5.25 * EMU_PER_IN)
TITLE_Y, TITLE_H = int(0.18 * EMU_PER_IN), int(0.82 * EMU_PER_IN)
BAR_Y, BAR_W, BAR_H = int(1.04 * EMU_PER_IN), int(0.5 * EMU_PER_IN), int(0.035 * EMU_PER_IN)
BODY_Y, BODY_BOTTOM = int(1.2 * EMU_PER_IN), int(5.0 * EMU_PER_IN)
BODY_H = BODY_BOTTOM - BODY_Y
FOOT_Y, FOOT_H = int(5.22 * EMU_PER_IN), int(0.25 * EMU_PER_IN)

DECORATIVE = (
    '<a:extLst><a:ext uri="{C183D7F6-B498-43B3-948B-1728B52AA6E4}">'
    '<adec:decorative xmlns:adec="http://schemas.microsoft.com/office/drawing/2017/'
    'decorative" val="1"/></a:ext></a:extLst>'
)


# Syntax colors for code blocks: every token color has >= 4.5:1 contrast on white.
CODE_COLORS: dict[str, tuple[str, bool, bool]] = {  # token -> (color, bold, italic)
    "Keyword": ("#1B3A5C", True, False),
    "ControlFlow": ("#1B3A5C", True, False),
    "Import": ("#1B3A5C", True, False),
    "DataType": ("#1B3A5C", False, False),
    "String": ("#00704A", False, False),
    "Char": ("#00704A", False, False),
    "VerbatimString": ("#00704A", False, False),
    "SpecialString": ("#00704A", False, False),
    "Comment": ("#595959", False, True),
    "Documentation": ("#595959", False, True),
    "CommentVar": ("#595959", False, True),
    "Annotation": ("#595959", False, True),
    "DecVal": ("#A33A00", False, False),
    "BaseN": ("#A33A00", False, False),
    "Float": ("#A33A00", False, False),
    "Constant": ("#A33A00", False, False),
    "SpecialChar": ("#A33A00", False, False),
    "BuiltIn": ("#0072B2", False, False),
    "Function": ("#0072B2", False, False),
}


def code_theme() -> str:
    """Pandoc highlight theme (JSON) in the lecture palette; unlisted tokens use ink."""
    base = json.loads(
        subprocess.run(
            ["pandoc", "--print-highlight-style", "pygments"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    )
    base["text-color"] = "#1A1A1A"
    base["background-color"] = None
    for token, style in base["text-styles"].items():
        color, bold, italic = CODE_COLORS.get(token, ("#1A1A1A", False, False))
        style.update(
            {
                "text-color": color,
                "background-color": None,
                "bold": bold,
                "italic": italic,
                "underline": False,
            }
        )
    return json.dumps(base, indent=2) + "\n"


def _xfrm(x: int, y: int, w: int, h: int) -> str:
    return f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'


def _bar(shape_id: int, y: int) -> str:
    """A short accent-colored rule, marked decorative for screen readers."""
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="Accent bar">{DECORATIVE}'
        '</p:cNvPr><p:cNvSpPr/><p:nvPr userDrawn="1"/></p:nvSpPr>'
        f'<p:spPr>{_xfrm(LEFT, y, BAR_W, BAR_H)}<a:prstGeom prst="rect"><a:avLst/>'
        '</a:prstGeom><a:solidFill><a:schemeClr val="accent1"/></a:solidFill>'
        "<a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p>"
        '<a:endParaRPr lang="en-US"/></a:p></p:txBody></p:sp>'
    )


def _textbox(
    shape_id: int,
    name: str,
    box: tuple[int, int, int, int],
    runs: str,
    *,
    algn: str = "l",
    anchor: str = "ctr",
) -> str:
    """A non-placeholder text box (shown on every slide that uses the layout)."""
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{name}"/>'
        '<p:cNvSpPr txBox="1"/><p:nvPr userDrawn="1"/></p:nvSpPr>'
        f'<p:spPr>{_xfrm(*box)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        "<a:noFill/></p:spPr>"
        f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" '
        f'anchor="{anchor}"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
        f'<a:p><a:pPr algn="{algn}"/>{runs}</a:p></p:txBody></p:sp>'
    )


def _run(text: str, size: int, color: str, bold: bool = False) -> str:
    """A text run; ``text`` is escaped for XML (an ``&`` in a credit line stays valid)."""
    b = ' b="1"' if bold else ""
    return (
        f'<a:r><a:rPr lang="en-US" sz="{size}"{b} dirty="0"><a:solidFill>'
        f'<a:srgbClr val="{color}"/></a:solidFill></a:rPr>'
        f"<a:t>{html.escape(text, quote=False)}</a:t></a:r>"
    )


def _footer_shapes(first_id: int) -> str:
    """Footer text (left) and slide number (right)."""
    number = (
        '<a:fld id="{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}" type="slidenum">'
        f'<a:rPr lang="en-US" sz="1000"><a:solidFill><a:srgbClr val="{MUTED}"/>'
        "</a:solidFill></a:rPr><a:t>‹#›</a:t></a:fld>"
    )
    return _textbox(
        first_id,
        "Footer text",
        (LEFT, FOOT_Y, int(7 * EMU_PER_IN), FOOT_H),
        _run(FOOTER, 1000, MUTED),
    ) + _textbox(
        first_id + 1,
        "Slide number",
        (LEFT + WIDTH - int(1.0 * EMU_PER_IN), FOOT_Y, int(1.0 * EMU_PER_IN), FOOT_H),
        number,
        algn="r",
    )


def _sub_once(pattern: str, repl: str, text: str, flags: int = re.S) -> str:
    """``re.sub`` that must match exactly once (fails loudly on template drift)."""
    new, n = re.subn(pattern, lambda _m: repl, text, count=0, flags=flags)
    if n != 1:
        raise ValueError(f"expected one match for {pattern!r}, found {n}")
    return new


def _replace_all(text: str, old: str, new: str) -> str:
    """``str.replace`` that must find ``old`` at least once (fails loudly on template drift)."""
    if old not in text:
        raise ValueError(f"expected at least one {old!r}")
    return text.replace(old, new)


def _placeholder(xml: str, ph: str) -> tuple[int, int]:
    """Span of the ``<p:sp>`` whose placeholder tag matches ``ph`` (a regex)."""
    spans = [
        (m.start(), m.end())
        for m in re.finditer(r"<p:sp>.*?</p:sp>", xml, re.S)
        if re.search(ph, m.group(0))
    ]
    if len(spans) != 1:
        raise ValueError(f"expected one placeholder matching {ph!r}, found {len(spans)}")
    return spans[0]


def _edit_placeholder(
    xml: str,
    ph: str,
    box: tuple[int, int, int, int] | None = None,
    lst_style: str | None = None,
    body_pr: str | None = None,
) -> str:
    """Replace geometry, list style, and/or body properties of one placeholder."""
    start, end = _placeholder(xml, ph)
    sp = xml[start:end]
    if box is not None:
        sp = _sub_once(r"<p:spPr/>|<p:spPr>.*?</p:spPr>", f"<p:spPr>{_xfrm(*box)}</p:spPr>", sp)
    if lst_style is not None:
        sp = _sub_once(r"<a:lstStyle/>|<a:lstStyle>.*?</a:lstStyle>", lst_style, sp)
    if body_pr is not None:
        sp = _sub_once(r"<a:bodyPr[^>]*/>|<a:bodyPr[^>]*>.*?</a:bodyPr>", body_pr, sp)
    return xml[:start] + sp + xml[end:]


def _append_shapes(xml: str, shapes: str) -> str:
    return _sub_once(r"</p:spTree>", shapes + "</p:spTree>", xml)


def _levels(sizes: list[int], extra: str = "") -> str:
    """List style with one ``defRPr`` size per level (sizes in 1/100 pt)."""
    lv = "".join(
        f'<a:lvl{i}pPr{extra}><a:defRPr sz="{s}"/></a:lvl{i}pPr>'
        for i, s in enumerate(sizes, start=1)
    )
    return f"<a:lstStyle>{lv}</a:lstStyle>"


def style_theme(xml: str) -> str:
    """Fonts and color scheme."""
    xml = re.sub(r'(<a:(?:major|minor)Font><a:latin typeface=")[^"]*(")', rf"\g<1>{FONT}\2", xml)
    scheme = (
        '<a:clrScheme name="AML"><a:dk1><a:srgbClr val="1A1A1A"/></a:dk1>'
        '<a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1B3A5C"/></a:dk2>'
        '<a:lt2><a:srgbClr val="F1F5F9"/></a:lt2>'
        '<a:accent1><a:srgbClr val="0072B2"/></a:accent1>'
        '<a:accent2><a:srgbClr val="D55E00"/></a:accent2>'
        '<a:accent3><a:srgbClr val="009E73"/></a:accent3>'
        '<a:accent4><a:srgbClr val="CC79A7"/></a:accent4>'
        '<a:accent5><a:srgbClr val="1B3A5C"/></a:accent5>'
        '<a:accent6><a:srgbClr val="595959"/></a:accent6>'
        '<a:hlink><a:srgbClr val="0072B2"/></a:hlink>'
        '<a:folHlink><a:srgbClr val="1B3A5C"/></a:folHlink></a:clrScheme>'
    )
    return _sub_once(r"<a:clrScheme .*?</a:clrScheme>", scheme, xml)


def style_master(xml: str) -> str:
    """Title and body text styles plus placeholder geometry on the slide master."""
    title_style = (
        '<p:titleStyle><a:lvl1pPr algn="l" defTabSz="342900" rtl="0" eaLnBrk="1" '
        'latinLnBrk="0" hangingPunct="1"><a:lnSpc><a:spcPct val="90000"/></a:lnSpc>'
        '<a:spcBef><a:spcPct val="0"/></a:spcBef><a:buNone/>'
        '<a:defRPr sz="2800" b="1" kern="1200"><a:solidFill><a:schemeClr val="tx2"/>'
        '</a:solidFill><a:latin typeface="+mj-lt"/><a:ea typeface="+mj-ea"/>'
        '<a:cs typeface="+mj-cs"/></a:defRPr></a:lvl1pPr></p:titleStyle>'
    )
    xml = _sub_once(r"<p:titleStyle>.*?</p:titleStyle>", title_style, xml)
    body = re.search(r"<p:bodyStyle>.*?</p:bodyStyle>", xml, re.S)
    if body is None:
        raise ValueError("slide master has no <p:bodyStyle>")
    b = body.group(0)
    sizes = (2000, 1800, 1600, 1400, 1400, 1400, 1400, 1400, 1400)
    for lvl, size in enumerate(sizes, start=1):
        # Only the first defRPr after each level's pPr: replace exactly one per level.
        b, n = re.subn(
            rf'(<a:lvl{lvl}pPr[^>]*>.*?<a:defRPr sz=")\d+(")',
            rf"\g<1>{size}\2",
            b,
            count=1,
            flags=re.S,
        )
        if n != 1:
            raise ValueError(f"slide master body style has no level {lvl} font size")
    # Navy bullets; slightly more space between items.
    b = _replace_all(
        b,
        '<a:buFont typeface="Arial"/>',
        '<a:buClr><a:schemeClr val="tx2"/></a:buClr><a:buFont typeface="Arial"/>',
    )
    b = _replace_all(
        b,
        '<a:spcBef><a:spcPct val="20000"/></a:spcBef>',
        '<a:spcBef><a:spcPts val="600"/></a:spcBef>',
    )
    xml = xml.replace(body.group(0), b)
    xml = _edit_placeholder(
        xml,
        r'<p:ph type="title"/>',
        box=(LEFT, TITLE_Y, WIDTH, TITLE_H),
        body_pr='<a:bodyPr vert="horz" lIns="0" tIns="45720" rIns="0" '
        'bIns="45720" rtlCol="0" anchor="b"><a:normAutofit/>'
        "</a:bodyPr>",
    )
    return _edit_placeholder(
        xml,
        r'<p:ph type="body" idx="1"/>',
        box=(LEFT, BODY_Y, WIDTH, BODY_H),
        body_pr='<a:bodyPr vert="horz" lIns="0" tIns="45720" rIns="0" '
        'bIns="45720" rtlCol="0"><a:normAutofit/></a:bodyPr>',
    )


def style_title_slide(xml: str) -> str:
    """Left-aligned title slide with a course label, accent bar, and credits."""
    title_y, title_h = int(1.55 * EMU_PER_IN), int(1.05 * EMU_PER_IN)
    xml = _edit_placeholder(
        xml,
        r'<p:ph type="ctrTitle"/>',
        box=(LEFT, title_y, WIDTH, title_h),
        lst_style='<a:lstStyle><a:lvl1pPr algn="l"><a:defRPr sz="3600"/></a:lvl1pPr></a:lstStyle>',
        body_pr='<a:bodyPr lIns="0" rIns="0" anchor="b"><a:normAutofit/></a:bodyPr>',
    )
    sub_lvl = "".join(
        f'<a:lvl{i}pPr marL="0" indent="0" algn="l"><a:buNone/>'
        f'<a:defRPr sz="{1800 if i > 1 else 2000}"><a:solidFill><a:srgbClr val="{MUTED}"/>'
        f"</a:solidFill></a:defRPr></a:lvl{i}pPr>"
        for i in range(1, 10)
    )
    xml = _edit_placeholder(
        xml,
        r'<p:ph type="subTitle" idx="1"/>',
        box=(LEFT, int(2.8 * EMU_PER_IN), WIDTH, int(1.6 * EMU_PER_IN)),
        lst_style=f"<a:lstStyle>{sub_lvl}</a:lstStyle>",
        body_pr='<a:bodyPr lIns="0" rIns="0"><a:normAutofit/></a:bodyPr>',
    )
    shapes = (
        _textbox(
            101,
            "Course label",
            (LEFT, int(1.2 * EMU_PER_IN), WIDTH, int(0.35 * EMU_PER_IN)),
            _run("Applied Machine Learning", 1600, "0072B2", bold=True),
        )
        + _bar(102, title_y + title_h + int(0.06 * EMU_PER_IN))
        + _textbox(
            103,
            "Credits",
            (LEFT, int(4.75 * EMU_PER_IN), TITLE_TEXT_W, int(0.5 * EMU_PER_IN)),
            _run(CREDITS, 1000, MUTED),
            anchor="b",
        )
    )
    return _append_shapes(xml, shapes)


def style_content_layout(xml: str, name: str) -> str:
    """Bar, footer, and geometry for the content layouts pandoc uses."""
    if name == "Two Content":
        half = (WIDTH - int(0.3 * EMU_PER_IN)) // 2
        for idx, x in (("1", LEFT), ("2", LEFT + half + int(0.3 * EMU_PER_IN))):
            xml = _edit_placeholder(
                xml,
                rf'<p:ph sz="half" idx="{idx}"/>',
                box=(x, BODY_Y, half, BODY_H),
                lst_style=_levels([1800, 1600, 1400, 1400, 1400, 1400, 1400, 1400, 1400]),
            )
    elif name == "Comparison":
        half = (WIDTH - int(0.3 * EMU_PER_IN)) // 2
        head_h = int(0.45 * EMU_PER_IN)
        for head, body, x in (
            (r'<p:ph type="body" idx="1"/>', r'<p:ph sz="half" idx="2"/>', LEFT),
            (
                r'<p:ph type="body" sz="quarter" idx="3"/>',
                r'<p:ph sz="quarter" idx="4"/>',
                LEFT + half + int(0.3 * EMU_PER_IN),
            ),
        ):
            xml = _edit_placeholder(xml, head, box=(x, BODY_Y, half, head_h))
            xml = _edit_placeholder(
                xml,
                body,
                box=(x, BODY_Y + head_h, half, BODY_H - head_h),
                lst_style=_levels([1800, 1600, 1400, 1400, 1400, 1400, 1400, 1400, 1400]),
            )
    elif name == "Content with Caption":
        # pandoc: text (body idx=2) followed by an image or table (content idx=1).
        text_w = int(3.5 * EMU_PER_IN)
        gap = int(0.25 * EMU_PER_IN)
        xml = _edit_placeholder(
            xml,
            r'<p:ph type="title"/>',
            box=(LEFT, TITLE_Y, WIDTH, TITLE_H),
            lst_style="<a:lstStyle/>",
            body_pr="<a:bodyPr/>",
        )
        xml = _edit_placeholder(
            xml,
            r'<p:ph type="body" sz="half" idx="2"/>',
            box=(LEFT, BODY_Y, text_w, BODY_H),
            lst_style=_levels([1800, 1600, 1400, 1400, 1400, 1400, 1400, 1400, 1400]),
        )
        xml = _edit_placeholder(
            xml,
            r'<p:ph idx="1"/>',
            box=(LEFT + text_w + gap, BODY_Y, WIDTH - text_w - gap, BODY_H),
            lst_style=_levels([1800, 1600, 1400, 1400, 1400, 1400, 1400, 1400, 1400]),
        )
    return _append_shapes(xml, _bar(101, BAR_Y) + _footer_shapes(102))


def build(out: Path) -> None:
    """Write the styled reference template to ``out``.

    Args:
        out: Destination ``.pptx`` path.

    Raises:
        subprocess.CalledProcessError: If pandoc cannot export its default reference doc.
    """
    default = subprocess.run(
        ["pandoc", "--print-default-data-file", "reference.pptx"], capture_output=True, check=True
    ).stdout
    out.parent.mkdir(parents=True, exist_ok=True)
    content_layouts = {
        "Title and Content",
        "Two Content",
        "Comparison",
        "Title Only",
        "Content with Caption",
    }
    with (
        zipfile.ZipFile(io.BytesIO(default)) as zin,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout,
    ):
        for item in zin.infolist():
            data = zin.read(item.filename)
            name = item.filename
            if name == "ppt/theme/theme1.xml":
                data = style_theme(data.decode()).encode()
            elif name == "ppt/slideMasters/slideMaster1.xml":
                data = style_master(data.decode()).encode()
            elif name.startswith("ppt/slideLayouts/slideLayout") and name.endswith(".xml"):
                xml = data.decode()
                m = re.search(r'<p:cSld name="([^"]*)"', xml)
                layout = m.group(1) if m else ""
                if layout == "Title Slide":
                    xml = style_title_slide(xml)
                elif layout in content_layouts:
                    xml = style_content_layout(xml, layout)
                data = xml.encode()
            zout.writestr(item, data)
    theme = out.with_name("aml-code.theme")
    theme.write_text(code_theme(), encoding="utf-8")
    print(f"wrote {out} and {theme}")


def main() -> None:
    """Parse arguments and build the template."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output .pptx path")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
