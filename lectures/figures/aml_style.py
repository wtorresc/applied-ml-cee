"""Shared matplotlib style for lecture figures.

White background, near-black text, and the four Okabe-Ito hues that keep >= 3:1 contrast
on white (WCAG 2.1, 1.4.11). Every series also gets its own marker and line style, so no
figure depends on color alone. Text uses Atkinson Hyperlegible when TeX Live ships it,
matching the slides; otherwise DejaVu Sans.
"""

from __future__ import annotations

import logging
import subprocess
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import font_manager
from matplotlib.colors import LightSource, Normalize
from matplotlib.patheffects import withStroke
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

PALETTE: list[str] = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
MARKERS: list[str] = ["o", "s", "^", "D"]
LINESTYLES: list[str] = ["-", "--", "-.", ":"]
# fontTools warns about the OTF head-table timestamps; the fonts are fine.
logging.getLogger("fontTools").setLevel(logging.ERROR)

INK = "#1A1A1A"
MUTED = "#595959"
NAVY = "#1B3A5C"
# A white outline that keeps a label readable over lines and surfaces: path_effects=HALO.
HALO: list[withStroke] = [withStroke(linewidth=3, foreground="white")]

# Figures are drawn at the size they occupy on a 10 x 5.625 in slide (template/aml-reference.pptx),
# so pandoc does not rescale them and 14 pt labels stay ~19 pt on a 13.33 in projector slide.
FULL: tuple[float, float] = (9.0, 3.6)  # whole content area
SIDE: tuple[float, float] = (5.2, 3.6)  # right of a text column (Content with Caption)
DPI = 250


def _atkinson_family() -> str | None:
    """Register Atkinson Hyperlegible Next from TeX Live, if present.

    Returns:
        The font family name, or ``None`` when the font files are not found.
    """
    try:
        found = subprocess.run(
            ["kpsewhich", "AtkinsonHyperlegibleNext-Regular.otf"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except FileNotFoundError:
        return None
    if not found:
        return None
    font_dir = Path(found).parent
    for weight in ("Regular", "Bold", "RegularItalic"):
        path = font_dir / f"AtkinsonHyperlegibleNext-{weight}.otf"
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    return font_manager.FontProperties(fname=found).get_name()


def apply() -> None:
    """Apply the lecture style (fonts, colours, sizes). Seeding is ``common.seed_all``."""
    family = _atkinson_family()
    if family is None:
        warnings.warn(
            "Atkinson Hyperlegible Next not found (kpsewhich); figures fall back to "
            "DejaVu Sans and will not match the slide font",
            stacklevel=2,
        )
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ([family] if family else []) + ["DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.titlecolor": NAVY,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.prop_cycle": cycler(color=PALETTE) + cycler(linestyle=LINESTYLES),
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
            "legend.frameon": False,
        }
    )


def style3d(ax: Axes3D, *, elev: float = 28.0, azim: float = -125.0) -> None:
    """Style a 3D axes: light-grey panes and grid, muted ticks, and the viewing angle.

    Args:
        ax: Axes made with ``projection="3d"``.
        elev: Viewing elevation, degrees.
        azim: Viewing azimuth, degrees.
    """
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.965, 0.965, 0.965, 1.0))
        axis.pane.set_edgecolor("#D0D0D0")
    with mpl.rc_context({"grid.color": "#DDDDDD", "grid.linewidth": 0.6}):
        ax.grid(True)
    ax.tick_params(labelsize=10, pad=1)
    ax.view_init(elev=elev, azim=azim)


def surface3d(
    ax: Axes3D,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    cmap: str = "cividis",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 1.0,
) -> None:
    """A surface coloured by height and lit from the upper left, so folds and valleys read.

    Args:
        ax: Axes made with ``projection="3d"``.
        X: Grid x coordinates, shape ``(ny, nx)``.
        Y: Grid y coordinates, shape ``(ny, nx)``.
        Z: Heights, shape ``(ny, nx)``.
        cmap: Colour map of the heights.
        vmin: Height at the bottom of the colour map (default: ``Z``'s minimum).
        vmax: Height at the top of the colour map (default: ``Z``'s maximum).
        alpha: Opacity.
    """
    norm = Normalize(Z.min() if vmin is None else vmin, Z.max() if vmax is None else vmax)
    # Hill-shading strength. LightSource differentiates Z per grid index (its dx = dy = 1), so
    # this exaggeration scales with X's units over the grid size, not with true slopes: it is
    # a tuned look, kept because every 3D figure in the decks was checked with it.
    span = np.ptp(X) / max(np.ptp(Z), 1e-12)
    rgb = LightSource(azdeg=315, altdeg=40).shade(
        Z, cmap=plt.get_cmap(cmap), norm=norm, blend_mode="soft", vert_exag=span
    )
    ax.plot_surface(
        X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False,
        shade=False, alpha=alpha,
    )  # fmt: skip


def save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    """Save ``fig`` as ``<out_dir>/<name>.png`` (``DPI``) and close it.

    Args:
        fig: Figure to save.
        out_dir: Output directory (created if missing).
        name: File stem.

    Returns:
        Path of the written PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.05, metadata={"Software": None})
    plt.close(fig)
    return path


def thumbnails(out_dir: Path, names: tuple[str, ...], out_name: str = "remember") -> Path:
    """Paste four of a lecture's figures into a 2 x 2 grid (the "Things to remember" picture).

    Args:
        out_dir: The figure directory holding ``<name>.png`` for each name.
        names: Four figure stems, in reading order.
        out_name: Stem of the grid image.

    Returns:
        Path of the written PNG.
    """
    if len(names) != 4:
        raise ValueError(f"a 2 x 2 grid needs four figures, got {len(names)}: {names}")
    ims = [Image.open(out_dir / f"{n}.png").convert("RGB") for n in names]
    w = 900
    thumbs = [im.resize((w, round(im.height * w / im.width))) for im in ims]
    h = max(t.height for t in thumbs)
    grid = Image.new("RGB", (2 * w + 30, 2 * h + 30), "white")
    for i, t in enumerate(thumbs):
        grid.paste(t, ((i % 2) * (w + 30), (i // 2) * (h + 30)))
    path = out_dir / f"{out_name}.png"
    grid.save(path, optimize=True)
    return path
