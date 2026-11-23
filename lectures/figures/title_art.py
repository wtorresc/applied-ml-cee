"""Title art for the lecture decks: abstract paintings drawn in code, one per lecture.

Each picture takes its subject from the lecture's title and draws it with the lecture's own
mathematics (a network's kink lines, a loss curve), in the palette and forms of Peruvian abstract
painting: dark glazed grounds, stepped Andean forms, portals and totems, a dark sun with a
luminous rim. An homage to Fernando de Szyszlo (1925-2017); no painting of his is copied.

A canvas is an RGB float array. Shapes are soft-edged masks painted over it as translucent glazes
whose colour varies with a smooth noise field, so surfaces look brushed rather than flat.

    canvas = Canvas(seed=14)
    canvas.ground(("#0f0a1c", "#3b0a12"))
    canvas.glaze(canvas.polygon([(0.2, 0.1), (0.5, 0.1), (0.5, 0.7)]), "#7a1e1e", 0.8)
    canvas.save(out_dir / "title_art.png")
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

# Width x height in pixels: the right panel of a 10 x 5.625 in title slide, 4 in wide, at 400 dpi
SIZE: tuple[int, int] = (1600, 2250)

Point = tuple[float, float]


def rgb(hex_color: str) -> np.ndarray:
    """A ``#rrggbb`` colour as floats in [0, 1]."""
    h = hex_color.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


class Canvas:
    """A raster painting. Coordinates are in units of the width: x in [0, 1], y in [0, H/W]."""

    def __init__(self, seed: int, size: tuple[int, int] = SIZE) -> None:
        """A black canvas and a random generator for its textures."""
        self.w, self.h = size
        self.rng = np.random.default_rng(seed)
        self.img = np.zeros((self.h, self.w, 3))
        self._texture: np.ndarray | None = None

    @property
    def top(self) -> float:
        """The y coordinate of the top edge."""
        return self.h / self.w

    def _px(self, p: Point) -> tuple[float, float]:
        """Canvas units to pixels (y up in units, down in pixels)."""
        return p[0] * self.w, self.h - p[1] * self.w

    def noise(self, scale: float, octaves: int = 3) -> np.ndarray:
        """Smooth noise in [-1, 1]: blurred random fields, ``scale`` the coarsest blur in px."""
        field = np.zeros((self.h, self.w))
        for k in range(octaves):
            s = scale / 2**k
            layer = gaussian_filter(self.rng.normal(size=(self.h, self.w)), (s, s * 1.6))
            field += layer / (np.abs(layer).max() + 1e-12) / 2**k
        return field / (np.abs(field).max() + 1e-12)

    def ground(self, colors: Sequence[str], grain: float = 0.06) -> None:
        """A vertical gradient through ``colors`` (bottom to top) with a brushed texture."""
        t = np.linspace(0, 1, self.h)[::-1][:, None]
        stops = np.linspace(0, 1, len(colors))
        cols = np.stack([rgb(c) for c in colors])
        base = np.stack([np.interp(t[:, 0], stops, cols[:, i]) for i in range(3)], axis=-1)
        self.img = np.repeat(base[:, None, :], self.w, axis=1)
        self.img *= 1 + grain * self.noise(40)[..., None]
        yy, xx = np.mgrid[0 : self.h, 0 : self.w]
        r = np.hypot((xx - self.w / 2) / self.w, (yy - self.h / 2) / self.h)
        self.img *= (1 - 0.55 * np.clip(r - 0.25, 0, 1))[..., None]  # vignette

    def polygon(self, points: Sequence[Point], soft: float = 2.0) -> np.ndarray:
        """Mask of a filled polygon, edges softened by ``soft`` px."""
        im = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(im).polygon([self._px(p) for p in points], fill=255)
        return gaussian_filter(np.asarray(im, dtype=np.float64) / 255.0, soft)

    def disc(self, center: Point, radius: float, soft: float = 3.0) -> np.ndarray:
        """Mask of a disc."""
        cx, cy = self._px(center)
        yy, xx = np.mgrid[0 : self.h, 0 : self.w]
        m = (np.hypot(xx - cx, yy - cy) <= radius * self.w).astype(np.float64)
        return gaussian_filter(m, soft)

    def lines(
        self, segments: Sequence[Sequence[Point]], width: float, soft: float = 1.0
    ) -> np.ndarray:
        """Mask of polylines, ``width`` in px."""
        im = Image.new("L", (self.w, self.h), 0)
        draw = ImageDraw.Draw(im)
        for seg in segments:
            draw.line([self._px(p) for p in seg], fill=255, width=max(1, round(width)))
        return gaussian_filter(np.asarray(im, dtype=np.float64) / 255.0, soft)

    def glaze(self, mask: np.ndarray, color: str, alpha: float, texture: float = 0.12) -> None:
        """Paint ``color`` through ``mask`` at opacity ``alpha``, its value varying with noise."""
        if self._texture is None:  # one texture field, shifted for each glaze (noise is slow)
            self._texture = self.noise(25, 2)
        shift = (int(self.rng.integers(self.h)), int(self.rng.integers(self.w)))
        tex = np.roll(self._texture, shift, axis=(0, 1))
        c = rgb(color)[None, None, :] * (1 + texture * tex[..., None])
        a = (alpha * mask)[..., None]
        self.img = self.img * (1 - a) + np.clip(c, 0, 1) * a

    def glow(self, mask: np.ndarray, color: str, strength: float, spread: float = 18.0) -> None:
        """Add light: the mask, plus a wide blurred halo, in ``color`` (additive)."""
        halo = gaussian_filter(mask, spread)
        light = (mask + 1.5 * halo / (halo.max() + 1e-12))[..., None] * rgb(color)[None, None, :]
        self.img = self.img + strength * light

    def save(self, path: Path) -> Path:
        """Write the PNG, with a fine film grain, and return its path."""
        out = np.clip(self.img * (1 + 0.025 * self.rng.normal(size=self.img.shape)), 0, 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.rint(out * 255).astype(np.uint8)).save(path, optimize=True)
        return path

    def save_title_art(self, out_dir: Path, alt: str) -> Path:
        """Write the deck's title picture and its alt text, the pair the deck build reads.

        ``tools/postprocess_pptx.py --title-art`` places ``title_art.png`` on the title slide
        and reads its alt text from ``title_art.txt`` beside it.

        Args:
            out_dir: The lecture's figure directory.
            alt: The picture's alt text.

        Returns:
            The PNG's path.
        """
        path = self.save(out_dir / "title_art.png")
        (out_dir / "title_art.txt").write_text(alt.strip() + "\n", encoding="utf-8")
        return path


def stepped(x0: float, y0: float, width: float, height: float, steps: int) -> list[Point]:
    """An Andean stepped form: a pyramid of ``steps`` terraces on the base (x0, y0)-(x0+width)."""
    pts: list[Point] = [(x0, y0)]
    dx, dy = width / (2 * steps), height / steps
    for k in range(steps):
        pts += [(x0 + k * dx, y0 + (k + 1) * dy), (x0 + (k + 1) * dx, y0 + (k + 1) * dy)]
    for k in range(steps, 0, -1):
        x = x0 + width - (k - 1) * dx
        pts += [(x, y0 + k * dy), (x, y0 + (k - 1) * dy)]
    return pts
