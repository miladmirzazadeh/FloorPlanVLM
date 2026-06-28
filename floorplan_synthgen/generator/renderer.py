"""Renderer: DXF -> PNG via ezdxf's matplotlib drawing add-on.

Produces a monochrome (black-on-white) architectural-looking image and the
exact model->pixel transform that the exporter needs. The image is sized so
that:

* a standard 900 mm door is at least 30 px wide,
* the shortest side is at least 800 px,
* the longest side is at most 2000 px.

The mapping is made exact (not left to matplotlib autoscale): one axes fills
the whole canvas, its data limits are the geometry bounds plus a margin, and
the figure aspect equals the data aspect - so the returned ``ModelTransform``
reproduces pixel positions to sub-pixel accuracy (verified by tests).
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

import ezdxf
from ezdxf import bbox
from ezdxf.addons.drawing import RenderContext, Frontend, config
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

from . import style
from .geometry import ModelTransform

FIG_DPI = 100
DOOR_REF_MM = 900.0
DOOR_REF_MIN_PX = 30.0
MIN_SHORT_SIDE = 800
MAX_LONG_SIDE = 2000


def _solve_ppm(range_x, range_y, denom, dpi):
    """Pixels-per-mm honouring the door / shortest / longest constraints."""
    ppm = dpi / (denom * 25.4)
    ppm = max(ppm, DOOR_REF_MIN_PX / DOOR_REF_MM)  # 900 mm door >= 30 px
    w = range_x * ppm
    h = range_y * ppm
    short = min(w, h)
    if short < MIN_SHORT_SIDE:
        ppm *= MIN_SHORT_SIDE / short
    w = range_x * ppm
    h = range_y * ppm
    long_side = max(w, h)
    if long_side > MAX_LONG_SIDE:
        ppm *= MAX_LONG_SIDE / long_side
    return ppm


def render(dxf_path: str, png_path: str, *, plan=None, dpi: int = 150,
           line_weight_style: str = "standard", monochrome: bool = True,
           noise_std: float = 0.0) -> Tuple[int, int, ModelTransform]:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    ext = bbox.extents(msp)
    if ext is None or not ext.has_data:
        raise ValueError(f"{dxf_path}: nothing to render")
    minx, miny = ext.extmin.x, ext.extmin.y
    maxx, maxy = ext.extmax.x, ext.extmax.y

    raw_span = max(maxx - minx, maxy - miny)
    margin = 0.03 * raw_span + 50.0
    xlim0, ylim0 = minx - margin, miny - margin
    range_x = (maxx + margin) - xlim0
    range_y = (maxy + margin) - ylim0

    denom = style.scale_denominator(plan.scale) if plan is not None else 100
    ppm = _solve_ppm(range_x, range_y, denom, dpi)
    img_w = max(1, round(range_x * ppm))
    img_h = max(1, round(range_y * ppm))

    fig = Figure(figsize=(img_w / FIG_DPI, img_h / FIG_DPI), dpi=FIG_DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.patch.set_facecolor("white")

    cfg = config.Configuration().with_changes(
        background_policy=config.BackgroundPolicy.WHITE,
        color_policy=(config.ColorPolicy.BLACK if monochrome else config.ColorPolicy.COLOR),
        lineweight_policy=config.LineweightPolicy.ABSOLUTE,
        lineweight_scaling=style.LINE_WEIGHT_STYLES.get(line_weight_style, 1.0),
        min_lineweight=0.0,
    )
    ctx = RenderContext(doc)
    backend = MatplotlibBackend(ax)
    Frontend(ctx, backend, config=cfg).draw_layout(msp, finalize=False)

    ax.set_xlim(xlim0, xlim0 + range_x)
    ax.set_ylim(ylim0, ylim0 + range_y)
    ax.set_aspect("equal")
    fig.savefig(png_path, dpi=FIG_DPI, facecolor="white")

    # actual size wins (matplotlib may round by +/-1 px)
    with Image.open(png_path) as im:
        im = im.convert("RGB")
        act_w, act_h = im.size
        if noise_std > 0:
            arr = np.asarray(im).astype(np.float32)
            arr += np.random.default_rng(0).normal(0, noise_std, arr.shape)
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        im.save(png_path)

    transform = ModelTransform(
        rotation_deg=(plan.rotation_deg if plan is not None else 0.0),
        rot_origin=(plan.rot_origin if plan is not None else (0.0, 0.0)),
        xlim0=xlim0, ylim0=ylim0, range_x=range_x, range_y=range_y,
        img_w=act_w, img_h=act_h,
    )
    return act_w, act_h, transform
