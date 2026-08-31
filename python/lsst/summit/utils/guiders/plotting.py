# This file is part of summit_utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.stats import mad_std
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from lsst.summit.utils.utils import RobustFitter

from .metrics import DEFAULT_RESIDUAL_SCALE
from lsst.utils.plotting.figures import make_figure

if TYPE_CHECKING:
    from .reading import GuiderData

__all__ = ["GuiderPlotter"]

LIGHT_BLUE = "#6495ED"
DEFAULT_CUTOUT_SIZE = 30


@dataclass
class StarInfo:
    """Container for star position information in a detector panel.

    Attributes
    ----------
    hasData : bool
        Whether valid star measurement data exists.
    refCenter : tuple[float, float]
        Reference center (median xroi_ref, yroi_ref) for the fixed frame.
    starCenter : tuple[float, float]
        Actual star centroid (xroi, yroi) for the current stamp.
    """

    hasData: bool
    refCenter: tuple[float, float]
    starCenter: tuple[float, float]

    @classmethod
    def from_stars_df(cls, starsDf: pd.DataFrame, detName: str, stampNum: int) -> StarInfo:
        """Create StarInfo from stars DataFrame.

        Parameters
        ----------
        starsDf : pandas.DataFrame
            Star measurements table.
        detName : str
            Detector name.
        stampNum : int
            Stamp index; negative implies stacked image.

        Returns
        -------
        StarInfo
            Star position information.

        Raises
        ------
        ValueError
            If no rows exist for the detector.
        """
        mask1 = starsDf["detector"] == detName
        if not mask1.any():
            raise ValueError(f"No rows for detector {detName!r}")

        # Reference center (fixed frame)
        refX = float(starsDf.loc[mask1, "xroi_ref"].median())
        refY = float(starsDf.loc[mask1, "yroi_ref"].median())
        refCenter = (refX, refY)

        mask2 = starsDf["stamp"] == stampNum
        mask = mask1 & mask2

        # Fallback: no row for that stamp or stacked request
        if (not mask.any()) or (stampNum < 0):
            return cls(hasData=True, refCenter=refCenter, starCenter=refCenter)

        row = starsDf.loc[mask, ["xroi", "yroi"]].iloc[0]
        starCenter = (float(row["xroi"]), float(row["yroi"]))
        return cls(hasData=True, refCenter=refCenter, starCenter=starCenter)

    @classmethod
    def from_image_center(cls, shape: tuple[int, int]) -> StarInfo:
        """Create StarInfo centered on image.

        Parameters
        ----------
        shape : tuple[int, int]
            Image shape (height, width).

        Returns
        -------
        StarInfo
            Star position centered on image.
        """
        center = (shape[1] / 2.0, shape[0] / 2.0)
        return cls(hasData=False, refCenter=center, starCenter=center)


STRIP_PLOT_KWARGS: dict[str, dict] = {
    "centroidAltAz": {
        "ylabel": "Centroid Offset [arcsec]",
        "unit": "arcsec",
        "col": ["dalt", "daz"],
        "title": "Alt/Az Centroid Offsets",
    },
    "centroidPixel": {
        "ylabel": "Centroid Offset [pixels]",
        "col": ["dx", "dy"],
        "unit": "pixels",
        "title": "CCD Pixel Centroid Offsets",
    },
    "flux": {
        "ylabel": "Magnitude Offset [mmag]",
        "col": ["magoffset"],
        "unit": "mmag",
        "scale": 1,  # keep scale to mmag
        "title": "Flux Magnitude Offsets",
    },
    "rotator": {
        "ylabel": "Rotation Angle [arcsec]",
        "col": ["dtheta"],
        "unit": "arcsec",
        "scale": 1.0,
        "title": "Rotation Angle",
    },
    "ellip": {
        "ylabel": "Ellipticity",
        "col": ["e1_altaz", "e2_altaz"],
        "unit": "",
        "title": "",
    },
    "psf": {
        "ylabel": "PSF FWHM [arcsec]",
        "col": ["fwhm"],
        "scale": 1.0,
        "unit": "arcsec",
        "title": "PSF FWHM",
    },
}


def _getWriter(filename: str) -> str:
    """
    Get the appropriate writer for saving animations based on file extension.

    Parameters
    ----------
    extension : `str`
        Filename to determine the writer type.

    Returns
    -------
    writer : `str`
        The name of the writer to use.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    _, extension = os.path.splitext(filename)
    match extension.lower():
        case ".gif":
            return "pillow"
        case ".mp4":
            return "ffmpeg"
        case _:
            raise ValueError(f"Unsupported file extension for {filename}: {extension}")


@dataclass(frozen=True)
class MosaicLayout:
    """Corner-paired guider mosaic layout.

    Each corner raft's two SG panels sit together in that raft's focal-plane
    corner (R00 lower-left, R04 lower-right, R40 upper-left, R44 upper-right),
    with their 2" circles tangent at a point on the diagonal (45/135/225/315
    deg). The arrangement is symmetric under x->-x and y->-y, as the focal plane
    is. SG0/SG1 within a corner keep their physical relative orientation.

    Positions are figure fractions; ``radius`` is the panel half-size (= the 2"
    circle radius) and ``pairDist`` the diagonal distance from center to each
    pair's tangent point. Panels are ``2*radius`` squares so the inscribed
    circles of a pair (centers 2*radius apart) touch exactly.
    """

    radius: float = 0.135
    pairDist: float = 0.27
    # Where each detector's name sits, all OUTSIDE the 2" circle: 'ot' = above
    # the circle (default). The top pair reaches the figure top, so those label
    # to the side instead ('l'/'r').
    labelCorners: dict[str, str] = field(
        default_factory=lambda: {
            "R40_SG1": "l", "R44_SG0": "r",
            "R44_SG1": "ot", "R04_SG0": "ot", "R40_SG0": "ot",
            "R00_SG1": "ot", "R00_SG0": "ot", "R04_SG1": "ot",
        }
    )

    @property
    def positions(self) -> dict[str, tuple[float, float]]:
        """Center (figure-fraction x, y) of each guider panel."""
        r, p, cx, cy = self.radius, self.pairDist, 0.5, 0.5
        rr = r / np.sqrt(2)  # perpendicular offset so the pair touches on the diagonal
        return {
            # top-right (R44), tangent at (+p,+p)
            "R44_SG0": (cx + p - rr, cy + p + rr), "R44_SG1": (cx + p + rr, cy + p - rr),
            # top-left (R40), tangent at (-p,+p)
            "R40_SG1": (cx - p + rr, cy + p + rr), "R40_SG0": (cx - p - rr, cy + p - rr),
            # bottom-left (R00), tangent at (-p,-p)
            "R00_SG1": (cx - p - rr, cy - p + rr), "R00_SG0": (cx - p + rr, cy - p - rr),
            # bottom-right (R04), tangent at (+p,-p)
            "R04_SG0": (cx + p + rr, cy - p + rr), "R04_SG1": (cx + p - rr, cy - p - rr),
        }

    def build(
        self, *, figsize: tuple[float, float] = (12, 12), **_: Any
    ) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        """Build the Agg-backed figure with explicitly placed panel axes.

        Parameters
        ----------
        figsize : `tuple[float, float]`, optional
            Figure size in inches (width, height); keep it square so the circles
            render round.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The created figure.
        axs : `dict[str, matplotlib.axes.Axes]`
            Mapping from panel label (guider names + 'center', 'arrow') to axes.
        """
        fig = make_figure(figsize=figsize, constrained_layout=False)
        r = self.radius
        axs: dict[str, plt.Axes] = {}
        for name, (x, y) in self.positions.items():
            axs[name] = fig.add_axes((x - r, y - r, 2 * r, 2 * r))
        # exposure metadata (center) with the Alt/Az arrow directly beneath it
        axs["center"] = fig.add_axes((0.34, 0.45, 0.32, 0.14))
        axs["arrow"] = fig.add_axes((0.43, 0.30, 0.14, 0.13))
        return fig, axs


class GuiderPlotter:
    # The MARKERS lists are used for plotting different detector
    MARKERS: list[str] = ["o", "x", "+", "*", "^", "v", "s", "p"]

    def __init__(self, guiderData: GuiderData, starsDf: pd.DataFrame | None = None) -> None:
        self.log = logging.getLogger(__name__)
        self.expId = guiderData.expid
        self.layout: MosaicLayout = MosaicLayout()

        self.guiderData = guiderData

        # Some metadata information
        self.expTime = self.guiderData.guiderDurationSec
        self.camRotAngle = self.guiderData.camRotAngle

        self.starsDf: pd.DataFrame = pd.DataFrame()
        if starsDf is not None:
            self.starsDf = starsDf.loc[starsDf["expid"] == self.expId].reset_index(drop=True)
        self.withStars: bool = not self.starsDf.empty

        # set seaborn style
        sns.set_style("white")
        sns.set_context("talk", font_scale=0.8)

    def setupFigure(self, figsize: tuple[float, float] = (12, 12)) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        """
        Create a figure and axes using the guider mosaic layout.

        Parameters
        ----------
        figsize : `tuple[float, float]`, optional
            Figure size in inches (width, height).

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The created figure.
        axs : `dict[str, matplotlib.axes.Axes]`
            Mapping of panel name to axes.
        """
        fig, axs = self.layout.build(figsize=figsize)
        return fig, axs

    def stripPlot(
        self, plotType: str = "centroidAltAz", saveAs: str | None = None, coveragePct: int = 68,
        residualScale: float = DEFAULT_RESIDUAL_SCALE,
    ) -> plt.Figure:
        """
        Plot time-series strip plot for a chosen metric.

        This renders one or more panels vs elapsed time, fits a robust linear
        trend, annotates slope/significance/scatter, and draws reference zero
        lines or median PSF as needed.

        Parameters
        ----------
        plotType : `str`, optional
            Metric key in `STRIP_PLOT_KWARGS` (e.g., 'centroidAltAz').
        saveAs : `str`, optional
            If not None, path to save the figure.
        coveragePct : `int`, optional
            Central percentile span used to derive y-limits (e.g. 68 -> 16–84).

        Returns
        -------
        stripFig : `matplotlib.figure.Figure`
            Figure containing the strip plot panels.
        """
        if self.starsDf.empty:
            raise ValueError("starsDf is empty. No data to make a stripPlot.")

        cfg = STRIP_PLOT_KWARGS.get(plotType)
        if cfg is None:
            raise ValueError(f"Unknown plotType: {plotType}")
        # from here, tell mypy it’s a dict[str, Any]
        cfg = cast(dict[str, Any], cfg)

        # get alt/az
        alt = self.guiderData.alt
        az = self.guiderData.az

        n = len(cfg["col"])
        fig = make_figure(figsize=(8 * n, 6))
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        axes = fig.subplots(nrows=1, ncols=n, sharex=True, sharey=True)
        axes = np.atleast_1d(axes)

        cols = cfg["col"]
        scale = float(cfg.get("scale", 1.0))
        unit = cfg.get("unit", "")
        expTime = float(self.expTime)

        df = self.starsDf.loc[self.starsDf["stamp"] > 0, ["elapsed_time", "detector", *cols]].copy()

        q1, q3 = (100 - coveragePct) / 2, 100 - (100 - coveragePct) / 2
        yvals = (df[cols].to_numpy(dtype=float) * scale).ravel()
        p16, p84 = np.nanpercentile(yvals, [q1, q3])
        sigma = mad_std(yvals, ignore_nan=True)
        ylims = (p16 - 2.5 * sigma, p84 + 2.5 * sigma)

        def _zero(ax, c):
            label = {
                "daz": f"Az: {az:0.5f} deg",
                "dalt": f"Alt: {alt:0.5f} deg",
                "dx": "CCD X",
                "dy": "CCD Y",
                "e1_altaz": "e1",
                "e2_altaz": "e2",
                "magoffset": "Magnitude Offset",
                "dtheta": "Rotation Offset",
            }.get(c, "")
            ax.axhline(0 if c != "fwhm" else np.nanmedian(df[c] * scale), color="grey", ls="--", label=label)

        for i, (ax, c) in enumerate(zip(axes, cols)):
            _zero(ax, c)
            # 5-sigma inlier band on the residual scatter (not the raw,
            # trend-inflated spread); the old default rejected too many points.
            fitter = RobustFitter(residualScale=residualScale, residualFromFit=True)
            res = fitter.fit(x=np.asarray(df["elapsed_time"].values), y=(df[c].values * scale))
            txt = (
                f"Slope: {expTime * res.slope:.2f} {unit}/exposure\n"
                f"Significance: {abs(res.slopeTValue):.1f} σ\n"
                f"scatter: {res.scatter:.3f} {unit}\n"
            )
            ax.text(
                0.02,
                0.98,
                txt,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.3"),
            )
            fitter.plotBestFit(ax=ax, color=LIGHT_BLUE, label="", lw=2)

            for j, det in enumerate(df["detector"].unique()):
                m = self.MARKERS[j % len(self.MARKERS)]
                msk = df["detector"].eq(det)
                ax.scatter(
                    df.loc[msk, "elapsed_time"],
                    df.loc[msk, c] * scale,
                    color="lightgrey",
                    alpha=0.6,
                    marker=m,
                    label=f"{det}" if i == 0 else "",
                )
                out = msk & fitter.outlierMask
                ax.scatter(
                    df.loc[out, "elapsed_time"], df.loc[out, c] * scale, color=LIGHT_BLUE, alpha=0.2, marker=m
                )

            if i == 0:
                ax.set_ylabel(cfg["ylabel"])
            ax.set_xlabel("Elapsed time [sec]")
            ax.set_ylim(*ylims)
            ax.legend(fontsize=10, ncol=4, loc="lower left")

        title = cfg["title"] + f"\n Expid: {self.expId}"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        if saveAs:
            fig.savefig(saveAs, dpi=120)
        return fig

    def _starMosaic(
        self,
        stampNum: int = 2,
        fig: plt.Figure | None = None,
        axs: dict[str, plt.Axes] | None = None,
        plo: float = 90.0,
        phi: float = 99.0,
        cutoutSize: int = -1,
        isAnimated: bool = False,
        saveAs: str | None = None,
    ) -> list[Artist]:
        """
        Internal: plot a mosaic of guider stamps for a given stamp index.
        Wraps the plotting logic for static and animated frames.
        """
        if fig is None or axs is None:
            fig, axs = self.setupFigure(figsize=(9, 9))

        if not self.withStars and cutoutSize > 0:
            self.log.warning("No stars data available. Using full frame.")
            cutoutSize = -1

        nStamps = len(self.guiderData)
        view = self.guiderData.view
        camAngle = self.guiderData.camRotAngle

        jitter: float | None = None
        artists: list[Artist] = []

        for detName in self.guiderData.guiderNames:
            ax = axs[detName]

            # Get star info for this detector
            starInfo = self._getStarInfo(detName, stampNum)

            # Render the stamp
            imObj = renderStampPanel(
                ax,
                self.guiderData,
                detName,
                stampNum,
                viewCenter=starInfo.refCenter,
                cutoutSize=cutoutSize,
                plo=plo,
                phi=phi,
            )
            artists.append(imObj)

            # Static overlays (reference frame elements)
            if not isAnimated:
                addReferenceOverlays(
                    ax, detName, starInfo.refCenter, cutoutSize,
                    labelCorner=self.layout.labelCorners.get(detName, "ot"),
                )

            # Animated overlays (star position elements)
            if starInfo.hasData and cutoutSize > 0:
                # Rotated crosshair follows the star
                crosshairArtists = plotCrosshairRotated(
                    starInfo.starCenter,
                    90 + camAngle,
                    ax,
                    color="grey",
                    size=cutoutSize,
                )
                artists.extend(crosshairArtists)

                # Star centroid marker
                starCross = plotStarCentroid(
                    ax,
                    starInfo.starCenter,
                    markerSize=8,
                )
                artists.extend(starCross)

                jitter = getStdCentroid(self.starsDf, self.expId)

        # Center panel annotation
        stampInfo = annotateStampInfo(
            axs["center"], expid=self.expId, stampNum=stampNum, nStamps=nStamps,
            view=view, jitter=jitter, xy=(0.5, 0.5),
        )
        artists.append(stampInfo)

        # Arrow panel
        if not isAnimated:
            drawArrows(
                axs["arrow"],
                cutoutSize if cutoutSize > 0 else DEFAULT_CUTOUT_SIZE,
                90.0 + self.camRotAngle,
            )

        # Clear ticks - detector panels keep spines for full frame,
        # center/arrow never have spines
        for name, ax in axs.items():
            isDetector = name not in ("center", "arrow")
            clearAxisTicks(ax, isSpine=isDetector and cutoutSize < 0)

        if saveAs:
            fig.savefig(saveAs, dpi=120)

        return artists

    def _getStarInfo(self, detName: str, stampNum: int) -> StarInfo:
        """Get star info for a detector, falling back to image center
        if unavailable.
        """
        if not self.withStars:
            shape = self.guiderData[detName, 0].shape
            return StarInfo.from_image_center(shape)

        detMask = self.starsDf["detector"] == detName
        if not detMask.any():
            shape = self.guiderData[detName, 0].shape
            return StarInfo.from_image_center(shape)

        return StarInfo.from_stars_df(self.starsDf, detName, stampNum)

    def plotMosaic(
        self,
        stampNum: int = 2,
        plo: float = 90.0,
        phi: float = 99.0,
        cutoutSize: int = -1,
        saveAs: str | None = None,
        figsize: tuple[float, float] = (9, 9),
    ) -> plt.Figure:
        """
        Plot a mosaic of guider stamps (a single stamp or a stacked image).

        Parameters
        ----------
        stampNum : `int`, optional
            Stamp index; values < 0 select the stacked (coadd) image.
        fig : `matplotlib.figure.Figure`, optional
            Existing figure to draw on; created if ``None``.
        axs : `dict[str, matplotlib.axes.Axes]`, optional
            Axes dictionary for the mosaic; created if ``None``.
        plo : `float`, optional
            Lower percentile for intensity scaling.
        phi : `float`, optional
            Upper percentile for intensity scaling.
        cutoutSize : `int`, optional
            Square cutout size around star center; -1 uses full frame.
        isAnimated : `bool`, optional
            If True, skip static overlays intended only for static displays.
        saveAs : `str`, optional
            If provided, path and filename to which the figure is saved.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The resulting figure.
        """
        fig, axs = self.setupFigure(figsize=figsize)
        self._starMosaic(
            stampNum=stampNum,
            fig=fig,
            axs=axs,
            plo=plo,
            phi=phi,
            cutoutSize=cutoutSize,
            isAnimated=False,
            saveAs=saveAs,
        )
        return fig

    def makeAnimation(
        self,
        cutoutSize: int,
        saveAs: str = "",
        fps: int = 5,
        dpi: int = 80,
        plo: float = 50,
        phi: float = 99,
        figsize: tuple[float, float] = (9, 9),
        holdFrames: int = 2,
    ) -> animation.ArtistAnimation:
        """Create a gif or mp4 of the guider mosaic across sequential stamps.

        Parameters
        ----------
        cutoutSize : `int`, optional
            Size scale used for arrow lengths and framing.
        saveAs : `str`, optional
            Output filepath for the GIF.
        fps : `int`, optional
            Frames per second.
        dpi : `int`, optional
            Output resolution in dots per inch.
        plo : `float`, optional
            Lower percentile for image scaling.
        phi : `float`, optional
            Upper percentile for image scaling.
        figsize : `tuple[float, float]`, optional
            Figure size in inches.
        holdFrames : `int`, optional
            Number of frames to hold the first and last frames.

        Returns
        -------
        ani : `matplotlib.animation.ArtistAnimation`
            The created animation object.
        """
        # build canvas
        fig, axs = self.setupFigure(figsize=figsize)

        # number of frames
        total = len(self.guiderData)

        # initial (stacked) frame
        artists0 = self._starMosaic(
            stampNum=-1,
            fig=fig,
            axs=axs,
            plo=plo,
            phi=phi,
            cutoutSize=cutoutSize,
            isAnimated=False,
        )

        frames = holdFrames * [artists0]

        # sequential stamps
        for i in range(1, total):
            artists = self._starMosaic(
                stampNum=i,
                fig=fig,
                axs=axs,
                plo=plo,
                phi=phi,
                cutoutSize=cutoutSize,
                isAnimated=True,
            )
            frames.append(artists)
        frames += holdFrames * [artists0]

        # create animation
        ani = animation.ArtistAnimation(fig, frames, interval=1000 / fps, blit=True, repeat_delay=1000)

        if saveAs:
            writer = _getWriter(saveAs)
            ani.save(saveAs, fps=fps, dpi=dpi, writer=writer)
        return ani


def getStdCentroid(statsDf: pd.DataFrame, expId: int) -> float:
    """
    Compute combined (quadrature) corrected centroid scatter for an exposure.

    Parameters
    ----------
    statsDf : `pandas.DataFrame`
        Statistics DataFrame containing centroid scatter columns.
    expId : `int`
        Exposure identifier.

    Returns
    -------
    jitter : `float`
        Quadrature sum of corrected AZ and ALT centroid scatter (arcsec).
    """
    stdAz = mad_std(statsDf.loc[statsDf["expid"] == expId, "dalt"].to_numpy())
    stdAlt = mad_std(statsDf.loc[statsDf["expid"] == expId, "daz"].to_numpy())
    return np.hypot(stdAz, stdAlt)


def drawArrows(
    ax: plt.Axes,
    cutoutSize: int,
    rotAngle: float = 0.0,
    *,
    baseLabels: tuple[str, str] = ("Y DVCS", "X DVCS"),
    overlayLabels: tuple[str, str] = ("Alt", "Az"),
    baseColor: str = LIGHT_BLUE,
    overlayColor: str = "lightgrey",
    center: tuple[float, float] | None = None,
) -> None:
    """
    Draw reference arrows for instrument (DVCS) and rotated Alt/Az axes.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    cutoutSize : `int`
        Size scale used for arrow lengths and framing.
    rotAngle : `float`, optional
        Rotation angle (degrees) for overlay (Alt/Az) axes.
    baseLabels : `tuple[str, str]`, optional
        Labels for the base (unrotated) Y/X axes.
    overlayLabels : `tuple[str, str]`, optional
        Labels for the rotated Alt/Az axes.
    baseColor : `str`, optional
        Color for base axes arrows.
    overlayColor : `str`, optional
        Color for rotated axes arrows.
    center : `tuple[float, float]`, optional
        Arrow origin; defaults to cutout center when None.
    """
    x0, y0 = (cutoutSize // 2, cutoutSize // 2) if center is None else center
    L = cutoutSize / 3.0

    def _draw(theta_deg: float, color: str, labels: tuple[str, str]) -> np.ndarray:
        t = np.radians(theta_deg)
        # unit vectors rotated by t
        dx_az, dy_az = L * np.cos(t), L * np.sin(t)  # +X (Az)
        dx_alt, dy_alt = -L * np.sin(t), L * np.cos(t)  # +Y (Alt)

        for dx, dy, label in ((dx_az, dy_az, labels[1]), (dx_alt, dy_alt, labels[0])):
            coords = (x0, y0, dx, dy)
            ax.arrow(
                *coords,
                color=color,
                width=cutoutSize / 120,
                head_width=cutoutSize / 30,
                head_length=cutoutSize / 20,
                length_includes_head=True,
                zorder=10,
            )
            ax.text(x0 + dx + 0.5, y0 + dy, label, color=color, fontsize=6, fontweight="bold")
        return np.array([min(dx_az, dx_alt), min(dy_az, dy_alt)])

    # Base DVCS axes (θ=0), then rotated Alt/Az overlay
    _ = _draw(0.0, baseColor, baseLabels)
    deltas = _draw(rotAngle, overlayColor, overlayLabels)
    deltas = np.where(deltas > 0, 0, deltas)  # only negative shifts

    # Framing
    ax.set_aspect("equal", adjustable="box")
    border = cutoutSize * 0.15
    xmin = min(x0 - border, x0 + deltas[0] - border)
    ymin = min(y0 - border, y0 + deltas[1] - border)

    ax.set_xlim(xmin - border, xmin + cutoutSize + border)
    ax.set_ylim(ymin - border, ymin + cutoutSize + border)
    clearAxisTicks(ax, isSpine=True)
    ax.set_axis_off()


def renderStampPanel(
    ax: plt.Axes,
    guiderData: GuiderData,
    detName: str,
    stampNum: int,
    *,
    viewCenter: tuple[float, float],
    cutoutSize: int = -1,
    plo: float = 50.0,
    phi: float = 99.0,
) -> plt.AxesImage:
    """
    Render a single detector stamp with zoom around viewCenter.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    guiderData : `GuiderData`
        Guider data container.
    detName : `str`
        Detector name.
    stampNum : `int`
        Stamp index; negative for stacked/coadd.
    viewCenter : `tuple[float, float]`
        Center for the view (xlim/ylim) and intensity scaling.
    cutoutSize : `int`, optional
        Zoom size; -1 for full frame.
    plo : `float`, optional
        Lower percentile for intensity scaling.
    phi : `float`, optional
        Upper percentile for intensity scaling.

    Returns
    -------
    image : `matplotlib.image.AxesImage`
        The image artist (for animation).
    """
    # Get image
    img = guiderData.getStampArrayCoadd(detName) if stampNum < 0 else guiderData[detName, stampNum]
    h, w = img.shape

    # Crop and calculate scaling
    if cutoutSize > 0:
        cx, cy = int(viewCenter[0]), int(viewCenter[1])
        half = cutoutSize // 2
        # Crop bounds (clamped to image)
        y0, y1 = max(0, cy - half), min(h, cy + half)
        x0, x1 = max(0, cx - half), min(w, cx + half)
    else:
        y0, y1 = 0, h
        x0, x1 = 0, w

    region = img[y0:y1, x0:x1]
    vmin, vmax = np.nanpercentile(region, [plo, phi])
    # Offset the extent by -0.5 so integer (0-based) pixel coordinates fall at
    # pixel CENTERS. This aligns the star-centroid marker (drawn at xroi, yroi)
    # and the crosshair with the displayed star, instead of the pixel corners.
    extent: tuple[float, float, float, float] = (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5)

    # Render only the visible region
    im = ax.imshow(
        region,
        origin="lower",
        cmap="Greys",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=extent,
        animated=True,
    )

    # Set zoom limits
    if cutoutSize > 0:
        halfF = cutoutSize / 2.0
        ax.set_xlim(viewCenter[0] - halfF, viewCenter[0] + halfF)
        ax.set_ylim(viewCenter[1] - halfF, viewCenter[1] + halfF)

    ax.set_aspect("equal", "box")
    return im


def labelDetector(
    ax: plt.Axes,
    name: str,
    *,
    corner: str = "tl",
    color: str = LIGHT_BLUE,
    fontsize: int = 9,
    weight: str = "bold",
    pad: tuple[float, float] = (0.025, 0.025),
) -> plt.Text:
    """Place the detector name text inside a panel.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    name : `str`
        Detector name to display.
    corner : `str`, optional
        Two-letter code: t/b (top/bottom) + l/r (left/right).
    color : `str`, optional
        Text color.
    fontsize : `int`, optional
        Font size.
    weight : `str`, optional
        Font weight.
    pad : `tuple[float, float]`, optional
        Relative (x,y) padding in axes fraction.

    Returns
    -------
    text : `matplotlib.text.Text`
        Created text artist.
    """
    # Placements outside the panel (hence outside the inscribed 2" circle):
    #   'ot' = centered just above the circle; 'l'/'r' = beside it (vertically
    #   centered), for panels that reach the top edge where 'ot' would clip.
    if corner == "ot":
        return ax.text(
            0.5, 1.06, name, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=fontsize, weight=weight, color=color,
        )
    if corner in ("l", "r"):
        sidePad = 0.06
        xpos, ha = (-sidePad, "right") if corner == "l" else (1 + sidePad, "left")
        return ax.text(
            xpos, 0.5, name, transform=ax.transAxes, ha=ha, va="center",
            fontsize=fontsize, weight=weight, color=color,
        )

    ha, va = ("left", "top") if corner[0] == "t" else ("left", "bottom")
    ha = "right" if corner[1] == "r" else ha
    va = "top" if corner[0] == "t" else va

    xpad, ypad = pad
    xpos = xpad if ha == "left" else 1 - xpad
    ypos = 1 - ypad if va == "top" else ypad

    txt = ax.text(
        xpos,
        ypos,
        name,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        weight=weight,
        color=color,
    )
    return txt


def annotateStampInfo(
    ax: plt.Axes,
    *,
    expid: int,
    stampNum: int,
    nStamps: int,
    view: str | None = None,
    jitter: float | None = None,
    extra: str | None = None,
    xy: tuple[float, float] = (1.085, -0.10),
) -> plt.Text:
    """
    Annotate exposure metadata and stamp index on the center panel.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    expid : `int`
        Exposure identifier.
    stampNum : `int`
        Current stamp index (0-based). Negative implies stacked.
    nStamps : `int`
        Total number of stamps available.
    view : `str`, optional
        Orientation descriptor.
    jitter : `float`, optional
        Combined centroid scatter (arcsec).
    extra : `str`, optional
        Additional free-form text.
    xy : `tuple[float, float]`, optional
        Annotation position in axes coordinates.

    Returns
    -------
    text : `matplotlib.text.Text`
        Created text artist.
    """
    dayObs, seqNum = str(expid)[:8], int(str(expid)[-5:])

    text = ""
    if jitter is not None:
        text += f"Center Stdev.: {jitter:.2f} arcsec\n"
    if dayObs is not None and seqNum is not None and view is not None:
        text += f"day_obs: {dayObs}\nseq_num: {seqNum}\norientation: {view}\n"
    text += f"Stamp #: {stampNum + 1:02d}" if stampNum >= 0 else f"Stacked w/ {nStamps} stamps"
    if extra is not None:
        text += "\n" + extra

    txt = ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        color="grey",
    )
    clearAxisTicks(ax, isSpine=True)
    return txt


def addReferenceOverlays(
    ax: plt.Axes,
    detName: str,
    refCenter: tuple[float, float],
    cutoutSize: int,
    labelCorner: str = "tl",
) -> None:
    """Add static reference overlays: detector label, crosshairs,
    and guide circles.

    These are static elements that don't change during animation.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    detName : `str`
        Detector name.
    refCenter : `tuple[float, float]`
        Reference center coordinates (fixed frame).
    cutoutSize : `int`
        Cutout size; influences overlay scaling.
    """
    labelDetector(ax, detName, corner=labelCorner)
    if cutoutSize > 0:
        ax.axvline(refCenter[0], color=LIGHT_BLUE, lw=1.25, linestyle="--", alpha=0.75)
        ax.axhline(refCenter[1], color=LIGHT_BLUE, lw=1.25, linestyle="--", alpha=0.75)
        radii = [10, 5]
        drawGuideCircles(
            ax,
            refCenter,
            radii=radii,
            colors=[LIGHT_BLUE, LIGHT_BLUE],
            labels=["2″", "1″"],
            linewidth=2.0,
        )


def drawGuideCircles(
    ax: plt.Axes,
    center: tuple[float, float],
    radii: Sequence[float],
    colors: Sequence[str],
    labels: Sequence[str] | None = None,
    text_offset: float = 1.0,
    **circle_kwargs: Any,
) -> list[plt.Text]:
    """
    Draw concentric guide circles with optional labels.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    center : `tuple[float, float]`
        Center (x, y) of circles.
    radii : `Sequence[float]`
        Radii of circles.
    colors : `Sequence[str]`
        Edge colors per circle.
    labels : `Sequence[str]`, optional
        Labels per circle; None for no labels.
    text_offset : `float`, optional
        Offset for label placement along +x.
    **circle_kwargs : `dict`
        Extra keyword args passed to Circle.

    Returns
    -------
    texts : `list[matplotlib.text.Text]`
        List of label text artists.
    """
    x0, y0 = center
    txt_list: list[plt.Text] = []
    for i, r in enumerate(radii):
        c = Circle(
            (x0, y0),
            r,
            edgecolor=colors[i],
            facecolor="none",
            linestyle="--",
            **circle_kwargs,
        )
        ax.add_patch(c)
        label_txt = labels[i] if labels is not None else ""
        txt = ax.text(
            x0 + r + text_offset,
            y0 - r / 4.0,
            label_txt,
            color=colors[i],
            va="center",
            fontsize=8,
        )
        txt_list.append(txt)
    return txt_list


def plotCrosshairRotated(
    center: tuple[float, float],
    angle: float,
    ax: plt.Axes,
    color: str = "grey",
    size: int = 30,
) -> list[Line2D]:
    """
    Plot a rotated crosshair centered on given coordinates.

    Parameters
    ----------
    center : `tuple[float, float]`
        Center (x, y) in image coordinates.
    angle : `float`
        Rotation angle in degrees.
    ax : `matplotlib.axes.Axes`
        Target axes.
    color : `str`, optional
        Line color.
    size : `int`, optional
        Size scaling factor.

    Returns
    -------
    artists : `list[matplotlib.lines.Line2D]`
        Line artists for animation.
    """
    cross_length = 1.5 * size if size > 0 else 30
    theta = np.radians(angle)

    cx, cy = center
    # Horizontal line (rotated)
    dx = cross_length * np.cos(theta) / 2
    dy = cross_length * np.sin(theta) / 2
    (hline,) = ax.plot(
        [cx - dx, cx + dx],
        [cy - dy, cy + dy],
        color=color,
        ls="--",
        lw=1.0,
        alpha=0.5,
    )
    # Vertical line (rotated)
    dx_v = cross_length * np.cos(theta + np.pi / 2) / 2
    dy_v = cross_length * np.sin(theta + np.pi / 2) / 2
    (vline,) = ax.plot(
        [cx - dx_v, cx + dx_v],
        [cy - dy_v, cy + dy_v],
        color=color,
        ls="--",
        lw=1.0,
        alpha=0.5,
    )
    return [hline, vline]


def plotStarCentroid(
    ax: plt.Axes,
    centerCutout: tuple[float, float],
    *,
    markerSize: int = 8,
    errXY: tuple[float, float] | None = None,
    color: str = "firebrick",
) -> list[Line2D]:
    """
    Plot the measured star centroid (with optional error bars) on a cutout.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    centerCutout : `tuple[float, float]`
        Reference center in the cutout.
    markerSize : `int`, optional
        Marker size; <=0 disables plotting.
    errXY : `tuple[float, float]`, optional
        (xerr, yerr) half-lengths for error bars.
    color : `str`, optional
        Marker and line color.

    Returns
    -------
    artists : `list[matplotlib.lines.Line2D]`
        Marker and error bar line artists.
    """
    if markerSize <= 0:
        return []

    x_star, y_star = centerCutout
    xerr, yerr = errXY if errXY is not None else (0.0, 0.0)

    (marker,) = ax.plot(x_star, y_star, "o", color=color, markersize=markerSize)
    (hline,) = ax.plot([x_star - xerr, x_star + xerr], [y_star, y_star], color=color, lw=2.5)
    (vline,) = ax.plot([x_star, x_star], [y_star - yerr, y_star + yerr], color=color, lw=2.5)
    return [marker, hline, vline]


def clearAxisTicks(ax: plt.Axes, isSpine: bool = False) -> None:
    """
    Remove all ticks/labels; optionally keep spines.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    isSpine : `bool`, optional
        If True, retain spines; otherwise hide them.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if not isSpine:
        for spine in ax.spines.values():
            spine.set_visible(False)


def cropAroundCenter(
    image: np.ndarray, center: tuple[float, float], size: int
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Extract a square crop centered on the specified coordinates.

    Parameters
    ----------
    image : `numpy.ndarray`
        Source 2D image.
    center : `tuple[float, float]`
        Center (x, y) in original image coordinates.
    size : `int`
        Desired square size in pixels.

    Returns
    -------
    cropped : `numpy.ndarray`
        Cropped (and possibly padded) image of shape (size, size).
    new_center : `tuple[float, float]`
        Center coordinates within the cropped image.
    """
    x, y = center
    x, y = int(round(x)), int(round(y))
    half = size // 2

    y0 = max(0, y - half)
    y1 = min(image.shape[0], y + half)
    x0 = max(0, x - half)
    x1 = min(image.shape[1], x + half)

    cropped = image[y0:y1, x0:x1]

    # Calculate offsets for padding
    padTop = max(0, (half - y))
    padLeft = max(0, (half - x))
    padBottom = size - (padTop + (y1 - y0))
    padRight = size - (padLeft + (x1 - x0))

    if padTop > 0 or padBottom > 0 or padLeft > 0 or padRight > 0:
        cropped = np.pad(
            cropped,
            ((padTop, padBottom), (padLeft, padRight)),
            mode="constant",
            constant_values=np.nan,
        )

    # New center in cropped image
    newCenter = (center[0] - x0 + padLeft, center[1] - y0 + padTop)
    return cropped, newCenter
