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

__all__ = ["GuiderStarTracker"]

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .transformation import convertRoiToCcd, convertToAltaz, convertToFocalPlane

if TYPE_CHECKING:
    from .reading import GuiderData

from .detection import GuiderStarTrackerConfig, buildReferenceCatalog, makeBlankCatalog, trackStarAcrossStamp


class GuiderStarTracker:
    """
    Class to track stars in the Guider data.

    Parameters
    ----------
    guiderData : `GuiderData`
        GuiderData instance containing guider data and metadata.
    config : `GuiderStarTrackerConfig`, optional
        Config object with setup and quality control parameters. If None, a new
        default GuiderStarTrackerConfig is created per GuiderStarTracker
        instance.
    """

    def __init__(
        self,
        guiderData: GuiderData,
        config: GuiderStarTrackerConfig | None = None,
    ) -> None:
        """
        Initialize the GuiderStarTracker with guider data and configuration.

        Parameters
        ----------
        guiderData : `GuiderData`
            GuiderData instance containing guider data and metadata.
        config : `GuiderStarTrackerConfig`, optional
            Config object with setup and quality control parameters.
        """
        self.log = logging.getLogger(__name__)
        self.guiderData = guiderData
        self.nStamps = len(guiderData)
        self.expid = guiderData.expid
        self.shape = guiderData[guiderData.detNameMax, 0].shape

        # detection and QC parameters from config
        if config is None:
            config = GuiderStarTrackerConfig()
        self.config = config

        # initialize outputs
        self.blankStars = makeBlankCatalog()
        self.columns = list(self.blankStars.columns)

    def trackGuiderStars(self, refCatalog: None | pd.DataFrame = None) -> pd.DataFrame:
        """
        Track stars across guider exposures using a reference catalog.

        Parameters
        ----------
        refCatalog : `pd.DataFrame`, optional
            Reference catalog with known star positions per detector.

        Returns
        -------
        stars : `pd.DataFrame`
            DataFrame with tracked stars and their properties,
            including positions, fluxes, and residual offsets.
        """
        _refCatalog = None
        if refCatalog is None:
            self.log.info("Using self-generated refcat")
            _refCatalog = buildReferenceCatalog(self.guiderData, self.log, self.config)
            refCatalog = applyQualityCuts(_refCatalog, self.shape, self.config)

        if refCatalog.empty:
            self.log.warning(f"Reference catalog is empty for {self.expid}. No stars to track.")
            return self.blankStars

        trackedStarTables = []
        for guiderName in self.guiderData.guiderNames:
            refDet = refCatalog.query(f"detector == '{guiderName}'")
            if refDet.empty:
                # Diagnose: check what was in the unfiltered refcat
                preCut = _refCatalog.query(f"detector == '{guiderName}'") if _refCatalog is not None else None
                if preCut is not None and not preCut.empty:
                    best = preCut.sort_values("snr", ascending=False).iloc[0]
                    reason = _diagnoseQualityCutRejections(
                        preCut.sort_values("snr", ascending=False).iloc[:1], self.shape, self.config
                    )[0]
                    self.log.warning(
                        f"No stars in refcat for {guiderName} in {self.expid}. "
                        f"Best candidate: flux={best['flux']:.0f} snr={best['snr']:.1f} "
                        f"e={np.hypot(best['e1'], best['e2']):.2f} fwhm={best['fwhm']:.1f} "
                        f"pos=({best['xroi']:.0f},{best['yroi']:.0f}). Rejected: {reason}"
                    )
                else:
                    self.log.warning(
                        f"No stars detected for {guiderName} in {self.expid} " f"(empty before quality cuts)."
                    )
                continue

            table = self._trackStarForOneGuider(refDet, guiderName)
            if not table.empty:
                trackedStarTables.append(table)

        if not trackedStarTables:
            self.log.warning(f"No stars tracked for {self.expid} for any guider.")
            return self.blankStars

        # build the final catalog
        trackedStarCatalog = pd.concat(trackedStarTables, ignore_index=True)

        # Set unique IDs for all tracked stars
        trackedStarCatalog = setUniqueId(self.guiderData, trackedStarCatalog)

        # Make the final DataFrame with selected columns
        return trackedStarCatalog[self.columns]

    def _trackStarForOneGuider(self, refCatalog: pd.DataFrame, guiderName: str) -> pd.DataFrame:
        """
        Track stars for a single guider using the reference catalog.

        Iteratively tries to track stars starting with the highest SNR,
        and falls back to the next best candidates if quality cuts fail.

        Parameters
        ----------
        refCatalog : `pd.DataFrame`
            Reference catalog with known star positions per detector.
        guiderName : `str`
            Name of the guider to process.

        Returns
        -------
        stars : `pd.DataFrame`
            DataFrame with tracked stars for the specified guider.
        """
        gd = self.guiderData
        shape = self.shape
        cfg = self.config

        # Sort by SNR (brightest first) for iterative fallback
        refCatalogSorted = refCatalog.sort_values(by="snr", ascending=False).reset_index(drop=True)

        stars = None
        for idx, row in refCatalogSorted.iterrows():
            refCenter = (row["xroi"], row["yroi"])
            starSnr = row["snr"]

            self.log.debug(f"Attempting to track star {idx + 1} (SNR={starSnr:.2f}) for {guiderName}")

            # Measure the stars across all stamps for this guider
            starStamps = trackStarAcrossStamp(refCenter, gd, guiderName, cfg)

            if starStamps.empty:
                self.log.debug(f"Star {idx + 1} produced no detections")
                continue

            # Apply quality cuts to the tracked stars
            stars = applyQualityCuts(starStamps, shape, cfg)

            # Filter by minimum number of detections
            minStampDetections = int(len(gd) * cfg.minValidStampFraction)
            mask = stars["stamp"].groupby(stars["detector"]).transform("count") >= minStampDetections
            stars = stars[mask].copy()

            if not stars.empty:
                self.log.debug(
                    f"Successfully tracked star {idx + 1} (SNR={starSnr:.2f}) "
                    f"for {guiderName} with {len(stars)} detections"
                )
                break
            else:
                self.log.debug(f"Star {idx + 1} (SNR={starSnr:.2f}) failed quality cuts for {guiderName}")
                stars = None

        if stars is None or stars.empty:
            self.log.warning(
                f"No stars passed quality cuts for {guiderName} in {self.expid}. "
                f"Tried {len(refCatalogSorted)} candidates."
            )
            return self.blankStars

        # convert to CCD, focal plane and Alt/Az coordinates
        stars = convertToCcdFocalPlaneAltAz(stars, gd, guiderName)

        # Compute the rotator angle: theta = np.arctan2(yfp, xfp)
        stars = computeRotatorAngle(stars)

        # Convert e1, e2 to Alt/Az coordinates
        stars = convertEllipticity(stars, gd.camRotAngle)

        # Add timestamp and elapsed time
        stars = addTimeStamp(stars, gd, guiderName)

        # Compute Offsets
        stars = computeOffsets(stars)

        return stars


def addTimeStamp(stars: pd.DataFrame, guiderData: GuiderData, guiderName: str) -> pd.DataFrame:
    """
    Add timestamp and elapsed time to the star DataFrame.

    Parameters
    ----------
    stars : `pd.DataFrame`
        DataFrame with star measurements.
    guiderData : `GuiderData`
        GuiderData instance containing guider data and metadata.
    guiderName : `str`
        Name of the guider.

    Returns
    -------
    stars : `pd.DataFrame`
        DataFrame with added 'timestamp' and 'elapsed_time' columns.
    """
    gd = guiderData
    # the stamp are aligned with the index of the timestamps
    stampIndex = stars["stamp"].to_numpy()
    indices = np.array([ix for ix in stampIndex], dtype=int)

    # get the timestamp for each stamp
    timeStamp = gd.timestampMap[guiderName][indices]

    # inital time is the time of the first stamp of the guider with most stamps
    t0 = gd.timestampMap[gd.detNameMax][0]

    stars["timestamp"] = timeStamp
    stars["elapsed_time"] = np.where(timeStamp.mask, np.nan, (timeStamp - t0).sec)
    return stars


def convertToCcdFocalPlaneAltAz(stars: pd.DataFrame, guiderData: GuiderData, guiderName: str) -> pd.DataFrame:
    """
    Convert star positions to CCD, focal plane, and Alt/Az coordinates.

    Parameters
    ----------
    stars : `pd.DataFrame`
        DataFrame with star measurements in ROI coordinates.
    guiderData : `GuiderData`
        GuiderData instance containing guider data and metadata.
    guiderName : `str`
        Name of the guider.

    Returns
    -------
    stars : `pd.DataFrame`
        DataFrame with added columns for CCD, focal plane,
        and Alt/Az coordinates.
    """
    gd = guiderData
    obsTime = gd.obsTime
    detNum = gd.getGuiderDetNum(guiderName)
    wcs = gd.getWcs(guiderName)
    pixelScale = wcs.getPixelScale().asArcseconds()

    # Convert to CCD coordinates
    stars["xccd"], stars["yccd"] = convertRoiToCcd(
        stars["xroi"].to_numpy(), stars["yroi"].to_numpy(), gd, guiderName
    )

    # Convert to focal plane coordinates
    stars["xfp"], stars["yfp"] = convertToFocalPlane(
        stars["xccd"].to_numpy(), stars["yccd"].to_numpy(), detNum
    )

    # Convert to Alt/Az coordinates
    stars["alt"], stars["az"] = convertToAltaz(
        stars["xccd"].to_numpy(), stars["yccd"].to_numpy(), wcs, obsTime
    )

    # Convert fwhm to arcsec
    stars["fwhm"] = stars["fwhm"] * pixelScale

    return stars


def applyQualityCuts(
    stars: pd.DataFrame, shape: tuple[float, float], config: GuiderStarTrackerConfig
) -> pd.DataFrame:
    """
    Apply cuts according to min SNR, maximum ellipticity and edge margin.

    Parameters
    ----------
    stars : `pd.DataFrame`
        DataFrame with star measurements.
    shape : `tuple[float, float]`
        Shape of the ROI (cols, rows).
    config : `GuiderStarTrackerConfig`
        Configuration object with quality cut parameters.

    Returns
    -------
    stars : `pd.DataFrame`
        DataFrame with stars that passed the quality cuts.
    """
    minSnr = config.minSnr
    maxEllipticity = config.maxEllipticity
    edgeMargin = config.edgeMargin
    roiCols, roiRows = shape

    # Filter by minimum SNR
    mask1 = (stars["snr"] >= minSnr) & (stars["flux"] > 0) & (stars["flux_err"] > 0)

    # Filter by minimum number of detections and maximum ellipticity
    eabs = np.hypot(stars["e1"], stars["e2"])
    mask3 = (stars["e1"].abs() <= maxEllipticity) & (stars["e1"].abs() <= maxEllipticity)
    mask3 &= eabs <= maxEllipticity

    # Add edge margin mask
    mask4 = (
        (stars["xroi"] >= edgeMargin)
        & (stars["xroi"] <= roiRows - edgeMargin)
        & (stars["yroi"] >= edgeMargin)
        & (stars["yroi"] <= roiCols - edgeMargin)
    )
    # Combine all masks
    mask = mask1 & mask3 & mask4
    return stars[mask].copy()


def _diagnoseQualityCutRejections(
    stars: pd.DataFrame, shape: tuple[float, float], config: GuiderStarTrackerConfig
) -> list[str]:
    """Return per-row rejection reasons using the same masks as
    applyQualityCuts.

    Parameters
    ----------
    stars : `pd.DataFrame`
        Pre-cut DataFrame (rows that were all rejected).
    shape : `tuple[float, float]`
        ROI shape (rows, cols).
    config : `GuiderStarTrackerConfig`
        Configuration with quality cut thresholds.

    Returns
    -------
    reasons : `list[str]`
        One string per row describing which cuts failed.
    """
    minSnr = config.minSnr
    maxEllipticity = config.maxEllipticity
    edgeMargin = config.edgeMargin
    roiCols, roiRows = shape

    # Same masks as applyQualityCuts
    snrOk = (stars["snr"] >= minSnr) & (stars["flux"] > 0) & (stars["flux_err"] > 0)

    eabs = np.hypot(stars["e1"], stars["e2"])
    ellipOk = (stars["e1"].abs() <= maxEllipticity) & (stars["e1"].abs() <= maxEllipticity)
    ellipOk &= eabs <= maxEllipticity

    edgeOk = (
        (stars["xroi"] >= edgeMargin)
        & (stars["xroi"] <= roiRows - edgeMargin)
        & (stars["yroi"] >= edgeMargin)
        & (stars["yroi"] <= roiCols - edgeMargin)
    )

    reasons = []
    for i, (_, row) in enumerate(stars.iterrows()):
        r = []
        if not snrOk.iloc[i]:
            snr = row["snr"]
            r.append(f"snr={snr:.1f}" if np.isfinite(snr) else "snr=NaN")
        if not ellipOk.iloc[i]:
            e = np.hypot(row["e1"], row["e2"])
            r.append(f"e={e:.2f}" if np.isfinite(e) else "e=NaN")
        if not edgeOk.iloc[i]:
            r.append(f"edge(x={row['xroi']:.0f},y={row['yroi']:.0f})")
        reasons.append("|".join(r) if r else "?")
    return reasons


def setUniqueId(guiderData: GuiderData, stars: pd.DataFrame) -> pd.DataFrame:
    """
    Assign unique IDs to tracked stars.

    Parameters
    ----------
    guiderData : `GuiderData`
        GuiderData instance containing guider data and metadata.
    stars : `pd.DataFrame`
        DataFrame with star measurements.

    Returns
    -------
    stars : `pd.DataFrame`
        DataFrame with added 'detid' and 'trackid' columns.

    """
    # Create a numeric “global” starid:
    detMap = guiderData.guiderNameMap
    stars["detid"] = stars["detector"].map(detMap)
    stars["trackid"] = stars["detid"] * 1000 + stars["stamp"]
    return stars


def computeOffsets(stars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the offsets for each star in the catalog.

    Parameters
    ----------
    stars : `pd.DataFrame`
        DataFrame with star measurements.

    Returns
    -------
    stars : `pd.DataFrame`
        DataFrame with added offset columns:
            - dx, dy       : offsets in CCD coordinates (pixels)
            - dxfp, dyfp   : offsets in focal plane coordinates (mm)
            - dalt, daz    : offsets in Alt/Az coordinates (arcsec)
            - dtheta       : offset in rotator angle (arcsec)
            - magoffset    : magnitude offset (mmag)

    """
    # make reference positions
    stars["xroi_ref"] = stars.groupby("detector")["xroi"].transform("median")
    stars["yroi_ref"] = stars.groupby("detector")["yroi"].transform("median")
    stars["xccd_ref"] = stars.groupby("detector")["xccd"].transform("median")
    stars["yccd_ref"] = stars.groupby("detector")["yccd"].transform("median")
    stars["xfp_ref"] = stars.groupby("detector")["xfp"].transform("median")
    stars["yfp_ref"] = stars.groupby("detector")["yfp"].transform("median")
    stars["alt_ref"] = stars.groupby("detector")["alt"].transform("median")
    stars["az_ref"] = stars.groupby("detector")["az"].transform("median")
    stars["theta_ref"] = stars.groupby("detector")["theta"].transform("median")

    # Compute all your offsets
    stars["dx"] = stars["xccd"] - stars["xccd_ref"]
    stars["dy"] = stars["yccd"] - stars["yccd_ref"]
    stars["dxfp"] = stars["xfp"] - stars["xfp_ref"]
    stars["dyfp"] = stars["yfp"] - stars["yfp_ref"]
    stars["dalt"] = (stars["alt"] - stars["alt_ref"]) * 3600
    stars["daz"] = (stars["az"] - stars["az_ref"]) * 3600
    stars["dtheta"] = (stars["theta"] - stars["theta_ref"]) * 3600

    # Correct for cos(alt) in daz
    stars["daz"] = np.cos(stars["alt_ref"] * np.pi / 180) * stars["daz"]

    # compute mag offset
    stars["flux_ref"] = stars.groupby("detector")["flux"].transform("median")
    stars["flux_ref"] = pd.to_numeric(stars["flux_ref"], errors="coerce")
    stars["flux"] = pd.to_numeric(stars["flux"], errors="coerce")
    stars["magoffset"] = -2.5 * np.log10((stars["flux"] + 1e-12) / (stars["flux_ref"] + 1e-12))
    # convert to mmag and replace inf with nan
    stars["magoffset"] = 1000 * stars["magoffset"].replace([np.inf, -np.inf], np.nan)

    # not detections have nan offsets
    offsetCols = ["dx", "dy", "dxfp", "dyfp", "dalt", "daz", "dtheta", "magoffset"]
    stars.loc[stars["flux"].isna(), offsetCols] = np.nan

    return stars


def computeRotatorAngle(stars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the rotator angle (theta) and its propagated 1-sigma uncertainty
    for each row in `stars`.

    Parameters
    ----------
    stars : `pd.DataFrame`
        DataFrame with star measurements. Must contain 'xfp', 'yfp', 'xerr',
          and 'yerr' columns.

    Returns
    -------
    stars : `pd.DataFrame`
        DataFrame with added 'theta' and 'theta_err' columns.
    """
    mm = 0.001  # convert microns to mm
    xfp = stars["xfp"].to_numpy()
    yfp = stars["yfp"].to_numpy()
    sigmaX = stars["xerr"].to_numpy() * 10 * mm
    sigmaY = stars["yerr"].to_numpy() * 10 * mm

    # Angle in degrees
    theta = np.degrees(np.arctan2(yfp, xfp))

    # Denominator r^2 = x^2 + y^2
    denom = xfp**2 + yfp**2

    # Suppress warnings for rows where denom == 0; set sigmaTheta to NaN there.
    with np.errstate(divide="ignore", invalid="ignore"):
        sigmaThetaRad = np.sqrt((yfp**2 * sigmaX**2 + xfp**2 * sigmaY**2) / denom**2)
        sigmaTheta = np.degrees(sigmaThetaRad)
        sigmaTheta = np.where(denom == 0, np.nan, sigmaTheta)

    stars["theta"] = theta
    stars["theta_err"] = sigmaTheta * 3600  # convert to arcsec
    return stars


def convertEllipticity(stars: pd.DataFrame, camRotAngleDeg: float) -> pd.DataFrame:
    """
    Rotate ellipticity components (e1, e2) to Alt/Az.

    Parameters
    ----------
    stars : `pd.DataFrame`
        DataFrame with star measurements.
    camRotAngleDeg : `float`
        Camera rotator angle in degrees.

    Returns
    -------
    stars : `pd.DataFrame`
        DataFrame with added columns 'e1_altaz' and 'e2_altaz'.
    """
    e1_rot, e2_rot = rotateEllipticity(stars["e1"], stars["e2"], camRotAngleDeg)
    stars["e1_altaz"] = e1_rot
    stars["e2_altaz"] = e2_rot
    return stars


def rotateEllipticity(e1: float, e2: float, theta_deg: float) -> tuple[float, float]:
    """
    Rotate ellipticity components (e1, e2) by theta_deg degrees.

    Parameters
    ----------
    e1 : `float`
        First ellipticity component.
    e2 : `float`
        Second ellipticity component.
    theta_deg : `float`
        Rotation angle in degrees.

    Returns
    -------
    e1_rot, e2_rot : `tuple[float, float]`
        Rotated ellipticity components.
    """
    theta = np.deg2rad(theta_deg)
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    e1_rot = e1 * cos2t + e2 * sin2t
    e2_rot = -e1 * sin2t + e2 * cos2t
    return e1_rot, e2_rot
