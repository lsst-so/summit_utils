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
from dataclasses import asdict, dataclass, field

__all__ = [
    "runSourceDetection",
    "buildReferenceCatalog",
    "trackStarAcrossStamp",
    "makeBlankCatalog",
    "runGalSim",
]

import galsim
import numpy as np
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats

import lsst.afw.detection as afwDetect
from lsst.afw.image import ExposureF, ImageF, MaskedImageF
from lsst.afw.math import STDEVCLIP, makeStatistics

from .reading import GuiderData

log = logging.getLogger(__name__)

_DEFAULT_COLUMNS: str = (
    "trackid detector expid elapsed_time dalt daz dtheta dx dy "
    "fwhm xroi yroi xccd yccd xroi_ref yroi_ref xccd_ref yccd_ref "
    "dxfp dyfp xfp yfp alt az xfp_ref yfp_ref alt_ref az_ref "
    "xerr yerr theta theta_err theta_ref flux flux_err magoffset snr "
    "ixx iyy ixy e1 e2 e1_altaz e2_altaz "
    "ampname timestamp stamp detid filter exptime galsim_failed "
)
DEFAULT_COLUMNS: tuple[str, ...] = tuple(_DEFAULT_COLUMNS.split())


def makeBlankCatalog() -> pd.DataFrame:
    """
    Create a blank DataFrame with the default columns for a star catalog.

    Returns
    -------
    catalog : `pd.DataFrame`
        Empty catalog with the default schema.
    """
    return pd.DataFrame(columns=DEFAULT_COLUMNS)


@dataclass(frozen=True, slots=True)
class GuiderStarTrackerConfig:
    """Configuration for the GuiderStarTracker.

    Parameters
    ----------
    minSnr : `float`
        Minimum signal-to-noise ratio for star detection.
    minValidStampFraction : `float`
        Minimum fraction of stamps of valid detection per detector.
        If provided, this is used instead of `minStampDetections`.
    edgeMargin : `int`
        Margin in pixels to avoid edge effects in the image.
    maxEllipticity : `float`
        Maximum allowed ellipticity for a star to be considered valid.
    cutOutSize : `int`
        Size of the cutout around the star for tracking.
    aperSizeArcsec : `float`
        Aperture size in arcseconds for star detection.
    gain : `float`
        Gain factor for the guider data, used in flux calculations.
    nFallbackStamps : `int`
        Number of individual stamps to sample when coadd detection fails.
    peakSnrThreshold : `float`
        Minimum peak pixel SNR to consider a stamp non-empty during
        single-stamp fallback detection.
    """

    minSnr: float = 10.0
    minValidStampFraction: float = 0.5
    edgeMargin: int = 5
    maxEllipticity: float = 0.7
    cutOutSize: int = 50
    aperSizeArcsec: float = 5.0
    gain: float = 1.0
    nFallbackStamps: int = 10
    peakSnrThreshold: float = 5.0


def trackStarAcrossStamp(
    refCenter: tuple[float, float],
    guiderData: GuiderData,
    guiderName: str,
    config: GuiderStarTrackerConfig = GuiderStarTrackerConfig(),
) -> pd.DataFrame:
    """
    Track a star across all guider stamps and compute centroid, shape, and
    flux.

    GalSim is used for centroid and shape measurements. Flux is measured with
    aperture photometry.

    Parameters
    ----------
    refCenter : `tuple[float, float]`
        Reference position (x, y) in pixel coordinates for the star.
    guiderData : `GuiderData`
        Guider data containing image stamps and metadata.
    guiderName : `str`
        Name of the guider to process.
    config : `GuiderStarTrackerConfig`
        Configuration parameters for the star tracker.

    Returns
    -------
    stars : `pd.DataFrame`
        DataFrame containing the tracked star measurements across all stamps.
    """
    gd = guiderData
    expid = gd.expid
    wcs = gd.getWcs(guiderName)
    pixelScale = wcs.getPixelScale().asArcseconds()

    # Initialize parameters from config
    apertureRadius = config.aperSizeArcsec / pixelScale
    cutOutSize = config.cutOutSize
    gain = config.gain

    # check if the ref center is within the image bounds
    stampShape = gd[guiderName, 0].shape
    if not (0 <= refCenter[0] < stampShape[1]) or not (0 <= refCenter[1] < stampShape[0]):
        return makeBlankCatalog()

    # loop over stamps
    results = []
    for i in range(len(gd)):
        data = gd[guiderName, i]
        star = measureStarOnStamp(data, refCenter, cutOutSize, apertureRadius, gain=gain).toDataFrame()

        # Add stamp index
        if not star.empty:
            star["stamp"] = i
            results.append(star)

    # 3)  Concatenate
    if not results:
        return makeBlankCatalog()
    stars = pd.concat(results, ignore_index=True)

    # 4)  Add metadata
    stars["detector"] = guiderName
    stars["expid"] = expid
    stars["ampname"] = gd.getGuiderAmpName(guiderName)
    stars["detid"] = gd.getGuiderDetNum(guiderName)
    stars["filter"] = gd.header.get("filter", "UNKNOWN")
    stars["exptime"] = gd.guiderDurationSec
    return stars


def annulusBackgroundSubtraction(data: np.ndarray, annulus: tuple[float, float]) -> tuple[np.ndarray, float]:
    """
    Subtract background from the data using an annulus.

    Parameters
    ----------
    data : `np.ndarray`
        Image cutout data.
    annulus : `tuple[float, float]`
        Inner and outer radii (pixels) defining the background annulus.

    Returns
    -------
    dataBkgSub : `np.ndarray`
        Background-subtracted data.
    bkgStd : `float`
        Standard deviation of the background estimation.
    """
    rin, rout = annulus
    x0, y0 = data.shape[1] // 2, data.shape[0] // 2
    x, y = np.indices(data.shape)
    annMask = ((x - x0) ** 2 + (y - y0) ** 2 >= rin**2) & ((x - x0) ** 2 + (y - y0) ** 2 <= rout**2)
    annMask &= np.isfinite(data)
    _, bkgSub, bkgStd = sigma_clipped_stats(data[annMask], sigma=3.0)
    dataBkgSub = data - bkgSub
    return dataBkgSub, bkgStd


@dataclass
class StarMeasurement:
    xroi: float = field(default=np.nan)
    yroi: float = field(default=np.nan)
    xerr: float = field(default=0.0)
    yerr: float = field(default=0.0)
    e1: float = field(default=np.nan)
    e2: float = field(default=np.nan)
    ixx: float = field(default=np.nan)
    iyy: float = field(default=np.nan)
    ixy: float = field(default=np.nan)
    fwhm: float = field(default=np.nan)
    flux: float = field(default=np.nan)
    flux_err: float = field(default=0.0)
    snr: float = field(default=0.0)
    galsim_failed: bool = field(default=False)

    def toDataFrame(self) -> pd.DataFrame:
        """
        Convert this measurement to a single-row DataFrame.

        Returns
        -------
        row : `pd.DataFrame`
            Single-row DataFrame with measurement fields, or empty if invalid.
        """
        d = asdict(self)
        # Only drop the column if xroi is NaN (i.e., measurement failed)
        if not np.isfinite(d.get("xroi", np.nan)):
            # Return an empty DataFrame with all the keys as columns,
            return pd.DataFrame(columns=list(d.keys()))
        # Otherwise, return all columns, even if some are NaN
        return pd.DataFrame([d])

    def runAperturePhotometry(
        self, cutout: np.ndarray, radius: float, bkgStd: float = 1.0, gain: float = 1.0
    ) -> None:
        """
        Perform aperture photometry on a cutout image.

        Updates the flux, flux_err, and snr attributes of the StarMeasurement.

        Parameters
        ----------
        cutout : `np.ndarray`
            2D cutout image (background-subtracted).
        radius : `float`
            Aperture radius in pixels.
        bkgStd : `float`
            Background RMS per pixel.
        gain : `float`
            Detector gain (e-/ADU).
        """
        x0, y0 = self.xroi, self.yroi
        if np.isfinite(x0) and np.isfinite(y0):
            ny, nx = cutout.shape
            y, x = np.indices((ny, nx))
            x0, y0 = self.xroi, self.yroi

            # Background mask
            aperMask = (x - x0) ** 2 + (y - y0) ** 2 <= radius**2

            # Aperture sum
            fluxNet = np.nansum(cutout[aperMask])
            fluxNet = np.clip(fluxNet, 0, None)  # Ensure non-negative flux
            npix = aperMask.sum()

            # Flux error
            fluxErr = np.sqrt(fluxNet / gain + npix * bkgStd**2)
            snr = fluxNet / (fluxErr + 1e-9) if fluxErr > 0 else 0.0

            # Update the measurement
            self.flux = fluxNet
            self.flux_err = fluxErr
            self.snr = snr


def runSourceDetection(
    image: np.ndarray,
    threshold: float = 10,
    cutOutSize: int = 25,
    apertureRadius: int = 5,
    gain: float = 1.0,
    nPixMin: int = 10,
) -> pd.DataFrame:
    """
    Detect sources in an image and measure their properties.

    Parameters
    ----------
    image : `np.ndarray`
        2D image array.
    threshold : `float`
        Detection threshold in sigma units.
    cutOutSize : `int`
        Size of the cutout around each detected source (pixels).
    apertureRadius : `int`
        Aperture radius in pixels for photometry.
    gain : `float`
        Detector gain (e-/ADU).
    nPixMin : `int`
        Minimum number of pixels in a footprint for detection.

    Returns
    -------
    sources : `pd.DataFrame`
        DataFrame with detected source properties.
    """
    # Step 1: Convert numpy image to MaskedImage and Exposure
    exposure = ExposureF(MaskedImageF(ImageF(image)))

    # Step 2: Detect sources using STDEVCLIP for the background noise.
    # The input coadd images are dithered (see GuiderData.getStampArrayCoadd)
    # to prevent integer quantization from collapsing the pixel distribution
    # and causing STDEVCLIP to return 0. (See DM-54263.)
    footprints = None
    if not isBlankImage(image):
        median = np.nanmedian(image)
        exposure.image -= median
        imageStd = float(makeStatistics(exposure.getMaskedImage(), STDEVCLIP).getValue(STDEVCLIP))
        if imageStd <= 0:
            # Fallback: sigma68 is robust to quantization and bright stars.
            p16, p84 = np.nanpercentile(image, [16, 84])
            imageStd = (p84 - p16) / 2.0
        if imageStd <= 0:
            exposure.image += median
            return pd.DataFrame(columns=DEFAULT_COLUMNS)
        absThreshold = threshold * imageStd
        thresh = afwDetect.Threshold(absThreshold, afwDetect.Threshold.VALUE)
        footprints = afwDetect.FootprintSet(exposure.getMaskedImage(), thresh, "DETECTED", nPixMin)
        exposure.image += median

    if not footprints:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    nFootprints = len(footprints.getFootprints())
    results = []
    for fp in footprints.getFootprints():
        # Create a cutout of the image around the footprint
        refCenter = tuple(fp.getCentroid())
        star = measureStarOnStamp(image, refCenter, cutOutSize, apertureRadius, gain).toDataFrame()
        if not star.empty:
            results.append(star)
    if not results:
        if nFootprints > 0:
            log.warning(
                f"FootprintSet found {nFootprints} sources but GalSim "
                f"failed on all of them (cutOutSize={cutOutSize})."
            )
        return pd.DataFrame(columns=DEFAULT_COLUMNS)
    df = pd.concat([sf for sf in results], ignore_index=True)
    return df


def measureStarOnStamp(
    stamp: np.ndarray,
    refCenter: tuple[float, float],
    cutOutSize: int,
    apertureRadius: int,
    gain: float = 1.0,
) -> StarMeasurement:
    """
    Measure a star on a single stamp: background subtraction, shape, centroid,
    photometry.

    Parameters
    ----------
    stamp : `np.ndarray`
        Full stamp array.
    refCenter : `tuple[float, float]`
        Reference (x, y) pixel position for the cutout center.
    cutOutSize : `int`
        Size of the cutout in pixels.
    apertureRadius : `int`
        Aperture radius in pixels for photometry.
    gain : `float`
        Detector gain (e-/ADU).

    Returns
    -------
    measurement : `StarMeasurement`
        StarMeasurement object with populated fields (may be empty on failure).
    """
    cutout = getCutouts(stamp, refCenter, cutoutSize=cutOutSize)
    data = cutout.data.copy()

    if np.all(data == 0):
        return StarMeasurement()

    # Replace NaN (from border-clipped cutouts) with the median so
    # GalSim can still measure stars near the stamp edge.
    nan_mask = ~np.isfinite(data)
    if nan_mask.any():
        data[nan_mask] = np.nanmedian(data)

    # 1) Subtract the background
    annulus = (apertureRadius * 1.0, apertureRadius * 2)
    dataBkgSub, bkgStd = annulusBackgroundSubtraction(data, annulus)

    # 2)  Track the star across all stamps for this guider
    star = runGalSim(dataBkgSub, gain=gain, bkgStd=bkgStd)

    # If GalSim fails, fall back to flux-weighted centroid + aperture photometry.
    # This gives valid x, y, flux for tracking even without shape info.
    if not np.isfinite(star.xroi):
        weights = np.clip(dataBkgSub, 0, None)
        total = np.nansum(weights)
        if total > 0:
            yy, xx = np.indices(dataBkgSub.shape, dtype=float)
            star.xroi = float(np.nansum(xx * weights) / total)
            star.yroi = float(np.nansum(yy * weights) / total)
            star.galsim_failed = True

    # 3) Make aperture photometry measurements
    # Galsim flux is the normalization of the Gaussian, not w/ fixed aper.
    star.runAperturePhotometry(dataBkgSub, apertureRadius, gain=gain, bkgStd=bkgStd)

    # 4)  Add centroid and shape in amplifier roi coordinates
    star.xroi += cutout.xmin_original
    star.yroi += cutout.ymin_original
    return star


def runGalSim(
    imageArray: np.ndarray,
    gain: float = 1.0,
    bkgStd: float = 0.0,
) -> StarMeasurement:
    """
    Measure star properties with GalSim adaptive moments.

    Parameters
    ----------
    imageArray : `np.ndarray`
        Background-subtracted image cutout.
    gain : `float`
        Detector gain (e-/ADU).
    bkgStd : `float`
        Background RMS per pixel.

    Returns
    -------
    result : `StarMeasurement`
        Resulting measurement (empty if measurement failed).
    """
    gsImg = galsim.Image(imageArray)
    hsmRes = galsim.hsm.FindAdaptiveMom(gsImg, strict=False)
    success = hsmRes.error_message == ""

    if not success:
        result = StarMeasurement()
    else:
        xCentroid = hsmRes.moments_centroid.x
        yCentroid = hsmRes.moments_centroid.y
        flux = hsmRes.moments_amp
        sigma = hsmRes.moments_sigma
        e1 = hsmRes.observed_shape.e1
        e2 = hsmRes.observed_shape.e2
        fwhm = 2.355 * sigma

        # Calculate errors using GalSim's error estimation
        xErr, yErr = calcGalsimError(imageArray, hsmRes, gain=gain, bkgStd=bkgStd, correctForGain=True)

        # Calculate SNR and flux error
        ellipticity = np.sqrt(e1**2 + e2**2)
        nEff = 2 * np.pi * sigma**2 * np.sqrt(1 - ellipticity**2)
        shotNoise = np.sqrt(nEff * bkgStd**2)
        fluxErr = np.sqrt(max(0, flux / gain) + shotNoise**2)
        snr = flux / (shotNoise + 1e-9) if shotNoise > 0 else 0.0

        # Calculate second moments
        ixx = sigma**2 * (1 + e1)
        iyy = sigma**2 * (1 - e1)
        ixy = sigma**2 * e2

        result = StarMeasurement(
            xroi=xCentroid,
            yroi=yCentroid,
            xerr=xErr,
            yerr=yErr,
            e1=e1,
            e2=e2,
            ixx=ixx,
            iyy=iyy,
            ixy=ixy,
            fwhm=fwhm,
            flux=flux,
            flux_err=fluxErr,
            snr=snr,
        )
    return result


def calcGalsimError(
    imageArray: np.ndarray,
    shape: galsim.hsm.ShapeData,
    gain: float = 1.0,
    bkgStd: float = 0.0,
    correctForGain: bool = False,
) -> tuple[float, float]:
    """
    Estimate centroid errors from GalSim HSMShapeData.

    Parameters
    ----------
    imageArray : `np.ndarray`
        Image cutout used for measurement.
    shape : `galsim.hsm`
        GalSim HSM shape data result object.
    gain : `float`
        Detector gain (e-/ADU), ignored if `correctForGain` is False.
    bkgStd : `float`
        Background RMS per pixel.
    correctForGain : `bool`
        Whether to include gain-dependent weighting.

    Returns
    -------
    xerr : `float`
        Estimated x centroid uncertainty (pixels).
    yerr : `float`
        Estimated y centroid uncertainty (pixels).
    """
    if not shape or shape.error_message != "":
        return 0.0, 0.0

    x0 = shape.moments_centroid.x
    y0 = shape.moments_centroid.y
    sigma = shape.moments_sigma
    e1 = shape.observed_shape.e1
    e2 = shape.observed_shape.e2
    flux = shape.moments_amp

    kernel = makeEllipticalGaussianStar(
        shape=(imageArray.shape[0], imageArray.shape[1]),
        e1=e1,
        e2=e2,
        flux=1,
        sigma=sigma,
        center=(x0, y0),
    )

    weight = np.ones_like(imageArray) / (bkgStd**2 + 1e-9)
    if correctForGain:
        weight = np.ones_like(imageArray) / (bkgStd**2 + np.abs(flux * kernel / gain))

    mask = weight == 0.0
    data = imageArray.copy()
    if np.any(mask):
        kernelMasked = kernel.copy()
        data[mask] = kernelMasked[mask] * np.sum(data[~mask]) / np.sum(kernelMasked[~mask])

    u, v = np.meshgrid(np.arange(imageArray.shape[1]) - x0, np.arange(imageArray.shape[0]) - y0)
    usq = u**2
    vsq = v**2
    WI = kernel * data
    M00 = np.nansum(WI)
    WV = (kernel**2).astype(float)
    WV[~mask] /= weight[~mask]
    WV[mask] /= np.median(weight[~mask])
    WV = WV / float(M00**2)

    varM10 = 4 * np.sum(WV * usq)
    varM01 = 4 * np.sum(WV * vsq)
    xerr = np.sqrt(varM10)
    yerr = np.sqrt(varM01)
    return xerr, yerr


def makeEllipticalGaussianStar(
    shape: tuple[int, int],
    flux: float,
    sigma: float,
    e1: float,
    e2: float,
    center: tuple[float, float],
) -> np.ndarray:
    """
    Create an elliptical 2D Gaussian star with specified parameters.

    Parameters
    ----------
    shape : `tuple[int, int]`
        (ny, nx) output array shape.
    flux : `float`
        Total flux (normalization).
    sigma : `float`
        Gaussian sigma (pixels).
    e1 : `float`
        Ellipticity component e1.
    e2 : `float`
        Ellipticity component e2.
    center : `tuple[float, float]`
        (x0, y0) centroid position in pixels.

    Returns
    -------
    image : `np.ndarray`
        Generated model image.
    """
    y, x = np.indices(shape)
    x0, y0 = center
    u = x - x0
    v = y - y0

    # Second-moment matrix elements
    ixx = sigma**2 * (1 + e1)
    iyy = sigma**2 * (1 - e1)
    ixy = sigma**2 * e2

    # Inverse covariance matrix
    det = ixx * iyy - ixy**2
    invIxx = iyy / det
    invIyy = ixx / det
    invIxy = -ixy / det

    # Quadratic form: u^2 * invIxx + v^2 * invIyy + 2uv * invIxy
    r2 = invIxx * u**2 + invIyy * v**2 + 2 * invIxy * u * v

    e = np.sqrt(e1**2 + e2**2)
    norm = flux / (2 * np.pi * sigma**2 * np.sqrt(1 - e**2))
    image = norm * np.exp(-0.5 * r2)
    return image


def _detectOnSingleStamps(
    guiderData: GuiderData,
    guiderName: str,
    config: GuiderStarTrackerConfig,
    apertureRadius: int,
    log: logging.Logger,
) -> pd.DataFrame:
    """Try source detection on individual stamps when coadd detection fails.

    Samples up to ``config.nFallbackStamps`` stamps evenly spread across
    the sequence. Returns the detection with the highest SNR, or an empty
    DataFrame if no star is found on any stamp.

    A stamp is considered truly empty if its peak pixel SNR
    (max - median) / std is below ``config.peakSnrThreshold``.
    """
    nStamps = len(guiderData)
    if nStamps == 0:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    nSample = min(config.nFallbackStamps, nStamps)
    indices = np.linspace(0, nStamps - 1, nSample, dtype=int)

    bestSources = pd.DataFrame(columns=DEFAULT_COLUMNS)
    bestSnr = 0.0
    nTrulyEmpty = 0

    for idx in indices:
        stamp = guiderData[guiderName, idx].astype(np.float32)
        arr = stamp - np.nanmin(stamp)

        # Quick peak-SNR check before running full detection
        med = np.nanmedian(arr)
        std = np.nanstd(arr)
        if std > 0:
            peakSnr = (np.nanmax(arr) - med) / std
        else:
            peakSnr = 0.0

        if peakSnr < config.peakSnrThreshold:
            nTrulyEmpty += 1
            continue

        sources = runSourceDetection(
            arr,
            threshold=config.minSnr,
            apertureRadius=apertureRadius,
            cutOutSize=config.cutOutSize,
            gain=config.gain,
        )
        if not sources.empty and sources["snr"].max() > bestSnr:
            bestSnr = sources["snr"].max()
            bestSources = sources

    if not bestSources.empty:
        log.info(
            f"Single-stamp fallback recovered source on {guiderName} "
            f"(SNR={bestSnr:.1f}, {nTrulyEmpty}/{nSample} stamps empty)"
        )

    return bestSources


def buildReferenceCatalog(
    guiderData: GuiderData,
    log: logging.Logger,
    config: GuiderStarTrackerConfig = GuiderStarTrackerConfig(),
) -> pd.DataFrame:
    """
    Build a reference star catalog from each guider's coadded stamp.

    Parameters
    ----------
    guiderData : `GuiderData`
        Guider dataset containing stamps and metadata.
    log : `logging.Logger`
        Logger for warnings and diagnostics.
    config : `GuiderStarTrackerConfig`
        Star tracker configuration.

    Returns
    -------
    refCatalog : `pd.DataFrame`
        Concatenated reference catalog of brightest stars per guider.
    """
    expId = guiderData.expid
    minSnr = config.minSnr
    gain = config.gain
    cutOutSize = config.cutOutSize

    tableList = []
    for guiderName in guiderData.guiderNames:
        pixelScale = guiderData.getWcs(guiderName).getPixelScale().asArcseconds()
        apertureRadius = int(config.aperSizeArcsec / pixelScale)

        array = guiderData.getStampArrayCoadd(guiderName)
        array = array - np.nanmin(array)  # Ensure no negative values
        sources = runSourceDetection(
            array,
            threshold=minSnr,
            apertureRadius=apertureRadius,
            cutOutSize=cutOutSize,
            gain=gain,
        )
        detectionMethod = "coadd"
        if sources.empty:
            # Fallback: try detection on individual stamps. The coadd can
            # wash out stars that drift between frames or have artifacts.
            sources = _detectOnSingleStamps(guiderData, guiderName, config, apertureRadius, log)
            detectionMethod = "single_stamp"
        if sources.empty:
            log.warning(f"No sources detected in `buildReferenceCatalog`for {guiderName} in {expId}. ")
            continue

        sources["detection_method"] = detectionMethod

        sources.sort_values(by=["snr"], ascending=False, inplace=True)
        sources.reset_index(drop=True, inplace=True)

        detNum = guiderData.getGuiderDetNum(guiderName)
        sources["detector"] = guiderName
        sources["detid"] = detNum
        sources["starid"] = detNum * 100
        tableList.append(sources)

    if len(tableList) == 0:
        log.warning(f"buildReferenceCatalog failed - no stars detected in any guider for {expId}.")
        return makeBlankCatalog()

    refCatalog = pd.concat(tableList, ignore_index=True)
    return refCatalog


def getCutouts(imageArray: np.ndarray, refCenter: tuple[float, float], cutoutSize: int = 25) -> Cutout2D:
    """
    Get a cutout at the reference position from an image array.

    Parameters
    ----------
    imageArray : `np.ndarray`
        Full image array.
    refCenter : `tuple[float, float]`
        (x, y) center for the cutout in pixels.
    cutoutSize : `int`
        Size (pixels) of the square cutout.

    Returns
    -------
    cutout : `Cutout2D`
        Astropy Cutout2D object.
    """
    refX, refY = refCenter
    return Cutout2D(imageArray, (refX, refY), size=cutoutSize, mode="partial", fill_value=np.nan)


def isBlankImage(image: np.ndarray, fluxMin: float = 300, peakSnrMin: float = 5.0) -> bool:
    """
    Returns True if the image has no significant source (e.g., no star).

    An image is considered non-blank if the peak flux above the median
    exceeds ``fluxMin`` OR the peak pixel SNR (using MAD-based robust
    std) exceeds ``peakSnrMin``.

    Parameters
    ----------
    image : `np.ndarray`
        2D image data.
    fluxMin : `float`
        Minimum peak flux above median (ADU) to consider non-blank.
    peakSnrMin : `float`
        Minimum peak pixel SNR to consider non-blank.

    Returns
    -------
    bool
        True if the image is blank, False otherwise.
    """
    med = np.nanmedian(image)
    peakFlux = np.nanmax(image) - med

    # Absolute flux check
    if peakFlux > fluxMin:
        return False

    # SNR check with MAD-based robust std
    mad = np.nanmedian(np.abs(image - med))
    std = 1.4826 * mad
    if std <= 0:
        return True
    peakSnr = peakFlux / std
    return peakSnr < peakSnrMin
