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

import datetime
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import astropy.units as u
import matplotlib
import numpy as np
import numpy.typing as npt
import statsmodels.api as sm  # type: ignore[import-untyped]
from astro_metadata_translator import ObservationInfo
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.stats import mad_std
from astropy.time import Time
from deprecated.sphinx import deprecated
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression, RANSACRegressor  # type: ignore[import-untyped]

import lsst.afw.detection as afwDetect
import lsst.afw.detection as afwDetection
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.daf.base as dafBase
import lsst.daf.butler as dafButler
import lsst.geom as geom
import lsst.obs.lsst.translators
import lsst.pipe.base as pipeBase
import lsst.utils.packages as packageUtils
from lsst.afw.coord import Weather
from lsst.afw.detection import Footprint, FootprintSet
from lsst.daf.butler.cli.cliLog import CliLog
from lsst.obs.lsst import Latiss, LsstCam, LsstComCam, LsstComCamSim
from lsst.obs.lsst.translators.latiss import AUXTEL_LOCATION
from lsst.obs.lsst.translators.lsst import FILTER_DELIMITER, SIMONYI_LOCATION

from . import dateTime as newLocation
from .astrometry.utils import genericCameraHeaderToWcs

if TYPE_CHECKING:
    from lsst.afw.cameraGeom import Camera

__all__ = [
    "SIGMATOFWHM",
    "FWHMTOSIGMA",
    "EFD_CLIENT_MISSING_MSG",
    "GOOGLE_CLOUD_MISSING_MSG",
    "AUXTEL_LOCATION",
    "countPixels",
    "quickSmooth",
    "argMax2d",
    "dayObsIntToString",  # deprecated
    "dayObsSeqNumToVisitId",
    "getImageStats",
    "detectObjectsInExp",
    "fluxesFromFootprints",
    "fluxFromFootprint",
    "humanNameForCelestialObject",
    "getFocusFromHeader",
    "checkStackSetup",
    "setupLogging",
    "getCurrentDayObs_datetime",  # deprecated
    "getCurrentDayObs_int",  # deprecated
    "getCurrentDayObs_humanStr",  # deprecated
    "getExpRecordAge",  # deprecated
    "getSite",
    "getAltAzFromSkyPosition",
    "getExpPositionOffset",
    "starTrackerFileToExposure",
    "obsInfoToDict",
    "getFieldNameAndTileNumber",
    "getAirmassSeeingCorrection",
    "getFilterSeeingCorrection",  # deprecated
    "getBandpassSeeingCorrection",
    "getCdf",
    "getQuantiles",
    "digitizeData",
    "getBboxAround",
    "bboxToMatplotlibRectanle",
    "computeExposureId",
    "computeCcdExposureId",
    "getDetectorIds",
    "getImageArray",
    "getSunAngle",
    "RobustFitter",
]


SIGMATOFWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
FWHMTOSIGMA = 1 / SIGMATOFWHM

EFD_CLIENT_MISSING_MSG = (
    "ImportError: lsst_efd_client not found. Please install with:\n" "    pip install lsst-efd-client"
)

GOOGLE_CLOUD_MISSING_MSG = (
    "ImportError: Google cloud storage not found. Please install with:\n"
    "    pip install google-cloud-storage"
)


def countPixels(maskedImage: afwImage.MaskedImage, maskPlane: str) -> int:
    """Count the number of pixels in an image with a given mask bit set.

    Parameters
    ----------
    maskedImage : `lsst.afw.image.MaskedImage`
        The masked image,
    maskPlane : `str`
        The name of the bitmask.

    Returns
    -------
    count : `int``
        The number of pixels in with the selected mask bit
    """
    bit = maskedImage.mask.getPlaneBitMask(maskPlane)
    return len(np.where(np.bitwise_and(maskedImage.mask.array, bit))[0])


def quickSmooth(data: npt.NDArray[np.float64], sigma: float = 2) -> npt.NDArray[np.float64]:
    """Perform a quick smoothing of the image.

    Not to be used for scientific purposes, but improves the stretch and
    visual rendering of low SNR against the sky background in cutouts.

    Parameters
    ----------
    data : `np.array`
        The image data to smooth
    sigma : `float`, optional
        The size of the smoothing kernel.

    Returns
    -------
    smoothData : `np.array`
        The smoothed data
    """
    kernel = [sigma, sigma]
    smoothData = gaussian_filter(data, kernel, mode="constant")
    return smoothData


def argMax2d(array: npt.NDArray[np.float64]) -> tuple[tuple[float, float], bool, list[tuple[float, float]]]:
    """Get the index of the max value of an array and whether it's unique.

    If its not unique, returns a list of the other locations containing the
    maximum value, e.g. returns

    (12, 34), False, [(56,78), (910, 1112)]

    Parameters
    ----------
    array : `np.array`
        The data

    Returns
    -------
    maxLocation : `tuple`
        The coords of the first instance of the max value
    unique : `bool`
        Whether it's the only location
    otherLocations : `list` of `tuple`
        List of the other max values' locations, empty if False
    """
    uniqueMaximum = False
    maxCoords = np.where(array == np.max(array))
    # list of coords as tuples
    listMaxCoords = [(float(coord[0]), float(coord[1])) for coord in zip(*maxCoords)]
    if len(listMaxCoords) == 1:  # single unambiguous value
        uniqueMaximum = True

    return listMaxCoords[0], uniqueMaximum, listMaxCoords[1:]


@deprecated(
    reason="dayObsIntToString() has moved to lsst.summit.utils.dateTime. The function is unchanged, "
    "but you should change to import from there. This function will be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def dayObsIntToString(dayObs: int) -> str:
    """Convert an integer dayObs to a dash-delimited string.

    e.g. convert the hard to read 20210101 to 2021-01-01

    Parameters
    ----------
    dayObs : `int`
        The dayObs.

    Returns
    -------
    dayObs : `str`
        The dayObs as a string.
    """
    return newLocation.dayObsIntToString(dayObs)


def dayObsSeqNumToVisitId(dayObs: int, seqNum: int) -> int:
    """Get the visit id for a given dayObs/seqNum.

    Parameters
    ----------
    dayObs : `int`
        The dayObs.
    seqNum : `int`
        The seqNum.

    Returns
    -------
    visitId : `int`
        The visitId.

    Notes
    -----
    TODO: Remove this horrible hack once DM-30948 makes this possible
    programatically/via the butler.
    """
    if dayObs < 19700101 or dayObs > 35000101:
        raise ValueError(f"dayObs value {dayObs} outside plausible range")
    return int(f"{dayObs}{seqNum:05}")


def getImageStats(exp: afwImage.Exposure) -> pipeBase.Struct:
    """Calculate a grab-bag of stats for an image. Must remain fast.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The input exposure.

    Returns
    -------
    stats : `lsst.pipe.base.Struct`
        A container with attributes containing measurements and statistics
        for the image.
    """
    result = pipeBase.Struct()

    vi = exp.visitInfo
    expTime = vi.exposureTime
    md = exp.getMetadata()

    obj = vi.object
    mjd = vi.getDate().get()
    result.object = obj
    result.mjd = mjd

    fullFilterString = exp.filter.physicalLabel
    filt = fullFilterString.split(FILTER_DELIMITER)[0]
    grating = fullFilterString.split(FILTER_DELIMITER)[1]

    airmass = vi.getBoresightAirmass()
    rotangle = vi.getBoresightRotAngle().asDegrees()

    azAlt = vi.getBoresightAzAlt()
    az = azAlt[0].asDegrees()
    el = azAlt[1].asDegrees()

    result.expTime = expTime
    result.filter = filt
    result.grating = grating
    result.airmass = airmass
    result.rotangle = rotangle
    result.az = az
    result.el = el
    result.focus = md.get("FOCUSZ")

    data = exp.image.array
    result.maxValue = np.max(data)

    peak, uniquePeak, otherPeaks = argMax2d(data)
    result.maxPixelLocation = peak
    result.multipleMaxPixels = uniquePeak

    result.nBadPixels = countPixels(exp.maskedImage, "BAD")
    result.nSatPixels = countPixels(exp.maskedImage, "SAT")
    result.percentile99 = np.percentile(data, 99)
    result.percentile9999 = np.percentile(data, 99.99)

    sctrl = afwMath.StatisticsControl()
    sctrl.setNumSigmaClip(5)
    sctrl.setNumIter(2)
    statTypes = afwMath.MEANCLIP | afwMath.STDEVCLIP
    stats = afwMath.makeStatistics(exp.maskedImage, statTypes, sctrl)
    std, stderr = stats.getResult(afwMath.STDEVCLIP)
    mean, meanerr = stats.getResult(afwMath.MEANCLIP)

    result.clippedMean = mean
    result.clippedStddev = std

    return result


def detectObjectsInExp(
    exp: afwImage.Exposure, nSigma: float = 10, nPixMin: int = 10, grow: int = 0
) -> afwDetect.FootprintSet:
    """Quick and dirty object detection for an exposure.

    Return the footPrintSet for the objects in a preferably-postISR exposure.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure to detect objects in.
    nSigma : `float`
        The number of sigma for detection.
    nPixMin : `int`
        The minimum number of pixels in an object for detection.
    grow : `int`
        The number of pixels to grow the footprint by after detection.

    Returns
    -------
    footPrintSet : `lsst.afw.detection.FootprintSet`
        The set of footprints in the image.
    """
    median = np.nanmedian(exp.image.array)
    exp.image -= median

    threshold = afwDetect.Threshold(nSigma, afwDetect.Threshold.STDEV)
    footPrintSet = afwDetect.FootprintSet(exp.getMaskedImage(), threshold, "DETECTED", nPixMin)
    if grow > 0:
        isotropic = True
        footPrintSet = afwDetect.FootprintSet(footPrintSet, grow, isotropic)

    exp.image += median  # add back in to leave background unchanged
    return footPrintSet


def fluxesFromFootprints(
    footprints: afwDetect.FootprintSet | afwDetect.Footprint | Iterable[afwDetect.Footprint],
    parentImage: afwImage.Image,
    subtractImageMedian: bool = False,
) -> npt.NDArray[np.float64]:
    """Calculate the flux from a set of footprints, given the parent image,
    optionally subtracting the whole-image median from each pixel as a very
    rough background subtraction.

    Parameters
    ----------
    footprints : `lsst.afw.detection.FootprintSet` or
                 `lsst.afw.detection.Footprint` or
                 `iterable` of `lsst.afw.detection.Footprint`
        The footprints to measure.
    parentImage : `lsst.afw.image.Image`
        The parent image.
    subtractImageMedian : `bool`, optional
        Subtract a whole-image median from each pixel in the footprint when
        summing as a very crude background subtraction. Does not change the
        original image.

    Returns
    -------
    fluxes : `list` of `float`
        The fluxes for each footprint.

    Raises
    ------
    TypeError : raise for unsupported types.
    """
    median = 0
    if subtractImageMedian:
        median = np.nanmedian(parentImage.array)

    # poor person's single dispatch
    badTypeMsg = (
        "This function works with FootprintSets, single Footprints, and iterables of Footprints. "
        f"Got {type(footprints)}: {footprints}"
    )
    if isinstance(footprints, FootprintSet):
        footprints = footprints.getFootprints()
    elif isinstance(footprints, Sequence):
        if not isinstance(footprints[0], Footprint):
            raise TypeError(badTypeMsg)
    elif isinstance(footprints, Footprint):
        footprints = [footprints]
    else:
        raise TypeError(badTypeMsg)

    return np.array([fluxFromFootprint(fp, parentImage, backgroundValue=median) for fp in footprints])


def fluxFromFootprint(
    footprint: afwDetection.Footprint, parentImage: afwImage.Image, backgroundValue: float = 0
) -> float:
    """Calculate the flux from a footprint, given the parent image, optionally
    subtracting a single value from each pixel as a very rough background
    subtraction, e.g. the image median.

    Parameters
    ----------
    footprint : `lsst.afw.detection.Footprint`
        The footprint to measure.
    parentImage : `lsst.afw.image.Image`
        Image containing the footprint.
    backgroundValue : `bool`, optional
        The value to subtract from each pixel in the footprint when summing
        as a very crude background subtraction. Does not change the original
        image.

    Returns
    -------
    flux : `float`
        The flux in the footprint
    """
    if backgroundValue:  # only do the subtraction if non-zero for speed
        xy0 = parentImage.getBBox().getMin()
        return footprint.computeFluxFromArray(parentImage.array - backgroundValue, xy0)
    return footprint.computeFluxFromImage(parentImage)


def humanNameForCelestialObject(objName: str) -> list[str]:
    """Returns a list of all human names for obj, or [] if none are found.

    Parameters
    ----------
    objName : `str`
        The/a name of the object.

    Returns
    -------
    names : `list` of `str`
        The names found for the object
    """
    from astroquery.simbad import Simbad

    results = []
    try:
        simbadResult = Simbad.query_objectids(objName)
        for row in simbadResult:
            if row["ID"].startswith("NAME"):
                results.append(row["ID"].replace("NAME ", ""))
        return results
    except Exception:
        return []  # same behavior as for found but un-named objects


def _getAltAzZenithsFromSeqNum(
    butler: dafButler.Butler, dayObs: int, seqNumList: list[int]
) -> tuple[list[float], list[float], list[float]]:
    """Get the alt, az and zenith angle for the seqNums of a given dayObs.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The butler to query.
    dayObs : `int`
        The dayObs.
    seqNumList : `list` of `int`
        The seqNums for which to return the alt, az and zenith

    Returns
    -------
    azimuths : `list` of `float`
        List of the azimuths for each seqNum
    elevations : `list` of `float`
        List of the elevations for each seqNum
    zeniths : `list` of `float`
        List of the zenith angles for each seqNum
    """
    azimuths, elevations, zeniths = [], [], []
    for seqNum in seqNumList:
        md = butler.get("raw.metadata", day_obs=dayObs, seq_num=seqNum, detector=0)
        obsInfo = ObservationInfo(md)
        assert obsInfo.altaz_begin is not None, f"altaz_begin is None for seqNum={seqNum}"
        alt = obsInfo.altaz_begin.alt.value
        az = obsInfo.altaz_begin.az.value
        elevations.append(alt)
        zeniths.append(90 - alt)
        azimuths.append(az)
    return azimuths, elevations, zeniths


def getFocusFromHeader(exp: afwImage.Exposure) -> float | None:
    """Get the raw focus value from the header.

    Parameters
    ----------
    exp : `lsst.afw.image.exposure`
        The exposure.

    Returns
    -------
    focus : `float` or `None`
        The focus value if found, else ``None``.
    """
    md = exp.getMetadata()
    if "FOCUSZ" in md:
        return md["FOCUSZ"]
    return None


def checkStackSetup() -> None:
    """Check which weekly tag is being used and which local packages are setup.

    Designed primarily for use in notbooks/observing, this prints the weekly
    tag(s) are setup for lsst_distrib, and lists any locally setup packages and
    the path to each.

    Notes
    -----
    Uses print() instead of logger messages as this should simply print them
    without being vulnerable to any log messages potentially being diverted.
    """
    packages = packageUtils.getEnvironmentPackages(include_all=True)

    lsstDistribHashAndTags = packages["lsst_distrib"]  # looks something like 'g4eae7cb9+1418867f (w_2022_13)'
    lsstDistribTags = lsstDistribHashAndTags.split()[1]
    if len(lsstDistribTags.split()) == 1:
        tag = lsstDistribTags.replace("(", "")
        tag = tag.replace(")", "")
        print(f"You are running {tag} of lsst_distrib")
    else:  # multiple weekly tags found for lsst_distrib!
        print(f"The version of lsst_distrib you have is compatible with: {lsstDistribTags}")

    localPackages = []
    localPaths = []
    for package, tags in packages.items():
        if tags.startswith("LOCAL:"):
            path = tags.split("LOCAL:")[1]
            path = path.split("@")[0]  # don't need the git SHA etc
            localPaths.append(path)
            localPackages.append(package)

    if localPackages:
        print("\nLocally setup packages:")
        print("-----------------------")
        maxLen = max(len(package) for package in localPackages)
        for package, path in zip(localPackages, localPaths):
            print(f"{package:<{maxLen}s} at {path}")
    else:
        print("\nNo locally setup packages (using a vanilla stack)")


def setupLogging(longlog: bool = False) -> None:
    """Setup logging in the same way as one would get from pipetask run.

    Code that isn't run through the butler CLI defaults to WARNING level
    messages and no logger names. This sets the behaviour to follow whatever
    the pipeline default is, currently
        <logger_name> <level>: <message> e.g.
        lsst.isr INFO: Masking defects.
    """
    CliLog.initLog(longlog=longlog)


@deprecated(
    reason="getCurrentDayObs_datetime() has moved to lsst.summit.utils.dateTime. The function is unchanged,"
    " but renamed to getCurrentDayObsDatetime and you should change to import from there. This function will"
    " be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def getCurrentDayObs_datetime() -> datetime.date:
    """Get the current day_obs - the observatory rolls the date over at UTC-12

    Returned as datetime.date(2022, 4, 28)
    """
    return newLocation.getCurrentDayObsDatetime()


@deprecated(
    reason="getCurrentDayObs_int() has moved to lsst.summit.utils.dateTime. The function is unchanged,"
    " but renamed to getCurrentDayObsInt and you should change to import from there. This function will"
    " be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def getCurrentDayObs_int() -> int:
    """Return the current dayObs as an int in the form 20220428"""
    return newLocation.getCurrentDayObsInt()


@deprecated(
    reason="getCurrentDayObs_humanStr() has moved to lsst.summit.utils.dateTime. The function is unchanged,"
    " but renamed to getCurrentDayObsHumanStr and you should change to import from there. This function will"
    " be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def getCurrentDayObs_humanStr() -> str:
    """Return the current dayObs as a string in the form '2022-04-28'"""
    return newLocation.getCurrentDayObsHumanStr()


@deprecated(
    reason="getExpRecordAge() has moved to lsst.summit.utils.dateTime. The function is unchanged, "
    "but you should change to import from there. This function will be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def getExpRecordAge(expRecord: dafButler.DimensionRecord) -> float:
    """Get the time, in seconds, since the end of exposure.

    Parameters
    ----------
    expRecord : `lsst.daf.butler.DimensionRecord`
        The exposure record.

    Returns
    -------
    age : `float`
        The age of the exposure, in seconds.
    """
    return -1 * (expRecord.timespan.end - Time.now()).sec


def getSite() -> str:
    """Returns where the code is running.

    Returns
    -------
    location : `str`
        One of:
        ['tucson', 'summit', 'base', 'staff-rsp', 'rubin-devl', 'jenkins',
         'usdf-k8s', 'gha', 'local']

        ``gha`` is returned when running under GitHub Actions (signalled by
        ``RAPID_ANALYSIS_LOCATION=gha``, which the rubintv_production CI
        workflow sets). Like ``local``, it means "no functioning site
        available"; the dedicated value lets CI-only branching (skipping
        butler/uploader-bound tests) be expressed without dragging
        developer laptops into the same opt-out.

        ``local`` is returned when none of the known sites can be detected,
        which usually means the code is running on a developer's laptop or
        in some other environment without the standard env vars set. Callers
        that need a real site (e.g. to talk to S3, EFD or a Butler repo)
        should treat ``local`` (and ``gha``) as "no functioning site
        available" and either raise or skip; callers that only branch on
        ``site == "summit"`` style checks can treat them as a no-op
        fallthrough.
    """
    # All nublado instances guarantee that EXTERNAL_URL is set and uniquely
    # identifies it.
    location = os.getenv("EXTERNAL_INSTANCE_URL", "")
    if location == "https://tucson-teststand.lsst.codes":
        return "tucson"
    elif location == "https://summit-lsp.lsst.codes":
        return "summit"
    elif location == "https://base-lsp.lsst.codes":
        return "base"
    elif location == "https://usdf-rsp.slac.stanford.edu":
        return "staff-rsp"
    elif location == "https://usdf-rsp-dev.slac.stanford.edu":
        return "staff-rsp"  # we don't care this is the dev RSP, it's basically the same env wrt paths etc

    # if no EXTERNAL_URL, try HOSTNAME to see if we're on the dev nodes
    # it is expected that this will be extensible to SLAC
    hostname = os.getenv("HOSTNAME", "")
    interactiveNodes = ("sdfrome", "sdfiana")
    if hostname.startswith(interactiveNodes):
        return "rubin-devl"
    elif hostname == "htcondor.ls.lsst.org":
        return "base"
    elif hostname == "htcondor.cp.lsst.org":
        return "summit"

    jenkinsHome = os.getenv("JENKINS_HOME", "")
    if jenkinsHome != "":
        return "jenkins"

    # we're probably inside a k8s pod doing rapid analysis work at this point
    location = os.getenv("RAPID_ANALYSIS_LOCATION", "")
    if location == "TTS":
        return "tucson"
    if location == "BTS":
        return "base"
    if location == "SUMMIT":
        return "summit"
    if location == "USDF":
        return "usdf-k8s"
    # ``gha`` is set by the rubintv_production GitHub Actions workflow.
    # Compared case-insensitively because the existing TTS/BTS/SUMMIT/USDF
    # values above are all upper-case shouts and "gha" reads more
    # naturally lower-case in the workflow file -- accept either.
    if location.lower() == "gha":
        return "gha"

    # No known site detected — assume this is a developer machine.
    return "local"


def getAltAzFromSkyPosition(
    skyPos: geom.SpherePoint,
    visitInfo: afwImage.VisitInfo,
    doCorrectRefraction: bool = False,
    wavelength: float = 500.0,
    pressureOverride: float | None = None,
    temperatureOverride: float | None = None,
    relativeHumidityOverride: float | None = None,
) -> tuple[geom.Angle, geom.Angle]:
    """Get the alt/az from the position on the sky and the time and location
    of the observation.

    The temperature, pressure and relative humidity are taken from the
    visitInfo by default, but can be individually overridden as needed. It
    should be noted that the visitInfo never contains a nominal wavelength, and
    so this takes a default value of 500nm.

    Parameters
    ----------
    skyPos : `lsst.geom.SpherePoint`
        The position on the sky.
    visitInfo : `lsst.afw.image.VisitInfo`
        The visit info containing the time of the observation.
    doCorrectRefraction : `bool`, optional
        Correct for the atmospheric refraction?
    wavelength : `float`, optional
        The nominal wavelength in nanometers (e.g. 500.0), as a float.
    pressureOverride : `float`, optional
        The pressure, in bars (e.g. 0.770), to override the value supplied in
        the visitInfo, as a float.
    temperatureOverride : `float`, optional
        The temperature, in Celsius (e.g. 10.0), to override the value supplied
        in the visitInfo, as a float.
    relativeHumidityOverride : `float`, optional
        The relativeHumidity in the range 0..1 (i.e. not as a percentage), to
        override the value supplied in the visitInfo, as a float.

    Returns
    -------
    alt : `lsst.geom.Angle`
        The altitude.
    az : `lsst.geom.Angle`
        The azimuth.
    """
    skyLocation = SkyCoord(skyPos.getRa().asRadians(), skyPos.getDec().asRadians(), unit=u.rad)
    long = visitInfo.observatory.getLongitude()
    lat = visitInfo.observatory.getLatitude()
    ele = visitInfo.observatory.getElevation()
    earthLocation = EarthLocation.from_geodetic(long.asDegrees(), lat.asDegrees(), ele)

    refractionKwargs = {}
    if doCorrectRefraction:
        # wavelength is never supplied in the visitInfo so always take this
        wavelength = wavelength * u.nm

        if pressureOverride:
            pressure = pressureOverride
        else:
            pressure = visitInfo.weather.getAirPressure()
            # ObservationInfos (which are the "source of truth" use pascals) so
            # convert from pascals to bars
            pressure /= 100000.0
        pressure = pressure * u.bar

        if temperatureOverride:
            temperature = temperatureOverride
        else:
            temperature = visitInfo.weather.getAirTemperature()
        temperature = temperature * u.deg_C

        if relativeHumidityOverride:
            relativeHumidity = relativeHumidityOverride
        else:
            relativeHumidity = visitInfo.weather.getHumidity() / 100.0  # this is in percent
        relativeHumidity = relativeHumidity

        refractionKwargs = dict(
            pressure=pressure, temperature=temperature, relative_humidity=relativeHumidity, obswl=wavelength
        )

    obsTime = visitInfo.date.toAstropy()
    altAz = AltAz(obstime=obsTime, location=earthLocation, **refractionKwargs)

    obsAltAz = skyLocation.transform_to(altAz)
    alt = geom.Angle(obsAltAz.alt.degree, geom.degrees)
    az = geom.Angle(obsAltAz.az.degree, geom.degrees)

    return alt, az


def getExpPositionOffset(
    exp1: afwImage.Exposure,
    exp2: afwImage.Exposure,
    useWcs: bool = True,
    allowDifferentPlateScales: bool = False,
) -> pipeBase.Struct:
    """Get the change in sky position between two exposures.

    Given two exposures, calculate the offset on the sky between the images.
    If useWcs then use the (fitted or unfitted) skyOrigin from their WCSs, and
    calculate the alt/az from the observation times, otherwise use the nominal
    values in the exposures' visitInfos. Note that if using the visitInfo
    values that for a given pointing the ra/dec will be ~identical, regardless
    of whether astrometric fitting has been performed.

    Values are given as exp1-exp2.

    Parameters
    ----------
    exp1 : `lsst.afw.image.Exposure`
        The first exposure.
    exp2 : `lsst.afw.image.Exposure`
        The second exposure.
    useWcs : `bool`
        Use the WCS for the ra/dec and alt/az if True, else use the nominal/
        boresight values from the exposures' visitInfos.
    allowDifferentPlateScales : `bool`, optional
        Use to disable checking that plate scales are the same. Generally,
        differing plate scales would indicate an error, but where blind-solving
        has been undertaken during commissioning plate scales can be different
        enough to warrant setting this to ``True``.

    Returns
    -------
    offsets : `lsst.pipe.base.Struct`
        A struct containing the offsets:
            ``deltaRa``
                The diference in ra (`lsst.geom.Angle`)
            ``deltaDec``
                The diference in dec (`lsst.geom.Angle`)
            ``deltaAlt``
                The diference in alt (`lsst.geom.Angle`)
            ``deltaAz``
                The diference in az (`lsst.geom.Angle`)
            ``deltaPixels``
                The diference in pixels (`float`)
    """

    wcs1 = exp1.getWcs()
    wcs2 = exp2.getWcs()
    pixScaleArcSec = wcs1.getPixelScale().asArcseconds()
    if not allowDifferentPlateScales:
        assert np.isclose(
            pixScaleArcSec, wcs2.getPixelScale().asArcseconds()
        ), "Pixel scales in the exposures differ."

    if useWcs:
        p1 = wcs1.getSkyOrigin()
        p2 = wcs2.getSkyOrigin()
        alt1, az1 = getAltAzFromSkyPosition(p1, exp1.getInfo().getVisitInfo())
        alt2, az2 = getAltAzFromSkyPosition(p2, exp2.getInfo().getVisitInfo())
        ra1 = p1[0]
        ra2 = p2[0]
        dec1 = p1[1]
        dec2 = p2[1]
    else:
        az1 = exp1.visitInfo.boresightAzAlt[0]
        az2 = exp2.visitInfo.boresightAzAlt[0]
        alt1 = exp1.visitInfo.boresightAzAlt[1]
        alt2 = exp2.visitInfo.boresightAzAlt[1]

        ra1 = exp1.visitInfo.boresightRaDec[0]
        ra2 = exp2.visitInfo.boresightRaDec[0]
        dec1 = exp1.visitInfo.boresightRaDec[1]
        dec2 = exp2.visitInfo.boresightRaDec[1]

        p1 = exp1.visitInfo.boresightRaDec
        p2 = exp2.visitInfo.boresightRaDec

    angular_offset = p1.separation(p2).asArcseconds()
    deltaPixels = angular_offset / pixScaleArcSec

    ret = pipeBase.Struct(
        deltaRa=(ra1 - ra2).wrapNear(geom.Angle(0.0)),
        deltaDec=dec1 - dec2,
        deltaAlt=alt1 - alt2,
        deltaAz=(az1 - az2).wrapNear(geom.Angle(0.0)),
        deltaPixels=deltaPixels,
    )

    return ret


def starTrackerFileToExposure(filename: str, logger: logging.Logger | None = None) -> afwImage.Exposure:
    """Read the exposure from the file and set the wcs from the header.

    Parameters
    ----------
    filename : `str`
        The full path to the file.
    logger : `logging.Logger`, optional
        The logger to use for errors, created if not supplied.

    Returns
    -------
    exp : `lsst.afw.image.Exposure`
        The exposure.
    """
    if not logger:
        logger = logging.getLogger(__name__)
    exp = afwImage.ExposureF(filename)
    try:
        wcs = genericCameraHeaderToWcs(exp)
        exp.setWcs(wcs)
    except Exception as e:
        logger.warning(f"Failed to set wcs from header: {e}")

    # for some reason the date isn't being set correctly
    # DATE-OBS is present in the original header, but it's being
    # stripped out and somehow not set (plus it doesn't give the midpoint
    # of the exposure), so set it manually from the midpoint here
    try:
        newArgs = {}  # dict to unpack into visitInfo.copyWith - fill it with whatever needs to be replaced
        md = exp.getMetadata()

        begin = datetime.datetime.fromisoformat(md["DATE-BEG"])
        end = datetime.datetime.fromisoformat(md["DATE-END"])
        duration = end - begin
        mid = begin + duration / 2
        newTime = dafBase.DateTime(mid.isoformat(), dafBase.DateTime.Timescale.TAI)
        newArgs["date"] = newTime

        # AIRPRESS is being set as PRESSURE so afw doesn't pick it up
        # once we're using the butler for data we will just set it to take
        # PRESSURE in the translator instead of this
        weather = exp.visitInfo.getWeather()
        oldPressure = weather.getAirPressure()
        if not np.isfinite(oldPressure):
            pressure = md.get("PRESSURE")
            if pressure is not None:
                logger.info("Patching the weather info using the PRESSURE header keyword")
                newWeather = Weather(weather.getAirTemperature(), pressure, weather.getHumidity())
                newArgs["weather"] = newWeather

        if newArgs:
            newVi = exp.visitInfo.copyWith(**newArgs)
            exp.info.setVisitInfo(newVi)
    except Exception as e:
        logger.warning(f"Failed to set date from header: {e}")

    return exp


def obsInfoToDict(obsInfo: ObservationInfo) -> dict:
    """Convert an ObservationInfo to a dict.

    Parameters
    ----------
    obsInfo : `astro_metadata_translator.ObservationInfo`
        The ObservationInfo to convert.

    Returns
    -------
    obsInfoDict : `dict`
        The ObservationInfo as a dict.
    """
    return {prop: getattr(obsInfo, prop) for prop in obsInfo.all_properties.keys()}


def getFieldNameAndTileNumber(
    field: str, warn: bool = True, logger: logging.Logger | None = None
) -> tuple[str, int | None]:
    """Get the tile name and number of an observed field.

    It is assumed to always be appended, with an underscore, to the rest of the
    field name. Returns the name and number as a tuple, or the name unchanged
    if no tile number is found.

    Parameters
    ----------
    field : `str`
        The name of the field

    Returns
    -------
    fieldName : `str`
        The name of the field without the trailing tile number, if present.
    tileNum : `int`
        The number of the tile, as an integer, or ``None`` if not found.
    """
    if warn and not logger:
        logger = logging.getLogger("lsst.summit.utils.utils.getFieldNameAndTileNumber")

    if "_" not in field:
        if warn and logger is not None:
            logger.warning(
                f"Field {field} does not contain an underscore," " so cannot determine the tile number."
            )
        return field, None

    try:
        fieldParts = field.split("_")
        fieldNum = int(fieldParts[-1])
    except ValueError:
        if warn and logger is not None:
            logger.warning(
                f"Field {field} does not contain only an integer after the final underscore"
                " so cannot determine the tile number."
            )
        return field, None

    return "_".join(fieldParts[:-1]), fieldNum


def getAirmassSeeingCorrection(airmass: float) -> float:
    """Get the correction factor for seeing due to airmass.

    Parameters
    ----------
    airmass : `float`
        The airmass, greater than or equal to 1.

    Returns
    -------
    correctionFactor : `float`
        The correction factor to apply to the seeing.

    Raises
    ------
        ValueError raised for unphysical airmasses.
    """
    if airmass < 1:
        raise ValueError(f"Invalid airmass: {airmass}")
    return airmass ** (-0.6)


@deprecated(
    reason=". Will be removed after v28.0.",
    version="v27.0",
    category=FutureWarning,
)
def getFilterSeeingCorrection(filterName: str) -> float:
    """Get the correction factor for seeing due to a filter.

    Parameters
    ----------
    filterName : `str`
        The name of the filter, e.g. 'SDSSg_65mm'.

    Returns
    -------
    correctionFactor : `float`
        The correction factor to apply to the seeing.

    Raises
    ------
        ValueError raised for unknown filters.
    """
    return getBandpassSeeingCorrection(filterName)


def getBandpassSeeingCorrection(filterName: str) -> float:
    """Get the correction factor for seeing due to a filter.

    Parameters
    ----------
    filterName : `str`
        The name of the filter, e.g. 'SDSSg_65mm'.

    Returns
    -------
    correctionFactor : `float`
        The correction factor to apply to the seeing.

    Raises
    ------
        ValueError raised for unknown filters.
    """
    match filterName:
        case "SDSSg_65mm":  # LATISS
            return (474.41 / 500.0) ** 0.2
        case "SDSSr_65mm":  # LATISS
            return (628.47 / 500.0) ** 0.2
        case "SDSSi_65mm":  # LATISS
            return (769.51 / 500.0) ** 0.2
        case "SDSSz_65mm":  # LATISS
            return (871.45 / 500.0) ** 0.2
        case "SDSSy_65mm":  # LATISS
            return (986.8 / 500.0) ** 0.2
        case "u_02":  # ComCam
            return (370.697 / 500.0) ** 0.2
        case "g_01":  # ComCam
            return (476.359 / 500.0) ** 0.2
        case "r_03":  # ComCam
            return (619.383 / 500.0) ** 0.2
        case "i_06":  # ComCam
            return (754.502 / 500.0) ** 0.2
        case "z_03":  # ComCam
            return (866.976 / 500.0) ** 0.2
        case "y_04":  # ComCam
            return (972.713 / 500.0) ** 0.2
        case "u_24":  # LSSTCam
            return (365.1397 / 500.0) ** 0.2
        case "g_6":  # LSSTCam
            return (475.6104 / 500.0) ** 0.2
        case "r_57":  # LSSTCam
            return (618.6055 / 500.0) ** 0.2
        case "i_39":  # LSSTCam
            return (753.5490 / 500.0) ** 0.2
        case "z_20":  # LSSTCam
            return (868.0935 / 500.0) ** 0.2
        case "y_10":  # LSSTCam
            return (972.5979 / 500.0) ** 0.2
        case _:
            raise ValueError(f"Unknown filter name: {filterName}")


def getCdf(data: np.ndarray, scale: int, nBinsMax: int = 300_000) -> tuple[np.ndarray | float, float, float]:
    """Return an approximate cumulative distribution function scaled to
    the [0, scale] range.

    If the input data is all nan, then the output cdf will be nan as well as
    the min and max values.

    Parameters
    ----------
    data : `np.array`
        The input data.
    scale : `int`
        The scaling range of the output.
    nBinsMax : `int`, optional
        Maximum number of bins to use.

    Returns
    -------
    cdf : `np.array` of `int`
        A monotonically increasing sequence that represents a scaled
        cumulative distribution function, starting with the value at
        minVal, then at (minVal + 1), and so on.
    minVal : `float`
        An integer smaller than the minimum value in the input data.
    maxVal : `float`
        An integer larger than the maximum value in the input data.
    """
    flatData = data.ravel()
    size = flatData.size - np.count_nonzero(np.isnan(flatData))

    minVal = np.floor(np.nanmin(flatData))
    maxVal = np.ceil(np.nanmax(flatData)) + 1.0

    if np.isnan(minVal) or np.isnan(maxVal):
        # if either the min or max are nan, then the data is all nan as we're
        # using nanmin and nanmax. Given this, we can't calculate a cdf, so
        # return nans for all values
        return np.nan, np.nan, np.nan

    nBins = np.clip(int(maxVal) - int(minVal), 1, nBinsMax)

    hist, binEdges = np.histogram(flatData, bins=nBins, range=(int(minVal), int(maxVal)))

    cdf = (scale * np.cumsum(hist) / size).astype(np.int64)
    return cdf, minVal, maxVal


def getQuantiles(data: npt.NDArray[np.float64], nColors: int) -> npt.NDArray[np.floating[Any]]:
    """Get a set of boundaries that equally distribute data into
    nColors intervals. The output can be used to make a colormap of nColors
    colors.

    This is equivalent to using the numpy function:
        np.nanquantile(data, np.linspace(0, 1, nColors + 1))
    but with a coarser precision, yet sufficient for our use case. This
    implementation gives a significant speed-up. In the case of large
    ranges, np.nanquantile is used because it is more memory efficient.

    If all elements of ``data`` are nan then the output ``boundaries`` will
    also all be ``nan`` to keep the interface consistent.

    Parameters
    ----------
    data : `np.array`
        The input image data.
    nColors : `int`
        The number of intervals to distribute data into.

    Returns
    -------
    boundaries: `list` of `float`
        A monotonically increasing sequence of size (nColors + 1). These are
        the edges of nColors intervals.
    """
    dataRange = np.nanmax(data) - np.nanmin(data)
    if dataRange > 300_000:
        # Use slower but memory efficient nanquantile
        logger = logging.getLogger(__name__)
        logger.warning(f"Data range is very large ({dataRange}); using slower quantile code.")
        boundaries = np.nanquantile(data, np.linspace(0, 1, nColors + 1))
    else:
        cdf, minVal, maxVal = getCdf(data, nColors)
        if np.isnan(minVal):  # cdf calculation has failed because all data is nan
            return np.asarray([np.nan for _ in range(nColors)])
        assert isinstance(cdf, np.ndarray), "cdf is not an np.ndarray"
        scale = (maxVal - minVal) / len(cdf)

        boundaries = np.asarray([np.argmax(cdf >= i) * scale + minVal for i in range(nColors)] + [maxVal])

    return boundaries


def digitizeData(data: npt.NDArray[np.float64], nColors: int = 256) -> npt.NDArray[np.integer]:
    """
    Scale data into nColors using its cumulative distribution function.

    Parameters
    ----------
    data : `np.array`
        The input image data.
    nColors : `int`
        The number of intervals to distribute data into.

    Returns
    -------
    data: `np.array` of `int`
        Scaled data in the [0, nColors - 1] range.
    """
    cdf, minVal, maxVal = getCdf(data, nColors - 1)
    assert isinstance(cdf, np.ndarray), "cdf is not an np.ndarray"
    scale = (maxVal - minVal) / len(cdf)
    bins = np.floor((data * scale - minVal)).astype(np.int64)
    return cdf[bins]


def getBboxAround(centroid: geom.Point, boxSize: int, exp: afwImage.Exposure) -> geom.Box2I:
    """Get a bbox centered on a point, clipped to the exposure.

    If the bbox would extend beyond the bounds of the exposure it is clipped to
    the exposure, resulting in a non-square bbox.

    Parameters
    ----------
    centroid : `lsst.geom.Point`
        The source centroid.
    boxsize : `int`
        The size of the box to centre around the centroid.
    exp : `lsst.afw.image.Exposure`
        The exposure, so the bbox can be clipped to not overrun the bounds.

    Returns
    -------
    bbox : `lsst.geom.Box2I`
        The bounding box, centered on the centroid unless clipping to the
        exposure causes it to be non-square.
    """
    bbox = geom.BoxI().makeCenteredBox(centroid, geom.Extent2I(boxSize, boxSize))
    bbox = bbox.clippedTo(exp.getBBox())
    return bbox


def bboxToMatplotlibRectanle(bbox: geom.Box2I | geom.Box2D) -> matplotlib.patches.Rectangle:
    """Convert a bbox to a matplotlib Rectangle for plotting.

    Parameters
    ----------
    results : `lsst.geom.Box2I` or `lsst.geom.Box2D`
        The bbox to convert.

    Returns
    -------
    rectangle : `matplotlib.patches.Rectangle`
        The rectangle.
    """
    ll = bbox.minX, bbox.minY
    width, height = bbox.getDimensions()
    return Rectangle(ll, width, height)


def computeExposureId(instrument: str, controller: str, dayObs: int, seqNum: int) -> int:
    instrument = instrument.lower()
    if instrument == "latiss":
        return lsst.obs.lsst.translators.LatissTranslator.compute_exposure_id(dayObs, seqNum, controller)
    elif instrument == "lsstcomcam":
        return lsst.obs.lsst.translators.LsstComCamTranslator.compute_exposure_id(dayObs, seqNum, controller)
    elif instrument == "lsstcomcamsim":
        return lsst.obs.lsst.translators.LsstComCamSimTranslator.compute_exposure_id(
            dayObs, seqNum, controller
        )
    elif instrument == "lsstcam":
        return lsst.obs.lsst.translators.LsstCamTranslator.compute_exposure_id(dayObs, seqNum, controller)
    else:
        raise ValueError("Unknown instrument {instrument}")


def computeCcdExposureId(instrument: str, exposureId: int, detector: int) -> int:
    instrument = instrument.lower()
    if instrument == "latiss":
        if detector != 0:
            raise ValueError("Invalid detector {detector} for LATISS")
        return lsst.obs.lsst.translators.LatissTranslator.compute_detector_exposure_id(exposureId, detector)
    elif instrument == "lsstcomcam":
        if detector < 0 or detector >= 9:
            raise ValueError("Invalid detector {detector} for LSSTComCam")
        return lsst.obs.lsst.translators.LsstComCamTranslator.compute_detector_exposure_id(
            exposureId, detector
        )
    elif instrument == "lsstcomcamsim":
        if detector < 0 or detector >= 9:
            raise ValueError("Invalid detector {detector} for LSSTComCamSim")
        return lsst.obs.lsst.translators.LsstComCamSimTranslator.compute_detector_exposure_id(
            exposureId, detector
        )
    elif instrument == "lsstcam":
        if detector < 0 or detector >= 205:
            raise ValueError("Invalid detector {detector} for LSSTCam")
        return lsst.obs.lsst.translators.LsstCamTranslator.compute_detector_exposure_id(exposureId, detector)
    else:
        raise ValueError("Unknown instrument {instrument}")


def getDetectorIds(instrumentName: str) -> list[int]:
    """Get a list of detector IDs for a given instrument.

    Parameters
    ----------
    instrumentName : `str`
        The name of the instrument.

    Returns
    -------
    detectorIds : `list` of `int`
        The list of detector IDs.
    """
    camera = getCameraFromInstrumentName(instrumentName)
    return [detector.getId() for detector in camera]


def getCameraFromInstrumentName(instrumentName: str) -> Camera:
    """Get the camera object given the instrument name (case insenstive).

    Parameters
    ----------
    instrumentName : `str`
        The name of the instrument, e.g. "LATISS" or "LSSTCam". Case
        insenstive.

    Returns
    -------
    camera: `lsst.afw.cameraGeom.Camera`
        The camera object corresponding to the instrument name.

    Raises
    ------
    ValueError
        If the instrument name is not supported.
    """

    _instrument = instrumentName.lower()

    match _instrument:
        case "lsstcam":
            camera = LsstCam.getCamera()
        case "lsstcomcam":
            camera = LsstComCam.getCamera()
        case "lsstcomcamsim":
            camera = LsstComCamSim.getCamera()
        case "latiss":
            camera = Latiss.getCamera()
        case _:
            raise ValueError(f"Unsupported instrument: {instrumentName}")
    return camera


def calcEclipticCoords(ra: float, dec: float) -> tuple[float, float]:
    """Get the ecliptic coordinates of the specified ra and dec.

    Transform J2000 (ra, dec), both in degrees, to
    IAU1976 Ecliptic coordinates (also returning degrees).

    Matches the results of:

      from astropy.coordinates import SkyCoord, HeliocentricEclipticIAU76
        from dataclasses import dataclass
      import astropy.units as u

      p = SkyCoord(ra=ra0*u.deg, dec=dec0*u.deg, distance=1*u.au, frame='hcrs')
      ecl = p.transform_to(HeliocentricEclipticIAU76)
      print(ecl.lon.value, ecl.lat.value)

    except that it's fast.

    Parameters
    ----------
    ra : `float`
        The right ascension, in degrees.
    dec : `float`
        The declination, in degrees.

    Returns
    -------
    lambda : `float`
        The ecliptic longitude in degrees.
    beta : `float`
        The ecliptic latitude in degrees.
    """

    ra, dec = np.deg2rad(ra), np.deg2rad(dec)

    # IAU 1976 obliquity
    epsilon = np.deg2rad(23.43929111111111)
    cos_eps, sin_eps = np.cos(epsilon), np.sin(epsilon)

    sra, cra = np.sin(ra), np.cos(ra)
    sdec, cdec = np.sin(dec), np.cos(dec)

    lambda_ = np.arctan2((sra * cos_eps + sdec / cdec * sin_eps), cra)
    beta = np.arcsin(sdec * cos_eps - cdec * sin_eps * sra)

    # normalize
    if lambda_ < 0:
        lambda_ += np.pi * 2

    return (np.rad2deg(lambda_), np.rad2deg(beta))


def getImageArray(
    inputData: np.ndarray | afwImage.Exposure | afwImage.Image | afwImage.MaskedImage,
) -> np.ndarray:
    """Get the image data from anything image-like.

    Parameters
    ----------
    inputData : `numpy.ndarray`, `lsst.afw.image.Exposure`,
        `lsst.afw.image.Image`, or `lsst.afw.image.MaskedImage`
        The input data.
    Returns
    -------
    imageData : `numpy.ndarray`
        The image data.
    Raises
    ------
    TypeError
        If the input data is not a numpy array, lsst.afw.image.Exposure,
        lsst.afw.image.Image, or lsst.afw.image.MaskedImage.
    """
    match inputData:
        case np.ndarray():
            imageData = inputData
        case afwImage.MaskedImage():
            imageData = inputData.image.array
        case afwImage.Image():
            imageData = inputData.array
        case afwImage.Exposure():
            imageData = inputData.image.array
        case _:
            raise TypeError(
                "This function accepts numpy array, lsst.afw.image.Exposure components."
                f" Got {type(inputData)}"
            )
    return imageData


def getSunAngle(time: Time | None = None) -> float:
    """Get the angle of the sun to the horizon at the specified time.

    If no time is specified, the current time is used. Positive numbers means
    the sun is above the horizon, negative means it is below.

    Returns
    -------
    sun_alt : `float`
        The angle of the sun to the horizon, in degrees.
    """
    if time is None:
        time = Time.now()
    sun_altaz = get_sun(time).transform_to(AltAz(obstime=time, location=SIMONYI_LOCATION))
    return sun_altaz.alt.deg


@dataclass(slots=True)
class RobustFitResult:
    slope: float
    intercept: float
    scatter: float
    slopePValue: float
    slopeStdErr: float
    slopeTValue: float
    interceptPValue: float
    interceptStdErr: float
    interceptTValue: float
    outlierMask: np.ndarray


class RobustFitter:
    """
    Robust linear fit using RANSAC + OLS, reusable across datasets.

    This class fits a robust linear trend to x and y using scikit-learn's
    RANSAC for inlier detection, followed by an OLS fit on inliers to compute
    slope, intercept, scatter, standard errors, t-values, and p-values.
    Configuration (e.g., minSamples) is set at init. Call `fit()` per dataset.
    Results are returned as a `RobustFitResult` and stored internally for later
    access (e.g., plotting).

    Parameters
    ----------
    minSamples : `float`, optional
        Minimum fraction of samples chosen randomly from the original data.
    """

    def __init__(self, *, minSamples: float = 0.2) -> None:
        self.minSamples = minSamples
        self._clearState()

    def _clearState(self) -> None:
        """Reset stored fit state."""
        self.model: Any = None
        self.olsModel: Any = None
        self.outlierMask: np.ndarray | None = None
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.slope: float = np.nan
        self.intercept: float = np.nan
        self.ciSlope: tuple[float, float] = (np.nan, np.nan)
        self.ciIntercept: tuple[float, float] = (np.nan, np.nan)
        self.slopePValue = np.nan
        self.slopeStdErr = np.nan
        self.slopeTValue = np.nan
        self.scatter = np.nan

    @staticmethod
    def _defaultResidualThreshold(y: np.ndarray) -> float:
        """Compute default residual threshold from finite y values."""
        yFinite = np.asarray(y)[np.isfinite(y)]
        if yFinite.size == 0:
            raise ValueError("Cannot compute residual threshold: no finite y.")
        return 1.5 * float(mad_std(yFinite))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        residualThreshold: float | None = None,
        randomState: int = 42,
    ) -> RobustFitResult:
        """
        Fit robust line to (x, y) using RANSAC + OLS on inliers.

        Recomputes residual threshold each call if not provided.

        Parameters
        ----------
        x : array-like
            Independent variable values.
        y : array-like
            Dependent variable values.
        residualThreshold : float, optional
            Residual threshold for inlier detection. If None, computed
            as 1.5 × MAD of y.
        randomState : int, optional
            Random seed for RANSAC.

        Returns
        -------
        result : `RobustFitResult`
            Best-fit parameters and statistics from the OLS inlier fit.

        Raises
        ------
        ValueError
            If no finite x/y values are available for fitting.
        """
        self._clearState()

        x = np.asarray(x)
        y = np.asarray(y)

        finiteMask = np.isfinite(x) & np.isfinite(y)
        if not finiteMask.any():
            raise ValueError("No finite x/y values to fit.")

        X = x[finiteMask].reshape(-1, 1)
        yFit = y[finiteMask]

        if residualThreshold is None:
            residualThreshold = self._defaultResidualThreshold(yFit)

        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=self.minSamples,
            residual_threshold=residualThreshold,
            random_state=randomState,
        )
        ransac.fit(X, yFit)

        slope = float(ransac.estimator_.coef_[0])
        intercept = float(ransac.estimator_.intercept_)

        self.x = x
        self.y = y

        outlierMask = np.ones_like(x, dtype=bool)
        outlierMask[finiteMask] = ~ransac.inlier_mask_
        self.outlierMask = outlierMask

        xIn = x[~outlierMask]
        yIn = y[~outlierMask]
        XDesign = sm.add_constant(xIn)
        ols = sm.OLS(yIn, XDesign).fit()
        ci = ols.conf_int(alpha=0.32)
        ciIntercept = (float(ci[0][0]), float(ci[0][1]))
        ciSlope = (float(ci[1][0]), float(ci[1][1]))
        scatter = float(np.nanstd(ols.resid, ddof=1)) if ols.resid.size > 1 else np.nan

        self.model = ransac
        self.olsModel = ols
        self.slope = slope
        self.intercept = intercept
        self.ciSlope = ciSlope
        self.ciIntercept = ciIntercept
        self.slopePValue = float(ols.pvalues[1]) if np.isfinite(ols.pvalues[1]) else np.nan
        self.slopeStdErr = float(ols.bse[1]) if np.isfinite(ols.bse[1]) else np.nan
        self.slopeTValue = float(ols.tvalues[1]) if np.isfinite(ols.tvalues[1]) else np.nan
        self.scatter = scatter

        return RobustFitResult(
            slope=slope,
            intercept=intercept,
            scatter=scatter,
            slopePValue=self.slopePValue,
            slopeStdErr=self.slopeStdErr,
            slopeTValue=self.slopeTValue,
            interceptPValue=float(ols.pvalues[0]) if np.isfinite(ols.pvalues[0]) else np.nan,
            interceptStdErr=float(ols.bse[0]) if np.isfinite(ols.bse[0]) else np.nan,
            interceptTValue=float(ols.tvalues[0]) if np.isfinite(ols.tvalues[0]) else np.nan,
            outlierMask=self.outlierMask,
        )

    def reset(self) -> None:
        """Clear any stored fit state."""
        self._clearState()

    def plotBestFit(self, ax: matplotlib.axes.Axes, label=None, color=None, alphaBand=0.2, lw=2, nBins=5):
        """Plot the best fit line, confidence interval,
        and optionally scatter/binned data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, uses current axes.
        label : str, optional
            Label for the best fit line.
        color : str or None, optional
            Color for the fit line and confidence band.
        alphaBand : float, optional
            Alpha transparency for the confidence interval band.
        lw : int, optional
            Line width for the best fit line.
        nBins : int, optional
            Number of bins for binned statistics.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        # Handle case where self.x contains only NaNs
        if self.x is None or self.y is None or self.outlierMask is None or self.olsModel is None:
            raise RuntimeError("Fit must be called before plotting.")

        if np.isnan(self.x).all():
            raise ValueError("All x values are NaN; cannot plot best fit.")

        xx = np.linspace(np.nanmin(self.x), np.nanmax(self.x), 200)
        X_design = sm.add_constant(xx)

        pred = self.olsModel.get_prediction(X_design)
        summary_frame = pred.summary_frame(alpha=0.05)  # 95% intervals

        mean = summary_frame["mean"]
        ci_lo = summary_frame["mean_ci_lower"]
        ci_hi = summary_frame["mean_ci_upper"]

        # Plot best fit
        ax.plot(xx, mean, color=color, label=label, lw=lw)
        # Plot confidence interval band
        ax.fill_between(xx, ci_lo, ci_hi, color=color, alpha=alphaBand)

        mask = self.outlierMask
        xin, yin = self.x[~mask], self.y[~mask]  # inliers
        xout, _ = self.x[mask], self.y[mask]  # outliers

        # Bin data in nbins
        if np.all(np.isnan(xout)):
            # Skip binning if all values are NaN
            pass
        else:
            bin_centers, means, stds, bin_counts = self.getBinnedData(xin, yin, nBins)

            # Plot binned means and stds
            ax.errorbar(
                bin_centers,
                means,
                color=color or "black",
                yerr=stds,
                fmt="o",
                capsize=4,
                markersize=6,
                alpha=0.7,
                zorder=10,
            )

        return ax

    @staticmethod
    def getBinnedData(
        x: np.ndarray, y: np.ndarray, nbins: int
    ) -> tuple[np.ndarray, list[float], list[float], list[int]]:
        """Get binned statistics for x and y."""
        bins = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        digitized = np.digitize(x, bins) - 1
        means = []
        stds = []
        bin_counts = []
        for i in range(nbins):
            bin_y = y[digitized == i]
            if len(bin_y) > 0:
                means.append(np.nanmedian(bin_y))
                if len(bin_y) > 1:
                    stds.append(np.nanstd(bin_y, ddof=1) / np.sqrt(len(bin_y)))
                else:
                    stds.append(np.nan)
                bin_counts.append(len(bin_y))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                bin_counts.append(0)
        return bin_centers, means, stds, bin_counts
