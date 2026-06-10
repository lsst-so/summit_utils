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

__all__ = [
    "calculateMountErrors",
    "plotMountErrors",
    "MountErrors",
    "getLinearRates",
    "getAltAzOverPeriod",
    "calculateHexRms",
]

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import astropy.units as u
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from matplotlib.dates import num2date
from matplotlib.ticker import FuncFormatter

from lsst.summit.utils.dateTime import dayObsIntToString
from lsst.summit.utils.tmaUtils import filterBadValues
from lsst.utils.plotting.figures import make_figure

from .mountData import getAzElRotHexDataForExposure

if TYPE_CHECKING:
    from astropy.time import Time
    from lsst_efd_client import EfdClient
    from matplotlib.figure import Figure

    from lsst.daf.butler import DimensionRecord

    from .mountData import MountData


NON_TRACKING_IMAGE_TYPES = ["BIAS", "FLAT"]

COMCAM_ANGLE_TO_EDGE_OF_FIELD_ARCSEC = 1800.0
LSSTCAM_ANGLE_TO_EDGE_OF_FIELD_ARCSEC = 8500.0

# These levels determine the colouring of the cells in the RubinTV.
# Yellow for warning level, red for bad level
MOUNT_IMAGE_WARNING_LEVEL = 0.05
MOUNT_IMAGE_BAD_LEVEL = 0.10  # and red for this

N_REPLACED_WARNING_LEVEL = 999999  # fill these values in once you've spoken to Craig and Brian
N_REPLACED_BAD_LEVEL = 999999  # fill these values in once you've spoken to Craig and Brian

SIMONYI_LOCATION = EarthLocation.of_site("Rubin:Simonyi")
EARTH_ROTATION = 15.04106858  # degrees/hour


@dataclass
class MountErrors:
    azRms: float
    elRms: float
    rotRms: float
    camHexRms: float
    m2HexRms: float
    imageAzRms: float
    imageElRms: float
    imageRotRms: float
    imageImpactRms: float
    residualFiltering: bool
    nReplacedAz: int
    nReplacedEl: int


def tickFormatter(value: float, tick_number: float) -> str:
    # Convert the value to a string without subtracting large numbers
    # tick_number is unused.
    return f"{value:.2f}"


def calculateMountErrors(
    expRecord: DimensionRecord,
    client: EfdClient,
    maxDelta: float = 0.1,
    doFilterResiduals: bool = False,
    useMockPointingModelResidualsAboveAzEl: float = 100.0,
    useMockPointingModelResidualsAboveRot: float = 15.0,
) -> tuple[MountErrors, MountData] | tuple[None, None]:
    """Queries the EFD over a given exposure and calculates the RMS errors
    for the axes, optionally using a pointing model to calculate residuals.

    Parameters
    ----------
    expRecord : `DimensionRecord`
        The exposure record containing the necessary fields for calculations.
    client : `EfdClient`
        The EFD client to query for mount data.
    maxDelta : `float`, optional
        The maximum delta for filtering bad values, by default 0.1.
    doFilterResiduals : `bool`, optional
        Whether to filter residuals.
    useMockPointingModelResidualsAboveAzEl : `float`, optional
        The threshold above which to use the mock pointing model residuals, as
        an RMS, in arcseconds, for the azimuth and elevation axes.
    useMockPointingModelResidualsAboveRot : `float`, optional
        The threshold above which to use the mock pointing model residuals, as
        an RMS, in arcseconds, for the rotator.

    Returns
    -------
    tuple[MountErrors, MountData] | tuple[None, None]
        A tuple containing the mount errors and mount data, or (None, None) if
        the exposure type is non-tracking.
    """
    logger = logging.getLogger(__name__)

    imgType = expRecord.observation_type.upper()
    if imgType in NON_TRACKING_IMAGE_TYPES:
        logger.info(f"Skipping mount torques for non-tracking image type {imgType} for {expRecord.id}")
        return None, None

    mountData = getAzElRotHexDataForExposure(client, expRecord)

    elevation = 90 - expRecord.zenith_angle

    azError = mountData.azimuthData["azError"].to_numpy()
    elError = mountData.elevationData["elError"].to_numpy()
    rotError = mountData.rotationData["rotError"].to_numpy()
    nReplacedAz = 0
    nReplacedEl = 0
    if doFilterResiduals:
        # Filtering out bad values
        nReplacedAz = filterBadValues(azError, maxDelta)
        nReplacedEl = filterBadValues(elError, maxDelta)
        mountData.azimuthData["azError"] = azError
        mountData.elevationData["elError"] = elError

    # Calculate the linear demand model
    if len(mountData.azimuthData) == len(mountData.elevationData):
        azModelValues, elModelValues = getAltAzOverPeriod(expRecord, nPoints=len(mountData.azimuthData))
    else:
        azModelValues, _ = getAltAzOverPeriod(expRecord, nPoints=len(mountData.azimuthData))
        _, elModelValues = getAltAzOverPeriod(expRecord, nPoints=len(mountData.elevationData))

    _, _, rotRate = getLinearRates(expRecord)

    azimuthData = mountData.azimuthData
    azValues = np.asarray(azimuthData["actualPosition"])
    azMedian = np.median(azValues)
    azModelMedian = np.median(azModelValues)
    # subtract the overall offset
    azModelValues -= azModelMedian - azMedian
    azimuthData["linearModel"] = azModelValues
    azLinearError = (azValues - azModelValues) * 3600
    azLinearRms = np.sqrt(np.mean(azLinearError * azLinearError))
    if azLinearRms > useMockPointingModelResidualsAboveAzEl:
        logger.warning(
            f"Azimuth pointing model RMS error {azLinearRms:.3f} arcsec is above threshold of "
            f"{useMockPointingModelResidualsAboveAzEl:.3f} arcsec, calculating errors vs astropy."
        )
        # If linear error is large, replace demand errors with linear error
        azimuthData["azError"] = azLinearError

    elevationData = mountData.elevationData
    elValues = np.asarray(elevationData["actualPosition"])
    elMedian = np.median(elValues)
    elModelMedian = np.median(elModelValues)
    # subtract the overall offset
    elModelValues -= elModelMedian - elMedian
    elevationData["linearModel"] = elModelValues
    elLinearError = (elValues - elModelValues) * 3600
    elLinearRms = np.sqrt(np.mean(elLinearError * elLinearError))
    if elLinearRms > useMockPointingModelResidualsAboveAzEl:
        logger.warning(
            f"Elevation pointing model RMS error {elLinearRms:.3f} arcsec is above threshold of "
            f"{useMockPointingModelResidualsAboveAzEl:.3f} arcsec, calculating errors vs astropy."
        )
        # If linear error is large, replace demand errors with linear error
        elevationData["elError"] = elLinearError

    rotationData = mountData.rotationData
    rotValues = np.asarray(rotationData["actualPosition"])
    rotValTimes = np.asarray(rotationData["timestamp"])
    rotModelValues = np.zeros_like(rotValues)
    rotMedian = np.median(rotValues)
    rotTimesMedian = np.median(rotValTimes)
    rotModelValues = rotMedian + rotRate * (rotValTimes - rotTimesMedian)
    rotationData["linearModel"] = rotModelValues
    rotLinearError = (rotValues - rotModelValues) * 3600
    rotLinearRms = np.sqrt(np.mean(rotLinearError * rotLinearError))
    if rotLinearRms > useMockPointingModelResidualsAboveRot:
        logger.warning(
            f"Rotation pointing model RMS error {rotLinearRms:.3f} arcsec is above threshold of "
            f"{useMockPointingModelResidualsAboveAzEl:.3f} arcsec, calculating errors vs astropy."
        )
        # If linear error is large, replace demand errors with linear error
        rotationData["rotError"] = rotLinearError

    azError = mountData.azimuthData["azError"].to_numpy()
    elError = mountData.elevationData["elError"].to_numpy()
    rotError = mountData.rotationData["rotError"].to_numpy()

    azRms = np.sqrt(np.mean(azError * azError))
    elRms = np.sqrt(np.mean(elError * elError))
    rotRms = np.sqrt(np.mean(rotError * rotError))

    # Calculate Image impact RMS
    imageAzRms = azRms * np.cos(elevation * np.pi / 180.0)
    imageElRms = elRms
    imageRotRms = rotRms * LSSTCAM_ANGLE_TO_EDGE_OF_FIELD_ARCSEC * np.pi / 180.0 / 3600.0
    camHexRms, m2HexRms = calculateHexRms(mountData)
    # TODO should the hex RMS values be added in quadrature?
    imageImpactRms = np.sqrt(imageAzRms**2 + imageElRms**2 + imageRotRms**2 + camHexRms**2 + m2HexRms**2)

    mountErrors = MountErrors(
        azRms=float(azRms),
        elRms=float(elRms),
        rotRms=float(rotRms),
        camHexRms=float(camHexRms),
        m2HexRms=float(m2HexRms),
        imageAzRms=float(imageAzRms),
        imageElRms=float(imageElRms),
        imageRotRms=float(imageRotRms),
        imageImpactRms=float(imageImpactRms),
        residualFiltering=doFilterResiduals,
        nReplacedAz=nReplacedAz,
        nReplacedEl=nReplacedEl,
    )

    return (mountErrors, mountData)


def plotMountErrors(
    mountData: MountData,
    mountErrors: MountErrors,
    figure=None,
    saveFilename: str = "",
) -> Figure:
    mountData = copy.deepcopy(mountData)  # Ensure we don't modify the original data
    mountErrors = copy.deepcopy(mountErrors)

    imageImpactRms = mountErrors.imageImpactRms
    expRecord = mountData.expRecord
    if expRecord is not None:
        dayObsString = dayObsIntToString(expRecord.day_obs)
        dataIdString = f"{expRecord.instrument} {dayObsString} - seqNum {expRecord.seq_num}"
        title = f"{dataIdString} - Exposure time = {expRecord.exposure_time:.1f}s"
    else:
        title = "Mount Errors"  # if the data is of unknown provenance

    if figure is None:
        figure = make_figure(figsize=(12, 8))
    else:
        figure.clear()
        ax = figure.gca()
        ax.clear()

    utc = ZoneInfo("UTC")
    chile_tz = ZoneInfo("America/Santiago")

    # Function to convert UTC to Chilean time
    def offset_time_aware(utc_time):
        # Ensure the time is timezone-aware in UTC
        if utc_time.tzinfo is None:
            utc_time = utc.localize(utc_time)
        return utc_time.astimezone(chile_tz)

    [[ax1, ax4], [ax2, ax5], [ax3, ax6]] = figure.subplots(
        3,
        2,
        sharex="col",
        sharey=False,
        gridspec_kw={
            "wspace": 0.35,
            "hspace": 0,
            "height_ratios": [2.5, 1, 1],
            "width_ratios": [1.4, 1],
            "left": 0.07,
            "right": 0.60,
        },
    )
    [ax7, ax8] = figure.subplots(
        2,
        1,
        sharex="col",
        sharey=False,
        gridspec_kw={
            "wspace": 0.35,
            "hspace": 0,
            "height_ratios": [1, 1],
            "width_ratios": [1],
            "left": 0.73,
            "right": 0.94,
        },
    )
    # [ax1, ax4] = [azimuth, rotator]
    # [ax2, ax5] = [azError, rotError]
    # [ax3, ax6] = [azTorque, rotTorque]
    # [ax7, ax8] = [camHex, m2Hex]

    # Use the native color cycle for the lines. Because they're on
    # different axes they don't cycle by themselves
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    lineColors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    nColors = len(lineColors)
    colorCounter = 0

    ax1.plot(
        mountData.azimuthData["actualPosition"],
        label="Azimuth position",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax1.plot(
        mountData.azimuthData["linearModel"],
        label="Azimuth linear model",
        ls="--",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax1.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax1.set_ylabel("Azimuth (degrees)")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        mountData.elevationData["actualPosition"],
        label="Elevation position",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax1_twin.plot(
        mountData.elevationData["linearModel"],
        label="Elevation linear model",
        ls="--",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1

    ax2.plot(
        mountData.azimuthData["azError"],
        label="Azimuth tracking error",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax2.plot(
        mountData.elevationData["elError"],
        label="Elevation tracking error",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax2.axhline(0.01, ls="-.", color="black")
    ax2.axhline(-0.01, ls="-.", color="black")
    ax2.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax2.set_ylabel("Tracking error (arcsec)")
    ax2.set_xticks([])  # remove x tick labels on the hidden upper x-axis
    ax2.set_ylim(-0.05, 0.05)
    ax2.set_yticks([-0.04, -0.02, 0.0, 0.02, 0.04])
    ax2.legend(loc="lower center")
    ax2.text(0.1, 0.9, f"Image impact = {imageImpactRms:.3f} arcsec (w/ rot&hex).", transform=ax2.transAxes)
    if mountErrors.residualFiltering:
        ax2.text(
            0.1,
            0.8,
            (
                f"{mountErrors.nReplacedAz} bad az values and "
                f"{mountErrors.nReplacedEl} bad el values were replaced"
            ),
            transform=ax2.transAxes,
        )
    ax3_twin = ax3.twinx()
    ax3.plot(
        mountData.azimuthData["actualTorque"],
        label="Azimuth torque",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax3_twin.plot(
        mountData.elevationData["actualTorque"],
        label="Elevation torque",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax3.set_ylabel("Azimuth torque (Nm)")
    ax3.set_xlabel("Time (UTC)")  # yes, it really is UTC, matplotlib converts this automatically!

    # put the ticks at an angle, and right align with the tick marks
    ax3.set_xticks(ax3.get_xticks())  # needed to supress a user warning
    xlabels = ax3.get_xticks()
    ax3.set_xticklabels(xlabels)
    ax3.tick_params(axis="x", rotation=45)
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    ax4.plot(
        mountData.rotationData["actualPosition"],
        label="Rotator position",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax4.plot(
        mountData.rotationData["linearModel"],
        label="Rotator linearModel",
        ls="--",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax4.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax4.yaxis.tick_right()
    ax4.set_ylabel("Rotator angle (degrees)")
    ax4.yaxis.set_label_position("right")
    ax5.plot(
        mountData.rotationData["rotError"],
        c=lineColors[colorCounter % nColors],
    )

    colorCounter += 1
    ax5.axhline(0.1, ls="-.", color="black")
    ax5.axhline(-0.1, ls="-.", color="black")
    ax5.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax5.set_ylabel("Tracking error (arcsec)")
    ax5.tick_params(labelbottom=False)  # Hide x-axis tick labels without removing ticks
    ax5.set_ylim(-0.5, 0.5)
    ax5.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    ax5.yaxis.tick_right()
    ax5.yaxis.set_label_position("right")

    ax6.plot(mountData.rotationTorques["torque0"], label="Torque0", c=lineColors[colorCounter % nColors])
    colorCounter += 1
    ax6.plot(mountData.rotationTorques["torque1"], label="Torque1", c=lineColors[colorCounter % nColors])
    ax6.set_xlabel("Time (UTC)")  # yes, it really is UTC, matplotlib converts this automatically!
    # put the ticks at an angle, and right align with the tick marks
    ax6.set_xticks(ax6.get_xticks())  # needed to supress a user warning
    xlabels = ax6.get_xticks()
    ax6.set_xticklabels(xlabels)
    ax6.tick_params(axis="x", rotation=45)
    ax6.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax6.yaxis.tick_right()
    ax6.yaxis.set_label_position("right")
    ax6.legend()

    hexNames = ["X", "Y", "Z", "U", "V", "W"]
    for i in [0, 1, 2]:
        camhex = mountData.camhexData[f"position{i}"]
        camhex -= np.median(camhex)
        ax7.plot(
            camhex,
            label=hexNames[i],
            c=lineColors[colorCounter % nColors],
        )
        colorCounter += 1
    ax7.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax7.set_ylabel("CamHex XYZ(micron) (minus median)")
    ax7.legend()
    ax7_twin = ax7.twinx()
    for i in [3, 4]:
        camhex = mountData.camhexData[f"position{i}"]
        camhex *= 3600.0  # convert to arcseconds
        camhex -= np.median(camhex)
        ax7_twin.plot(
            camhex,
            label=hexNames[i],
            c=lineColors[colorCounter % nColors],
        )
        colorCounter += 1
    ax7_twin.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax7_twin.set_ylabel("CamHex UV(arcsec) (minus median)")
    # ax7_twin.yaxis.tick_right()
    # ax7.yaxis.set_label_position("right")
    ax7_twin.legend()

    for i in [0, 1, 2]:
        m2hex = mountData.m2hexData[f"position{i}"]
        m2hex -= np.median(m2hex)
        ax8.plot(
            m2hex,
            label=hexNames[i],
            c=lineColors[colorCounter % nColors],
        )
        colorCounter += 1
    ax8.legend()
    ax8.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax8.set_ylabel("M2Hex XYZ(micron) (minus median)")
    ax8.set_xlabel("Time (UTC)")  # yes, it really is UTC, matplotlib converts this automatically!
    # put the ticks at an angle, and right align with the tick marks
    ax8.set_xticks(ax8.get_xticks())  # needed to supress a user warning
    xlabels = ax8.get_xticks()
    ax8.set_xticklabels(xlabels)
    ax8.tick_params(axis="x", rotation=45)
    ax8.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax8.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax8_twin = ax8.twinx()
    for i in [3, 4]:
        m2hex = mountData.m2hexData[f"position{i}"]
        m2hex *= 3600.0  # Convert to arcseconds
        m2hex -= np.median(m2hex)
        ax8_twin.plot(
            m2hex,
            label=hexNames[i],
            c=lineColors[colorCounter % nColors],
        )
        colorCounter += 1
    ax8_twin.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax8_twin.set_ylabel("M2Hex UV(arcsec) (minus median)")
    # ax7_twin.yaxis.tick_right()
    # ax7.yaxis.set_label_position("right")
    ax8_twin.legend()

    ax1_twin.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax1_twin.set_ylabel("Elevation (degrees)")
    ax1.tick_params(labelbottom=False)  # Hide x-axis tick labels without removing ticks
    # combine the legends and put inside the plot
    handles1a, labels1a = ax1.get_legend_handles_labels()
    handles1b, labels1b = ax1_twin.get_legend_handles_labels()
    handles2a, labels2a = ax3.get_legend_handles_labels()
    handles2b, labels2b = ax3_twin.get_legend_handles_labels()
    handles = handles1a + handles1b + handles2a + handles2b
    labels = labels1a + labels1b + labels2a + labels2b
    # ax2 is "in front" of ax1 because it has the vlines plotted on it, and
    # vlines are on ax2 so that they appear at the bottom of the legend, so
    # make sure to plot the legend on ax2, otherwise the vlines will go on
    # top of the otherwise-opaque legend.
    ax1_twin.legend(handles, labels, facecolor="white", framealpha=1)

    ax1.set_title("Azimuth and Elevation")
    ax4.set_title("Rotator")
    ax7.set_title("Hexapods")
    ax4.legend()
    figure.subplots_adjust(top=0.85)  # Adjust the top margin to make room for the suptitle
    figure.suptitle(title, fontsize=14, y=1.04)  # Adjust y to move the title up

    # Create the upper axis for Chilean time
    ax1_twiny = ax1.twiny()
    ax1_twiny.set_xlim(ax1.get_xlim())  # Set the limits of the upper axis to match the lower axis
    utcTicks = ax1.get_xticks()  # Use the same ticks as the lower UTC axis
    utcTickLabels = [num2date(tick, tz=utc) for tick in utcTicks]
    chileTickLabels = [offset_time_aware(label) for label in utcTickLabels]
    # Set the same tick positions but with Chilean time labels
    ax1_twiny.set_xticks(utcTicks)
    ax1_twiny.set_xticklabels([tick.strftime("%H:%M:%S") for tick in chileTickLabels])
    ax1_twiny.tick_params(axis="x", rotation=45)
    ax1_twiny.set_xlabel("Time (Chilean Time)")

    ax4_twiny = ax4.twiny()
    ax4_twiny.set_xlim(ax4.get_xlim())  # Set the limits of the upper axis to match the lower axis
    utcTicks = ax4.get_xticks()  # Use the same ticks as the lower UTC axis
    utcTickLabels = [num2date(tick, tz=utc) for tick in utcTicks]
    chileTickLabels = [offset_time_aware(label) for label in utcTickLabels]
    # Set the same tick positions but with Chilean time labels
    ax4_twiny.set_xticks(utcTicks)
    ax4_twiny.set_xticklabels([tick.strftime("%H:%M:%S") for tick in chileTickLabels])
    ax4_twiny.tick_params(axis="x", rotation=45)
    ax4_twiny.set_xlabel("Time (Chilean Time)")

    ax7_twiny = ax7.twiny()
    ax7_twiny.set_xlim(ax7.get_xlim())  # Set the limits of the upper axis to match the lower axis
    utcTicks = ax7.get_xticks()  # Use the same ticks as the lower UTC axis
    utcTickLabels = [num2date(tick, tz=utc) for tick in utcTicks]
    chileTickLabels = [offset_time_aware(label) for label in utcTickLabels]
    # Set the same tick positions but with Chilean time labels
    ax7_twiny.set_xticks(utcTicks)
    ax7_twiny.set_xticklabels([tick.strftime("%H:%M:%S") for tick in chileTickLabels])
    ax7_twiny.tick_params(axis="x", rotation=45)
    ax7_twiny.set_xlabel("Time (Chilean Time)")

    # Add exposure start and end:
    for ax in axs:
        if expRecord is not None:
            # assert expRecord is not None, "expRecord is None"
            ax.axvline(expRecord.timespan.begin.utc.datetime, ls="--", color="green")
            ax.axvline(expRecord.timespan.end.utc.datetime, ls="--", color="red")

    if saveFilename:
        figure.savefig(saveFilename, bbox_inches="tight")

    return figure


def getLinearRates(expRecord: DimensionRecord) -> tuple[float, float, float]:
    """Calculate the linear rates of motion for az, el, and rotation during an
    exposure.

    The rates are calculated based on the tracking RA and Dec, azimuth, zenith
    angle, and the exposure timespan. The rates are returned in degrees per
    second.

    Parameters
    ----------
    expRecord : `DimensionRecord`
        The exposure record containing the necessary fields for calculations.

    Returns
    -------
    azRate, elRate, rotRate: `tuple`[`float`, `float`, `float`]
        The azimuth rate, elevation rate, and rotator rate in degrees per
        second.
    """
    begin: Time = expRecord.timespan.begin
    end: Time = expRecord.timespan.end
    dT: float = (expRecord.timespan.end - expRecord.timespan.begin).value * 86400.0
    rotRate = (
        -EARTH_ROTATION
        * np.cos(SIMONYI_LOCATION.lat.rad)
        * np.cos(expRecord.azimuth * u.deg)
        / np.cos((90.0 - expRecord.zenith_angle) * u.deg)
        / 3600.0
    )
    skyLocation = SkyCoord(expRecord.tracking_ra * u.deg, expRecord.tracking_dec * u.deg)
    altAz1 = AltAz(obstime=begin, location=SIMONYI_LOCATION)
    altAz2 = AltAz(obstime=end, location=SIMONYI_LOCATION)
    obsAltAz1 = skyLocation.transform_to(altAz1)
    obsAltAz2 = skyLocation.transform_to(altAz2)
    elRate = float((obsAltAz2.alt.deg - obsAltAz1.alt.deg) / dT)
    azRate = float((obsAltAz2.az.deg - obsAltAz1.az.deg) / dT)

    # All rates are in degrees / second
    return azRate, elRate, float(rotRate.value)


def getAltAzOverPeriod(
    expRecord: DimensionRecord,
    nPoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the AltAz coordinates over a period.

    Parameters
    ----------
    begin : `Time`
        The beginning of the period.
    end : `Time`
        The end of the period.
    target : `SkyCoord`
        The sky coordinates to track.
    nPoints : `int`, optional
        The number of points to sample, by default 100.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The azimuth and elevation coordinates in degrees.
    """
    begin = expRecord.timespan.begin
    end = expRecord.timespan.end
    times = begin + (end - begin) * np.linspace(0, 1, nPoints)
    target = SkyCoord(expRecord.tracking_ra * u.deg, expRecord.tracking_dec * u.deg)
    altAzFrame = AltAz(obstime=times, location=SIMONYI_LOCATION)
    targetAltAz = target.transform_to(altAzFrame)
    az = targetAltAz.az
    if abs(az[0].degree) < 90.0:
        az_wrapped = az.wrap_at(180.0 * u.deg)
    else:
        az_wrapped = az.wrap_at(0.0 * u.deg)
    return az_wrapped.degree, targetAltAz.alt.degree


def calculateHexRms(mountData: MountData) -> tuple[float, float]:
    """Calculate the image impact of hexapod motions.

    Parameters
    ----------
    mountData : MountData
        The EFD data associated with the exposure

    Returns
    -------
    tuple[float, float]
        The image motions associated with the CamHex and M2Hex motions.
    """

    # The below image motion coefficients were calculated
    # with a Batoid simulation by Josh Meyers
    camHexXY = 1.00  # microns(image) / micron(hexapod)
    camHexUV = 4.92  # microns(image) / arcsecond(hexapod)
    m2HexXY = 1.13  # microns(image) / micron(hexapod)
    m2HexUV = 37.26  # microns(image) / arcsecond(hexapod)

    # Convert these to image impact in arcseconds
    # The 10.0 is microns / pixel
    pixelScale = 0.2  # arcseconds / pixel - find this elsewhere?
    camHexXY = camHexXY / 10.0 * pixelScale  # arcseconds(image) / micron(hexapod)
    camHexUV = camHexUV / 10.0 * pixelScale  # arcseconds(image) / arcsecond(hexapod)
    camCoefs = [camHexXY, camHexXY, 0, camHexUV, camHexUV, 0]
    m2HexXY = m2HexXY / 10.0 * pixelScale  # arcseconds(image) / micron(hexapod)
    m2HexUV = m2HexUV / 10.0 * pixelScale  # arcseconds(image) / arcsecond(hexapod)
    m2Coefs = [m2HexXY, m2HexXY, 0, m2HexUV, m2HexUV, 0]

    camHexMs = 0.0
    for i in [0, 1, 3, 4]:
        camhex = copy.deepcopy(mountData.camhexData[f"error{i}"])
        camhex *= camCoefs[i]
        camHexMs += np.mean(camhex * camhex)
    camHexRms = np.sqrt(camHexMs)  # in arcseconds image impact

    m2HexMs = 0.0
    for i in [0, 1, 3, 4]:
        m2hex = copy.deepcopy(mountData.m2hexData[f"error{i}"])
        m2hex *= m2Coefs[i]
        m2HexMs += np.mean(m2hex * m2hex)
    m2HexRms = np.sqrt(m2HexMs)  # in arcseconds image impact
    return (float(camHexRms), float(m2HexRms))
