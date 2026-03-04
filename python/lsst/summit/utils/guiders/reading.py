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
from typing import TYPE_CHECKING, Any

__all__ = [
    "GuiderReader",
    "convertRawStampsToView",
    "GuiderData",
]

from dataclasses import dataclass
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, computed_field

from lsst.afw import cameraGeom
from lsst.afw.image import ExposureF, ImageF, MaskedImageF, MaskX, VisitInfo
from lsst.daf.butler import Butler, DatasetNotFoundError
from lsst.meas.algorithms.stamps import Stamp, Stamps
from lsst.obs.lsst import LsstCam
from lsst.utils.plotting.figures import make_figure

from .plotting import GuiderPlotter, labelDetector
from .transformation import (
    getCamRotAngle,
    makeCcdToDvcsTransform,
    makeInitGuiderWcs,
    makeRoiBbox,
    roiImageToDvcs,
)

if TYPE_CHECKING:
    from lsst.afw.cameraGeom import Camera, Detector
    from lsst.geom import SkyWcs


def make_subplot(nrows=1, ncols=1, **fig_kwargs):
    """Return (fig, axs) using LSST's make_figure."""
    fig = make_figure(**fig_kwargs)
    axs = fig.subplots(nrows=nrows, ncols=ncols, squeeze=True)
    return fig, axs


class GuiderData(BaseModel):
    """
    LSST guider data container.

    Holds raw guider Stamps, WCS objects, and exposure metadata. Provides
    cached properties for timestamps, header information, and convenience
    (dict-like) access.

    Parameters
    ----------
    seqNum : `int`
        Sequence number.
    dayObs : `int`
        Day of observation (YYYYMMDD).
    guiderNameMap : `dict[str, int]`
        Detector name to detector number map.
    rawStampsMap : `dict[str, Stamps]`
        Detector name to raw Stamps map.
    wcsMap : `dict[str, SkyWcs]`
        Detector name to SkyWcs map.
    camRotAngle : `float`
        Camera rotation angle in degrees.
    isMedianSubtracted : `bool`
        If True, the raw stamps have been median-subtracted.
    columnMaskK : `float`
        Threshold factor used for column mask detection.
    view : `str`
        Output view: 'dvcs', 'ccd', or 'roi'.

    Cached properties
    -----------------
    guiderNames        : sorted detector names.
    detNameMax         : detector with most stamps.
    metadata           : metadata dict for detNameMax.
    header             : exposure-level header summary.
    expid              : DAYOBS * 100000 + SEQNUM.
    timestamps         : masked astropy Time array (TAI scale).
    stampsMap          : Stamps for all detectors in requested view.
    roiAmpNames        : detector → amplifier name map.
    guiderFrequency    : stamp cadence in Hz.

    Public methods
    --------------
    printHeaderInfo()       : print header summary.
    getWcs(det)             : return SkyWcs for detector.
    getStampArrayCoadd(det) : median stack of stamps.
    getGuiderAmpName(det)   : ROI amplifier name.
    getGuiderDetNum(det)    : detector number.
    plotMosaic(...)         : plot all detectors as a mosaic.
    plotStamp(...)          : plot one stamp from a single detector.
    makeGif(...)            : create a GIF animation of guider frames.

    Iteration and indexing
    ----------------------
    for det in guiderData        # iterates detector names
    guiderData.items()           # yields (det, Stamps)
    guiderData['R44_SG0']        # full Stamps object
    guiderData['R44_SG0', 3]     # stamp ndarray
    guiderData[3]                # stamp from detNameMax
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    seqNum: int
    dayObs: int
    guiderNameMap: dict[str, int]
    rawStampsMap: dict[str, Stamps]
    wcsMap: dict[str, Any]
    camRotAngle: float
    isMedianSubtracted: bool = False
    columnMaskK: float = 50.0
    view: str = "dvcs"

    def __repr__(self) -> str:
        """
        Return a compact one-line representation.

        Returns
        -------
        reprStr : `str`
            String summary with exposure id, number of stamps, view, and guider
            names.
        """
        return (
            f"GuiderData(seqNum={self.seqNum}, "
            f"dayObs={self.dayObs}, "
            f"nStamps={len(self)}, view='{self.view}', "
            f"guiders={self.guiderNames})"
        )

    def __len__(self) -> int:
        """
        Return number of stamps for the detector with the maximum count.

        Returns
        -------
        nStamps : `int`
            Number of stamps (time samples) in the exposure for the primary
            detector.
        """
        return len(self.rawStampsMap[self.detNameMax])

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def expid(self) -> int:
        """
        Exposure identifier.

        Returns
        -------
        expid : `int`
            The exposure id.
        """
        return self.dayObs * 100000 + self.seqNum

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def guiderNames(self) -> list[str]:
        """
        Names of the guider detectors.

        Returns
        -------
        guiderNames : `list[str]`
            Sorted list of guider detector names.
        """
        return sorted(list(self.guiderNameMap.keys()))

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def guiderIds(self) -> list[int]:
        """
        IDs of the guider detectors.

        Returns
        -------
        guiderIds : `list[int]`
            Sorted list of guider detector numeric IDs.
        """
        return sorted(list(self.guiderNameMap.values()))

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def detNameMax(self) -> str:
        """
        Detector name with the maximum number of stamps.

        Returns
        -------
        detName : `str`
            Detector name possessing the most stamps.
        """
        return max(self.rawStampsMap, key=lambda k: len(self.rawStampsMap[k]))

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def metadata(self) -> dict:
        """
        Metadata for the detector with the most stamps.

        Returns
        -------
        metadata : `dict`
            Copy of the detector-level metadata dictionary.
        """
        return self.rawStampsMap[self.detNameMax].metadata.toDict()

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def header(self) -> dict[str, str | float | None]:
        """
        Dictionary of header metadata for this GuiderData.
        Fields: n_stamps, freq, expid, filter, cam_rot_angle, start_time,
        roi_cols, roi_rows, shuttime, az_start, el_start, az_end, el_end,
        seeing.

        Returns
        -------
        header : `dict[str, str | float | None]`
            The headers.
        """
        md = self.metadata
        info: dict[str, str | float | None] = {}
        info["n_stamps"] = len(self)
        info["freq"] = self.guiderFrequency
        # Ensure DAYOBS and SEQNUM are ints before math, keep ≤79 chars
        info["expid"] = self.expid
        info["filter"] = md.get("FILTBAND", "Unknown")
        info["cam_rot_angle"] = self.camRotAngle
        info["start_time"] = md.get("GDSSTART", None)
        info["roi_cols"] = int(md.get("ROICOLS", 0))
        info["roi_rows"] = int(md.get("ROIROWS", 0))
        info["shuttime"] = metadata_to_float(md, "SHUTTIME", np.nan)
        info["guider_duration"] = self.guiderDurationSec
        info["az_start"] = metadata_to_float(md, "AZSTART", np.nan)
        info["el_start"] = metadata_to_float(md, "ELSTART", np.nan)
        info["az_end"] = metadata_to_float(md, "AZEND", np.nan)
        info["el_end"] = metadata_to_float(md, "ELEND", np.nan)
        return info

    def printHeaderInfo(self) -> None:
        """Print a concise summary of key header fields."""
        print(f"Data Id: {self.expid}, filter-band: {self.header['filter']}")
        print(f"ROI Shape (row, col): {self.header['roi_rows']}, {self.header['roi_cols']}")
        print(f"With nStamps {len(self)} at {self.guiderFrequency} Hz")
        print(
            f"Acq. Start Time: {self.header['start_time']} \n"
            f"with readout duration: {self.header['guider_duration']:.2f} sec"
        )

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def weather(self) -> WeatherInfo:
        """
        Weather information from metadata.

        Returns
        -------
        weather : `WeatherInfo`
            Parsed weather information.
        """
        return WeatherInfo.fromMetadata(self.metadata)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def timestampMap(self) -> dict[str, Time]:
        """
        Aligned timestamp arrays for all detectors.

        Returns
        -------
        timestampMap : `dict[str, Time]`
            Mapping from detector name to masked Time array (TAI scale).
        """
        nStamps = len(self.rawStampsMap[self.detNameMax])
        timestampMap: dict[str, Time] = {}
        for detName, stamps in self.rawStampsMap.items():
            timestampMap[detName] = standardizeGuiderTimestamps(stamps, nStamps)
        return timestampMap

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def guiderFrequency(self) -> float:
        """
        Guider stamp cadence.

        Returns
        -------
        frequency : `float`
            Median stamp frequency in Hz.
        """
        timestamps = self.timestampMap[self.detNameMax]
        # extract only the unmasked MJD values
        jd = timestamps[~timestamps.mask].jd
        period_sec = np.median(np.diff(jd)) * 86400.0
        return float(1.0 / period_sec)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def guiderDurationSec(self) -> float:
        """
        Total guider duration.

        Returns
        -------
        durationSec : `float`
            Total elapsed guider duration in seconds (including final period).
        """
        timestamps = self.timestampMap[self.detNameMax]
        jd = timestamps[~timestamps.mask].jd
        duration_sec = (jd[-1] - jd[0]) * 86400.0 + 1 / self.guiderFrequency
        return float(duration_sec)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def stampsMap(self) -> dict[str, Stamps]:
        """
        Stamps converted to requested view.

        Returns
        -------
        stampsMap : `dict[str, Stamps]`
            Mapping of detector name to converted Stamps object.
        """
        result: dict[str, Stamps] = {}
        if self.view == "roi":
            return self.rawStampsMap
        else:
            for detName, rawStamps in self.rawStampsMap.items():
                result[detName] = convertRawStampsToView(
                    rawStamps,
                    detName,
                    len(self),
                    view=self.view,
                )
            return result

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def roiAmpNames(self) -> dict[str, str]:
        """
        Amplifier names used in the ROI.

        Returns
        -------
        ampNames : `dict[str, str]`
            Mapping from detector name to amplifier name active in ROI.
        """
        ampNames: dict[str, str] = {}
        for detName in self.rawStampsMap.keys():
            md = self.rawStampsMap[detName].metadata.toDict()
            segment = md["ROISEG"]
            ampName = "C" + segment[7:]
            ampNames[detName] = ampName
        return ampNames

    def getStampArrayCoadd(self, detName: str) -> np.ndarray:
        """
        Get the median-stacked stamp across time for the specified detector.

        The stack is computed across the time axis after optional median-row
        bias removal (controlled by ``self.isMedianSubtracted``).

        Parameters
        ----------
        detName : `str`
            Guider detector name.

        Returns
        -------
        coadd : `np.ndarray`
            Median of all stamp arrays (time axis collapsed).
        """
        if detName not in self.stampsMap:
            raise KeyError(f"{detName!r} not present in stampsMap")

        stamps = self.stampsMap[detName]
        if len(stamps) == 0:
            raise ValueError(f"No stamps found for detector {detName!r}")

        # Collect arrays, with optional bias subtraction.
        arrList = [self[detName, idx] for idx in range(len(stamps))]
        # Add a uniform [0, 1) dither per stamp before the median to break
        # integer quantization. Without this, the coadd pixel distribution
        # can be so narrow that STDEVCLIP returns 0. (See DM-54263.)
        stack = np.array(arrList, dtype=np.float32)
        rng = np.random.default_rng(seed=0)
        stack += rng.uniform(0, 1, size=stack.shape).astype(np.float32)
        return np.nanmedian(stack, axis=0)

    def getGuiderAmpName(self, detName: str) -> str:
        """
        Return amplifier name for a guider detector.

        Parameters
        ----------
        detName : `str`
            Guider detector name.

        Returns
        -------
        ampName : `str`
            Amplifier name string.
        """
        if detName not in self.roiAmpNames:
            raise ValueError(f"Detector {detName} not found in roiAmpNames.")
        return self.roiAmpNames[detName]

    def getGuiderDetNum(self, detName: str) -> int:
        """
        Return detector number for a guider detector name.

        Parameters
        ----------
        detName : `str`
            Guider detector name.

        Returns
        -------
        detNum : `int`
            The numeric detector id.
        """
        if detName not in self.guiderNameMap:
            raise ValueError(f"Detector {detName} not found in guiderNameMap.")
        return self.guiderNameMap[detName]

    def getWcs(self, detName: str) -> SkyWcs:
        """
        Return the wcs for a guider detector.

        Parameters
        ----------
        detName : `str`
            Guider detector name.

        Returns
        -------
        wcs : `SkyWcs`
            SkyWcs for the detector.
        """
        try:
            return self.wcsMap[detName]
        except KeyError as e:
            raise KeyError(f"{detName!r} not found in wcsMap") from e

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def obsTime(self) -> Time:
        """
        Observation start time in TAI.

        Returns
        -------
        obsTime : `Time`
            Start time parsed from metadata (TAI scale).
        """
        gdstart = self.header["start_time"]
        return Time(gdstart, format="isot", scale="tai")

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def missingStampsMap(self) -> dict[str, int]:
        """
        Number of missing stamps in the primary detector.

        Returns
        -------
        nMissing : dict[str, int]
            Count of missing stamps (NaN timestamps) in detNameMax.
        """
        nMissing: dict[str, int] = {}
        for detName, timestamps in self.timestampMap.items():
            nMissing[detName] = np.sum(timestamps.mask)
        return nMissing

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def nMissingStamps(self) -> int:
        """
        Total number of missing stamps across all detectors.

        Returns
        -------
        nMissing : int
            Total count of missing stamps across all detectors.
        """
        return sum(self.missingStampsMap.values())

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def alt(self) -> float:
        """Return altitude (el_start) from the guider data header."""
        raw = self.header.get("el_start")
        if raw is None:
            raise KeyError("Header key 'el_start' is missing or None")
        return float(raw)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def az(self) -> float:
        """Return azimuth (az_start) from the guider data header."""
        raw = self.header.get("az_start")
        if raw is None:
            raise KeyError("Header key 'az_start' is missing or None")
        return float(raw)

    # Iterable / dict-like helpers
    def __iter__(self):
        """Iterate over detector names in guiderNames order."""
        return iter(self.guiderNames)

    def items(self):
        """Yield (detName, stamps) pairs like dict.items()."""
        for det in self.guiderNames:
            yield det, self.stampsMap[det]

    def keys(self):
        """Iterate over detector names (dict-like .keys())."""
        return iter(self.guiderNames)

    def values(self):
        """Iterate over Stamps objects in guiderNames order."""
        for det in self.guiderNames:
            yield self.stampsMap[det]

    def __getitem__(self, key):
        """
        Direct stamp access helper.

        Example usage:
        gd["R44_SG0"]            -> full `Stamps` object
        gd["R44_SG0", 3]         -> stamp #3 ndarray
        gd[3]                    -> stamp #3 from `detNameMax`

        Parameters
        ----------
        key : `str | int | slice | tuple`
            Access pattern:
            - 'DET' -> Stamps
            - ('DET', i) -> single stamp ndarray
            - i (int) -> stamp i from primary detector
            - ('DET', slice) -> list of ndarrays

        Returns
        -------
        result : `Stamps | np.ndarray | list[np.ndarray]`
            Retrieved object according to key specification.
        """
        # Single detector name -> full Stamps
        if isinstance(key, str):
            return self.stampsMap[key]

        # idx only -> assume detNameMax
        if isinstance(key, (int, slice)):
            key = (self.detNameMax, key)

        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError("Key must be (detName, idx) or (idx,)")
            detName, idx = key[0], key[1]

            if detName not in self.stampsMap:
                raise KeyError(f"{detName!r} not found in stampsMap")

            stamps = self.stampsMap[detName]
            # slice returns list of ndarrays
            if isinstance(idx, slice):
                arrays = [self._processStampArray(stamp, detName) for stamp in stamps[idx]]
                return arrays
            # int -> single ndarray
            return self._processStampArray(stamps[idx], detName)

        raise TypeError("Invalid key type for GuiderData indexing.")

    def _processStampArray(self, stamp: Stamp, detName: str) -> np.ndarray:
        """
        Convert a Stamp to an ndarray with optional median-row subtraction.

        Parameters
        ----------
        stamp : `Stamp`
            Input stamp object.
        detName : `str`
            Detector name (for row-axis logic).

        Returns
        -------
        array : `np.ndarray`
            2D image array (bias-corrected if configured).
        """
        arr = stamp.stamp_im.image.array
        return arr

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def plotter(self) -> GuiderPlotter:
        return GuiderPlotter(self)

    def plotStamp(
        self, detName: str, stampNum: int, plo: float = 90, phi: float = 99.5, figsize=(10, 8)
    ) -> plt.Figure:
        """
        Plot a single guider stamp.

        Parameters
        ----------
        detName : `str`
            Detector name.
        stampNum : `int`
            Stamp index.
        plo : `float`, optional
            Lower percentile stretch.
        phi : `float`, optional
            Upper percentile stretch.
        figsize : `tuple`, optional
            Figure size.

        Returns
        -------
        fig : `plt.Figure`
            The resulting figure.
        """
        fig, axs = make_subplot(nrows=1, ncols=1, figsize=figsize)
        img = self[detName, stampNum] if stampNum > 0 else self.getStampArrayCoadd(detName)
        _ = axs.imshow(
            img,
            origin="lower",
            vmin=np.nanpercentile(img, plo),
            vmax=np.nanpercentile(img, phi),
            cmap="Greys",
        )
        _ = labelDetector(axs, detName)
        axs.set_xlabel("X (pixels)", fontsize=11)
        axs.set_ylabel("Y (pixels)", fontsize=11)
        axs.set_title(f"{self.expid}")
        return fig


class GuiderReader:
    """
    Utility to fetch LSST guider data via Butler.

    Example:
        from lsst.summit.utils.guiders.reading import GuiderReader
        from lsst.daf.butler import Butler
        butler = Butler("embargo", collections="LSSTCam/raw/guider")

        seqNum, dayObs = 461, 20250425
        reader = GuiderReader(butler, view="dvcs")
        guiderData = reader.get(dayObs=dayObs, seqNum=seqNum)
    """

    def __init__(self, butler: Butler, view: str = "dvcs"):
        self.butler = butler
        self.view = view
        self.log = logging.getLogger(__name__)
        # Define camera objects
        self.camera = LsstCam.getCamera()

        # Build guiderNameMap
        self.guiderNameMap: dict[str, int] = {}
        for detector in self.camera:
            if detector.getType() == cameraGeom.DetectorType.GUIDER:
                detName = detector.getName()
                self.guiderNameMap[detName] = detector.getId()

        self.guiderDetNames = list(self.guiderNameMap.keys())
        self.nGuiders = len(self.guiderNameMap)

    def get(
        self,
        dayObs: int,
        seqNum: int,
        doSubtractMedian: bool = True,
        columnMaskK: float = 50.0,
        biasPercentile: float = 10.0,
        scienceDetNum: int = 94,
    ) -> GuiderData:
        """
        Retrieve guider data for a given dayObs / seqNum.

        Parameters
        ----------
        dayObs : `int`
            Day of observation in YYYYMMDD format.
        seqNum : `int`
            Sequence number.
        doSubtractMedian : `bool`, optional
            If True, subtract column bias from each stamp.
        columnMaskK : `float`, optional
            Threshold factor for column mask detection.
        biasPercentile : `float`, optional
            Percentile (0-100) for column bias estimation.
            Lower values avoid star/trail contamination.
        scienceDetNum : `int`, optional
            Science detector number for WCS reference.

        Returns
        -------
        guiderData : `GuiderData`
            Assembled guider data object.
        """
        # Check if the guider name is swapped (dayObs < 20250509)
        # modifies self.guiderNameMap in place if necessary
        self.applyGuiderNameSwapIfNeeded(dayObs)

        rawStampsDict = self.getGuiderRawStamps(dayObs, seqNum)

        # Determine the maximum number of stamps among all guiders
        nStampsList = [len(stamps) for stamps in rawStampsDict.values()]
        nStamps = max(nStampsList)

        if nStamps <= 1:
            raise RuntimeError(
                f"Only {nStamps} stamps found for dayObs {dayObs}, seqNum {seqNum}. "
                "At least 2 stamps are required to create GuiderData."
            )

        visitInfo = getVisitInfo(self.butler, dayObs, seqNum, scienceDetNum)
        wcsMapDict = makeInitGuiderWcs(self.camera, visitInfo)
        camRotAngle = getCamRotAngle(visitInfo)

        processedStampsDict = self.processStamps(
            rawStampsDict,
            doSubtractMedian,
            columnMaskK,
            biasPercentile,
        )
        guiderData = GuiderData(
            seqNum=seqNum,
            dayObs=dayObs,
            view=self.view,
            rawStampsMap=processedStampsDict,
            guiderNameMap=self.guiderNameMap,
            wcsMap=wcsMapDict,
            camRotAngle=camRotAngle,
            isMedianSubtracted=doSubtractMedian,
            columnMaskK=columnMaskK,
        )
        return guiderData

    def getGuiderRawStamps(self, dayObs: int, seqNum: int) -> dict[str, Stamps]:
        """
        Fetch raw guider Stamps for all detectors.

        Parameters
        ----------
        dayObs : `int`
            Observation day (YYYYMMDD).
        seqNum : `int`
            Sequence number.

        Returns
        -------
        rawStamps : `dict[str, Stamps]`
            Mapping from detector name to raw Stamps object.
        """
        rawStamps: dict[str, Stamps] = {}
        for detName, detNum in self.guiderNameMap.items():
            try:
                rawStamps[detName] = self.butler.get(
                    "guider_raw",
                    day_obs=dayObs,
                    seq_num=seqNum,
                    detector=detNum,
                    instrument="LSSTCam",
                )
            except DatasetNotFoundError as e:
                raise DatasetNotFoundError(f"No data for {detName} on {dayObs=} {seqNum=}") from e
        return rawStamps

    def processStamps(
        self,
        rawStampsDict: dict[str, Stamps],
        doSubtractMedian: bool,
        columnMaskK: float,
        biasPercentile: float = 10.0,
    ) -> dict[str, Stamps]:
        """
        Apply column bias subtraction and per-stamp column masking.


        Parameters
        ----------
        rawStampsDict : `dict[str, Stamps]`
            ROI view of stamps from butler `guider_raw`.
        doSubtractMedian : `bool`, optional
            If True, subtract column bias.
        columnMaskK : `float`, optional
            Threshold factor for column mask detection.
        biasPercentile : `float`, optional
            Percentile for column bias estimation.

        Returns
        -------
        processedStamps : `dict[str, Stamps]`
            New Stamps objects with processed (copied) data.
            Per-stamp column masks are included in each stamp's MaskedImageF.
        """
        processedStamps: dict[str, Stamps] = {}

        for detName, stamps in rawStampsDict.items():
            if len(stamps) == 0:
                processedStamps[detName] = stamps
                continue

            stampList: list[Stamp] = []
            for i in range(len(stamps)):
                # Work on a copy - never modify original
                data = stamps[i].stamp_im.image.array.copy()

                # Compute column bias using low percentile to avoid
                # star contamination (median pulls dark dips at star cols).
                percRows = np.nanpercentile(data, biasPercentile, axis=0)
                rPerc = np.nanpercentile(data, biasPercentile)
                medianValue = np.nanmedian(data)
                # Replace NaN values (fully masked columns) with global median
                percRows = np.where(np.isnan(percRows), medianValue, percRows)

                # Compute column mask on bias-subtracted data
                colMask = getColumnMask(
                    data - percRows[np.newaxis, :] - (rPerc - medianValue),
                    k=columnMaskK,
                )

                # Create mask array for MaskedImageF (BAD=1 for masked cols)
                nRows, nCols = data.shape
                maskArray = np.zeros((nRows, nCols), dtype=np.uint32)
                maskArray[colMask] = 1  # Set BAD bit for masked columns

                # Apply column bias subtraction
                if doSubtractMedian:
                    data = data - percRows[np.newaxis, :] - (rPerc - medianValue)

                # Fill masked columns with global median
                if colMask.any():
                    medianValue = np.nanmedian(data[~colMask]) if (~colMask).any() else np.nan
                    data[colMask] = medianValue

                # Create MaskedImageF with image and mask
                image = ImageF(array=data.astype(np.float32))
                mask = MaskX(nCols, nRows)
                mask.array[:] = maskArray
                outImg = MaskedImageF(image, mask)

                stampList.append(
                    Stamp(
                        stamp_im=outImg,
                        archive_element=stamps[i].archive_element,
                        metadata=stamps[i].metadata,
                    )
                )

            processedStamps[detName] = Stamps(stampList, stamps.metadata, use_mask=True, use_archive=True)

        return processedStamps

    def getAxisRowMap(self) -> dict[str, int]:
        """
        Axis mapping (row axis) for guider detectors.

        Returns
        -------
        axisMap : `dict[str, int]`
            Dictionary with detector names as keys and axis mapping as values.
            The axis mapping is 1 for X and 0 for Y.
        """
        camera = LsstCam.getCamera()
        axisMap: dict[str, int] = {}
        for detName, detNum in self.guiderNameMap.items():
            # Get the detector object from the camera
            detector = camera[detNum]
            # Determine the row axis based on the orientation
            nq = detector.getOrientation().getNQuarter()
            axisMap[detName] = 0 if nq % 2 == 0 else 1
        return axisMap

    def applyGuiderNameSwapIfNeeded(self, dayObs: int) -> None:
        """
        Apply guider name swap (SG0/SG1) for early data if required.

        Parameters
        ----------
        dayObs : `int`
            Observation day (YYYYMMDD) used to decide whether to swap.
        """
        if getattr(self, "_guiderNameMapSwapped", False):
            return  # Already swapped; do nothing
        if dayObs < 20250509:
            newMap = {}
            for detName, detNum in self.guiderNameMap.items():
                if detName.endswith("SG0"):
                    swapped = detName.replace("SG0", "SG1")
                elif detName.endswith("SG1"):
                    swapped = detName.replace("SG1", "SG0")
                else:
                    swapped = detName
                swappedDetNum = self.camera[swapped].getId()
                newMap[swapped] = swappedDetNum
            self.guiderNameMap = newMap
            self._guiderNameMapSwapped = True


def getVisitInfo(butler: Butler, dayObs: int, seqNum: int, scienceDetNum: int) -> VisitInfo:
    """
    Retrieve VisitInfo for a given dayObs / seqNum.

    Parameters
    ----------
    butler : `Butler`
        Active Butler instance.
    dayObs : `int`
        Observation day (YYYYMMDD).
    seqNum : `int`
        Sequence number.
    scienceDetNum : `int`
        Science detector number.

    Returns
    -------
    visitInfo : `VisitInfo`
        VisitInfo object retrieved from butler.
    """
    dataId = {
        "instrument": "LSSTCam",
        "day_obs": dayObs,
        "seq_num": seqNum,
        "detector": scienceDetNum,
    }
    visitInfo = butler.get("raw.visitInfo", dataId)
    return visitInfo


# Helper functions for stamp conversion
def _makeRoiTransforms(metadata: dict, detector: Detector, camera: Camera) -> tuple[tuple, str]:
    """
    Construct ROI transforms and derive amplifier name.

    Parameters
    ----------
    metadata : `dict`
        Exposure-level metadata.
    detector : `Detector`
        The detector.
    camera : `Camera`
        The camera object.

    Returns
    -------
    result : `tuple[tuple, str]`
        ((ccdViewBbox, fwd, back), ampName)
    """
    ampName = "C" + metadata["ROISEG"][7:]
    ccdViewBbox = makeRoiBbox(metadata, camera)
    fwd, back = makeCcdToDvcsTransform(ccdViewBbox, detector.getOrientation().getNQuarter())
    return (ccdViewBbox, fwd, back), ampName


def _convertMaskedImage(
    maskedImage: MaskedImageF,
    stampMetadata: dict[str, Any],
    metadata: dict[str, Any],
    transforms: tuple,
    detector: Detector,
    ampName: str,
    camera: Camera,
    view: str,
) -> Stamp:
    """
    Convert one masked image ROI and build a Stamp.

    Parameters
    ----------
    maskedImage : `MaskedImageF`
        Input masked image for the ROI.
    stampMetadata : `dict`
        Stamp-level metadata (modified in-place).
    metadata : `dict`
        Exposure-level metadata.
    transforms : `tuple`
        Tuple containing (ccdViewBbox, forwardTransform, inverseTransform).
    detector : `Detector`
        Camera detector.
    ampName : `str`
        Amplifier name.
    camera : `Camera`
        Camera object.
    view : `str`
        Output view ('dvcs', 'ccd', or 'roi').

    Returns
    -------
    stamp : `Stamp`
        Converted stamp object.
    """
    ccdViewBbox, fwd, back = transforms
    rawArray = maskedImage.getImage().getArray()
    dvcsArray = roiImageToDvcs(rawArray, metadata, detector, ampName, camera, view=view)
    outImg = MaskedImageF(dvcsArray)
    archiveElement = [ccdViewBbox, fwd, back]
    return Stamp(outImg, archiveElement, metadata=stampMetadata)


def _blankStamp(
    stampIdx: int,
    metadata: dict[str, Any],
    transforms: tuple,
) -> Stamp:
    """
    Create a blank (zero) stamp for a missing index.

    Parameters
    ----------
    stampIdx : `int`
        Missing stamp index.
    metadata : `dict`
        Original metadata container.
    transforms : `tuple`
        (ccdViewBbox, forwardTransform, inverseTransform).

    Returns
    -------
    stamp : `Stamp`
        Placeholder blank stamp with NaN timestamp.
    """
    ccdViewBbox, fwd, back = transforms
    nRows, nCols = int(metadata["ROIROWS"]), int(metadata["ROICOLS"])
    blankArray = np.zeros((nRows, nCols), dtype=np.float32)
    blankImg = ExposureF(MaskedImageF(ImageF(array=blankArray)))
    missingMetadata = metadata.copy()
    missingMetadata["DAQSTAMP"] = stampIdx
    missingMetadata["STMPTMJD"] = np.nan
    archiveElement = [ccdViewBbox, fwd, back]
    return Stamp(blankImg, archiveElement, metadata=missingMetadata)


def convertRawStampsToView(
    rawStamps: Stamps,
    detName: str,
    nStamps: int,
    view: str = "dvcs",
) -> Stamps:
    """
    Convert guider stamps from raw ROI to a requested view ('dvcs', 'ccd', or
    'roi'). Handles missing stamps and preserves metadata and order.

    Parameters
    ----------
    rawStamps : `Stamps`
        Input raw ROI stamps.
    detName : `str`
        Detector name.
    nStamps : `int`
        Target total number of stamps (after filling gaps).
    view : `str`, optional
        Output view: 'dvcs', 'ccd', or 'roi'.

    Returns
    -------
    stampsOut : `Stamps`
        Converted stamps with gaps filled by blank stamps.
    """
    camera = LsstCam.getCamera()
    detector = camera[detName]
    metadata = rawStamps.metadata
    metadataDict = metadata.toDict()

    # Ensure CCDSLOT matches swapped name if applicable
    metadata["CCDSLOT"] = detName[4:7]

    # Align timestamps and find valid/missing indices
    timestamps = standardizeGuiderTimestamps(rawStamps, nStamps)
    validIndices = np.where(~timestamps.mask)[0].tolist()
    missingIndices = np.where(timestamps.mask)[0].tolist()

    # Pre‑compute transforms once
    transforms, ampName = _makeRoiTransforms(metadataDict, detector, camera)

    stampsDict: dict[int, Stamp] = {}
    mIdx = 0  # index into masked images list

    # Column masking is now done in GuiderReader.processStamps() before this
    # function is called, so we no longer need to compute or apply it here.

    for idx in validIndices:
        maskedImage = rawStamps.getMaskedImages()[mIdx]
        stampMeta = rawStamps[mIdx].metadata
        stampMeta["DAQSTAMP"] = stampMeta.get("DAQSTAMP", idx)
        stampsDict[idx] = _convertMaskedImage(
            maskedImage, stampMeta, metadataDict, transforms, detector, ampName, camera, view
        )
        mIdx += 1

    # Fill gaps with blanks
    for idx in missingIndices:
        stampsDict[idx] = _blankStamp(idx, metadataDict, transforms)

    # Assemble in order
    stampList = [stampsDict[i] for i in range(nStamps)]
    return Stamps(stampList, metadata, use_archive=True)


def standardizeGuiderTimestamps(rawStamps: Stamps, nStamps: int) -> Time:
    """
    Return a masked `Time` array of length `nStamps` for one guider.

    Missing stamps are filled with `NaN` and masked so every detector shares a
    uniform, sortable timestamp vector. Result is in MJD, `scale="tai"`.

    Parameters
    ----------
    rawStamps : `Stamps`
        Input stamps (may have missing indices).
    nStamps : `int`
        Desired total length (maximum among guiders).

    Returns
    -------
    fullTimestamps : `Time`
        Masked Time array (scale='tai') with inferred cadence and gaps masked.
    """
    timestampsList = [stamp.metadata.get("STMPTMJD", np.nan) for stamp in rawStamps]
    mjdArray = np.ma.masked_invalid(timestampsList)
    timestamps = Time(mjdArray, format="mjd", scale="tai")

    # infer frequency from valid timestamps
    jdDiff = np.diff(timestamps.jd)
    freqDays = np.nanmedian(jdDiff)
    startJd = timestamps[0].jd

    if np.isnan(freqDays) or freqDays <= 0:
        raise ValueError(
            f"Invalid frequency {freqDays} derived from timestamps. "
            "Ensure that the timestamps are valid and evenly spaced."
        )

    timestampsIdeal = Time(startJd + np.arange(nStamps) * freqDays, format="jd", scale="tai")

    tolerance = 0.5 * freqDays
    actualJd = timestamps.jd

    # Build aligned timestamp list with np.nan where missing
    mjdList = []
    for idealJd in timestampsIdeal.jd:
        # check if any actual timestamp is close to this ideal timestamp
        if np.any(np.abs(actualJd - idealJd) < tolerance):
            # match found — find the closest
            idx = np.argmin(np.abs(actualJd - idealJd))
            mjdList.append(actualJd[idx])
        else:
            # no match — this is a missing timestamp
            mjdList.append(np.nan)

    mjdArray = np.ma.masked_invalid(mjdList)
    fullTimestamps = Time(mjdArray, format="mjd", scale="tai")
    fullTimestamps.freq = freqDays
    return fullTimestamps


@dataclass
class WeatherInfo:
    """
    Container for weather information.

    Parameters
    ----------
    temperature : `float`
        Temperature in degrees Celsius.
    humidity : `float`
        Relative humidity in percent.
    pressure : `float`
        Atmospheric pressure in hPa.
    windSpeed : `float`
        Wind speed in m/s.
    windDir : `float`
        Wind direction in degrees from North.
    """

    temperature: float
    humidity: float
    pressure: float
    windSpeed: float
    windDir: float
    seeing: float

    @staticmethod
    def fromMetadata(metadata: dict) -> WeatherInfo:
        """
        Create a WeatherInfo object from metadata dictionary.

        Parameters
        ----------
        metadata : `dict`
            Metadata dictionary containing weather keys.

        Returns
        -------
        weather : `WeatherInfo`
            Parsed weather information.
        """
        return WeatherInfo(
            temperature=metadata_to_float(metadata, "AIRTEMP", np.nan),
            humidity=metadata_to_float(metadata, "HUMIDITY", np.nan),
            pressure=metadata_to_float(metadata, "PRESSURE", np.nan),
            windSpeed=metadata_to_float(metadata, "WINDSPD", np.nan),
            windDir=metadata_to_float(metadata, "WINDDIR", np.nan),
            seeing=metadata_to_float(metadata, "SEEING", np.nan),
        )


def metadata_to_float(metadata: dict, key: str, default: float = np.nan) -> float:
    """
    Safely convert a metadata value to float.

    Parameters
    ----------
    metadata : `dict`
        Metadata dictionary.
    key : `str`
        Key to look up in the metadata.
    default : `float`, optional
        Default value if key is missing or conversion fails.

    Returns
    -------
    value : `float`
        Converted float value or default.
    """
    value = metadata[key]
    if value is None:
        return default
    else:
        return float(value)


def mad(x: np.ndarray) -> float:
    """
    Median absolute deviation metric for std.

    Parameters
    ----------
    x : `ndarray`
        Input array.

    Returns
    -------
    value : `float`
        Robust standard deviation
    """
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))


def getColumnMask(img: NDArray[np.floating], k: float = 6.0) -> NDArray[np.bool_]:
    """Return a boolean mask for bad columns based on robust median statistics.

    Parameters
    ----------
    img : `ndarray`
        2D image array.
    k : `float`, optional
        Threshold factor for MAD.

    Returns
    -------
    mask : `ndarray`
        Boolean mask of same shape as img, ``True`` for bad columns.
    """
    columnMedian = np.nanmedian(img, axis=0)
    center = np.nanmedian(columnMedian)

    # MAD of column medians
    diffs = np.abs(columnMedian - center)
    s = np.nanmedian(diffs)

    if s == 0 or not np.isfinite(s):
        return np.zeros(img.shape, dtype=bool)

    badCols = diffs > k * s
    return np.broadcast_to(badCols, img.shape).copy()
