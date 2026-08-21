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

__all__ = ["SpectrumExaminer"]

import warnings
from itertools import groupby
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.stats import sigma_clip
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.optimize import curve_fit

import lsst.afw.display as afwDisplay
import lsst.afw.image as afwImage
from lsst.atmospec.processStar import ProcessStarTask
from lsst.geom import Box2I
from lsst.obs.lsst.translators.lsst import FILTER_DELIMITER
from lsst.pipe.tasks.quickFrameMeasurement import QuickFrameMeasurementTask, QuickFrameMeasurementTaskConfig
from lsst.summit.utils.utils import getImageStats
from lsst.utils.plotting.figures import make_figure


class SpectrumExaminer:
    """Task for the QUICK spectral extraction of single-star dispersed images.

    For a full description of how this tasks works, see the run() method.
    """

    # ConfigClass = SummarizeImageTaskConfig
    # _DefaultName = "summarizeImage"

    def __init__(
        self,
        exp: afwImage.Exposure,
        display: afwDisplay.Display = None,
        debug: bool = False,
        savePlotAs: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.exp = exp
        self.display = display
        self.debug = debug
        self.savePlotAs = savePlotAs
        self.fig = make_figure(figsize=(10, 10))

        qfmTaskConfig = QuickFrameMeasurementTaskConfig()
        self.qfmTask = QuickFrameMeasurementTask(config=qfmTaskConfig)

        pstConfig = ProcessStarTask.ConfigClass()
        pstConfig.offsetFromMainStar = 400
        self.processStarTask = ProcessStarTask(config=pstConfig)

        self.imStats = getImageStats(exp)

        self.init()

    @staticmethod
    def bboxToAwfDisplayLines(box: Box2I) -> list[list[tuple[int, int]]]:
        """Takes a bbox, returns a list of lines such that they can be plotted:

        for line in lines:
            display.line(line, ctype='red')
        """
        x0 = box.beginX
        x1 = box.endX
        y0 = box.beginY
        y1 = box.endY
        return [[(x0, y0), (x1, y0)], [(x0, y0), (x0, y1)], [(x1, y0), (x1, y1)], [(x0, y1), (x1, y1)]]

    def eraseDisplay(self) -> None:
        if self.display:
            self.display.erase()

    def displaySpectrumBbox(self) -> None:
        if self.display:
            lines = self.bboxToAwfDisplayLines(self.spectrumbbox)
            for line in lines:
                self.display.line(line, ctype="red")
        else:
            print("No display set")

    def displayStarLocation(self) -> None:
        if self.display:
            self.display.dot("x", *self.qfmResult.brightestObjCentroid, size=50)
            self.display.dot("o", *self.qfmResult.brightestObjCentroid, size=50)
        else:
            print("No display set")

    def calcGoodSpectrumSection(self, threshold: int = 5, windowSize: int = 5) -> tuple[int, int]:
        length = len(self.ridgeLineLocations)
        chunks = length // windowSize
        stddevs = []
        for i in range(chunks + 1):
            stddevs.append(np.std(self.ridgeLineLocations[i * windowSize : (i + 1) * windowSize]))

        goodPoints = np.where(np.asarray(stddevs) < threshold)[0]
        minPoint = (goodPoints[2] - 2) * windowSize
        maxPoint = (goodPoints[-3] + 3) * windowSize
        minPoint = max(minPoint, 0)
        maxPoint = min(maxPoint, length)
        if self.debug:
            plt.plot(range(0, length + 1, windowSize), stddevs)
            plt.hlines(threshold, 0, length, colors="r", ls="dashed")
            plt.vlines(minPoint, 0, max(stddevs) + 10, colors="k", ls="dashed")
            plt.vlines(maxPoint, 0, max(stddevs) + 10, colors="k", ls="dashed")
            plt.title(f"Ridgeline scatter, windowSize={windowSize}")

        return (minPoint, maxPoint)

    def fit(self) -> None:
        def gauss(
            x: float | npt.NDArray[np.float64], a: float, x0: float, sigma: float
        ) -> float | npt.NDArray[np.float64]:
            return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

        data = self.spectrumData[self.goodSlice]
        nRows, nCols = data.shape
        # don't subtract the row median or even a percentile - seems bad
        # fitting a const also seems bad - needs some better thought

        parameters = np.zeros((nRows, 3))
        pCovs = []
        xs = np.arange(nCols)
        for rowNum, row in enumerate(data):
            peakPos = self.ridgeLineLocations[rowNum]
            amplitude = row[peakPos]
            width = 7
            try:
                pars, pCov = curve_fit(gauss, xs, row, [amplitude, peakPos, width], maxfev=100)
                pCovs.append(pCov)
            except RuntimeError:
                pars = [np.nan] * 3
            if not np.all([p < 1e7 for p in pars]):
                pars = [np.nan] * 3
            parameters[rowNum] = pars

        parameters[:, 0] = np.abs(parameters[:, 0])
        parameters[:, 2] = np.abs(parameters[:, 2])
        self.parameters = parameters

    def plot(self) -> None:
        # spectrum
        gs = self.fig.add_gridspec(4, 4)
        ax0 = self.fig.add_subplot(gs[0, 0:3])
        ax0.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
        d = self.spectrumData[self.goodSlice].T
        vmin = np.percentile(d, 1)
        vmax = np.percentile(d, 99)
        pos = ax0.imshow(self.spectrumData[self.goodSlice].T, vmin=vmin, vmax=vmax, origin="lower")
        div = make_axes_locatable(ax0)
        cax = div.append_axes("bottom", size="7%", pad="8%")
        self.fig.colorbar(pos, cax=cax, orientation="horizontal", label="Counts")

        # spectrum histogram
        axHist = self.fig.add_subplot(gs[0, 3])
        data = self.spectrumData
        histMax = np.nanpercentile(data, 99.99)
        histMin = np.nanpercentile(data, 0.001)
        axHist.hist(data[(data >= histMin) & (data <= histMax)].flatten(), bins=100)
        underflow = len(data[data < histMin])
        overflow = len(data[data > histMax])
        axHist.set_yscale("log", nonpositive="clip")
        axHist.set_title("Spectrum pixel histogram")
        text = f"Underflow = {underflow}"
        text += f"\nOverflow = {overflow}"
        anchored_text = AnchoredText(text, loc="upper right", pad=0.5)
        axHist.add_artist(anchored_text)

        # peak fluxes
        ax1 = self.fig.add_subplot(gs[1, 0:3])
        ax1.plot(self.ridgeLineValues[self.goodSlice], label="Raw peak value")
        ax1.plot(self.parameters[:, 0], label="Fitted amplitude")
        ax1.axhline(self.continuumFlux98, ls="dashed", color="g")
        ax1.set_ylabel("Peak amplitude (ADU)")
        ax1.set_xlabel("Spectrum position (pixels)")
        ax1.legend(
            title=f"Continuum flux = {self.continuumFlux98:.0f} ADU",
            loc="center right",
            framealpha=0.2,
            facecolor="black",
        )
        ax1.set_title("Ridgeline plot")

        # FWHM
        ax2 = self.fig.add_subplot(gs[2, 0:3])
        ax2.plot(self.parameters[:, 2] * 2.355, label="FWHM (pix)")
        fwhmValues = self.parameters[:, 2] * 2.355
        amplitudes = self.parameters[:, 0]
        minVal, maxVal = self.getStableFwhmRegion(fwhmValues, amplitudes)
        medianFwhm, bestFwhm = self.getMedianAndBestFwhm(fwhmValues, minVal, maxVal)

        ax2.axhline(medianFwhm, ls="dashed", color="k", label=f"Median FWHM = {medianFwhm:.1f} pix")
        ax2.axhline(bestFwhm, ls="dashed", color="r", label=f"Best FWHM = {bestFwhm:.1f} pix")
        ax2.axvline(minVal, ls="dashed", color="k", alpha=0.2)
        ax2.axvline(maxVal, ls="dashed", color="k", alpha=0.2)
        ymin = max(np.nanmin(fwhmValues) - 5, 0)
        if not np.isnan(medianFwhm):
            ymax = medianFwhm * 2
        else:
            ymax = 5 * ymin
        ax2.set_ylim(ymin, ymax)
        ax2.set_ylabel("FWHM (pixels)")
        ax2.set_xlabel("Spectrum position (pixels)")
        ax2.legend(loc="upper right", framealpha=0.2, facecolor="black")
        ax2.set_title("Spectrum FWHM")

        # row fluxes
        ax3 = self.fig.add_subplot(gs[3, 0:3])
        ax3.plot(self.rowSums[self.goodSlice], label="Sum across row")
        ax3.set_ylabel("Total row flux (ADU)")
        ax3.set_xlabel("Spectrum position (pixels)")
        ax3.legend(framealpha=0.2, facecolor="black")
        ax3.set_title("Row sums")

        # textbox top
        #         ax4 = plt.subplot2grid((4, 4), (1, 3))
        ax4 = self.fig.add_subplot(gs[1:3, 3])
        text = "short text"
        text = self.generateStatsTextboxContent(0)
        text += self.generateStatsTextboxContent(1)
        text += self.generateStatsTextboxContent(2)
        text += self.generateStatsTextboxContent(3)
        stats_text = AnchoredText(
            text,
            loc="center",
            pad=0.5,
            prop=dict(size=10.5, ma="left", backgroundcolor="white", color="black", family="monospace"),
        )
        ax4.add_artist(stats_text)
        ax4.axis("off")

        # textbox middle
        if self.debug:
            ax5 = self.fig.add_subplot(gs[2, 3])
            text = self.generateStatsTextboxContent(-1)
            stats_text = AnchoredText(
                text,
                loc="center",
                pad=0.5,
                prop=dict(size=10.5, ma="left", backgroundcolor="white", color="black", family="monospace"),
            )
            ax5.add_artist(stats_text)
            ax5.axis("off")

        self.fig.tight_layout()

        if self.savePlotAs:
            self.fig.savefig(self.savePlotAs)

    def init(self) -> None:
        pass

    def generateStatsTextboxContent(self, section: int) -> str:
        x, y = self.qfmResult.brightestObjCentroid

        vi = self.exp.visitInfo
        exptime = vi.exposureTime

        fullFilterString = self.exp.filter.physicalLabel
        filt = fullFilterString.split(FILTER_DELIMITER)[0]
        grating = fullFilterString.split(FILTER_DELIMITER)[1]

        airmass = vi.getBoresightAirmass()
        rotangle = vi.getBoresightRotAngle().asDegrees()

        azAlt = vi.getBoresightAzAlt()
        az = azAlt[0].asDegrees()
        el = azAlt[1].asDegrees()

        obj = self.exp.visitInfo.object

        lines = []

        if section == 0:
            lines.append("----- Star stats -----")
            lines.append(f"Star centroid @  {x:.0f}, {y:.0f}")
            lines.append(f"Star max pixel = {self.starPeakFlux:,.0f} ADU")
            lines.append(f"Star Ap25 flux = {self.qfmResult.brightestObjApFlux25:,.0f} ADU")
            lines.extend(["", ""])  # section break
            return "\n".join([line for line in lines])

        if section == 1:
            lines.append("------ Image stats ---------")
            imageMedian = np.median(self.exp.image.array)
            lines.append(f"Image median   = {imageMedian:.2f} ADU")
            lines.append(f"Exposure time  = {exptime:.2f} s")
            lines.extend(["", ""])  # section break
            return "\n".join([line for line in lines])

        if section == 2:
            lines.append("------- Rate stats ---------")
            lines.append(f"Star max pixel    = {self.starPeakFlux / exptime:,.0f} ADU/s")
            lines.append(f"Spectrum contiuum = {self.continuumFlux98 / exptime:,.1f} ADU/s")
            lines.extend(["", ""])  # section break
            return "\n".join([line for line in lines])

        if section == 3:
            lines.append("----- Observation info -----")
            lines.append(f"object  = {obj}")
            lines.append(f"filter  = {filt}")
            lines.append(f"grating = {grating}")
            lines.append(f"rotpa   = {rotangle:.1f}")

            lines.append(f"az      = {az:.1f}")
            lines.append(f"el      = {el:.1f}")
            lines.append(f"airmass = {airmass:.3f}")
            return "\n".join([line for line in lines])

        if section == -1:  # special -1 for debug
            lines.append("---------- Debug -----------")
            lines.append(f"spectrum bbox: {self.spectrumbbox}")
            lines.append(f"Good range = {self.goodSpectrumMinY},{self.goodSpectrumMaxY}")
            return "\n".join([line for line in lines])

        return ""

    def run(self) -> None:
        self.qfmResult = self.qfmTask.run(self.exp)
        self.intCentroidX = int(np.round(self.qfmResult.brightestObjCentroid)[0])
        self.intCentroidY = int(np.round(self.qfmResult.brightestObjCentroid)[1])
        self.starPeakFlux = self.exp.image.array[self.intCentroidY, self.intCentroidX]

        self.spectrumbbox = self.processStarTask.calcSpectrumBBox(
            self.exp, self.qfmResult.brightestObjCentroid, 200
        )
        self.spectrumData = self.exp.image[self.spectrumbbox].array

        self.ridgeLineLocations = np.argmax(self.spectrumData, axis=1)
        self.ridgeLineValues = self.spectrumData[
            range(self.spectrumbbox.getHeight()), self.ridgeLineLocations
        ]
        self.rowSums = np.sum(self.spectrumData, axis=1)

        coords = self.calcGoodSpectrumSection()
        self.goodSpectrumMinY = coords[0]
        self.goodSpectrumMaxY = coords[1]
        self.goodSlice = slice(coords[0], coords[1])

        self.continuumFlux90 = np.percentile(self.ridgeLineValues, 90)  # for emission stars
        self.continuumFlux98 = np.percentile(self.ridgeLineValues, 98)  # for most stars

        self.fit()
        self.plot()

        return

    @staticmethod
    def getMedianAndBestFwhm(fwhmValues: np.ndarray, minIndex: int, maxIndex: int) -> tuple[float, float]:
        with warnings.catch_warnings():  # to supress nan warnings, which are fine
            warnings.simplefilter("ignore")
            clippedValues = sigma_clip(fwhmValues[minIndex:maxIndex])
            # cast back with asArray needed becase sigma_clip returns
            # masked array which doesn't play nice with np.nan<med/percentile>
            clippedValues = np.asarray(clippedValues)
            medianFwhm = np.nanmedian(clippedValues)
            bestFocusFwhm = np.nanpercentile(np.asarray(clippedValues), 2)
        return medianFwhm, bestFocusFwhm

    def getStableFwhmRegion(
        self, fwhmValues: np.ndarray, amplitudes: np.ndarray, smoothing: int = 1, maxDifferential: int = 4
    ) -> tuple[int, int]:
        # smooth the fwhmValues values
        # differentiate
        # take the longest contiguous region of 1s
        # check section corresponds to top 25% in ampl to exclude 2nd order
        # if not, pick next longest run, etc
        # walk out from ends of that list over bumps smaller than maxDiff

        smoothFwhm = np.convolve(fwhmValues, np.ones(smoothing) / smoothing, mode="same")
        diff = np.diff(smoothFwhm, append=smoothFwhm[-1])

        indices = np.where(1 - np.abs(diff) < 1)[0]
        diffIndices = np.diff(indices)

        # [list(g) for k, g in groupby('AAAABBBCCD')] -->[['A', 'A', 'A', 'A'],
        #                              ... ['B', 'B', 'B'], ['C', 'C'], ['D']]
        indexLists = [list(g) for k, g in groupby(diffIndices)]
        listLengths = [len(lst) for lst in indexLists]

        amplitudeThreshold = np.nanpercentile(amplitudes, 75)
        sortedListLengths = sorted(listLengths)

        longestListLength = 0
        longestListIndex = 0
        for listLength in sortedListLengths[::-1]:
            longestListLength = listLength
            longestListIndex = listLengths.index(longestListLength)
            longestListStartTruePosition = int(np.sum(listLengths[0:longestListIndex]))
            longestListStartTruePosition += int(longestListLength / 2)  # we want the mid-run value
            if amplitudes[longestListStartTruePosition] > amplitudeThreshold:
                break

        startOfLongList = int(np.sum(listLengths[0:longestListIndex]))
        endOfLongList = startOfLongList + longestListLength

        endValue = endOfLongList
        for lst in indexLists[longestListIndex + 1 :]:
            value = lst[0]
            if value > maxDifferential:
                break
            endValue += len(lst)

        startValue = startOfLongList
        for lst in indexLists[longestListIndex - 1 :: -1]:
            value = lst[0]
            if value > maxDifferential:
                break
            startValue -= len(lst)

        startValue = int(max(0, startValue))
        endValue = int(min(len(fwhmValues), endValue))

        if not self.debug:
            return startValue, endValue

        medianFwhm, bestFocusFwhm = self.getMedianAndBestFwhm(fwhmValues, startValue, endValue)
        xlim = (-20, len(fwhmValues))

        plt.figure(figsize=(10, 6))
        plt.plot(fwhmValues)
        plt.vlines(startValue, 0, 50, "r")
        plt.vlines(endValue, 0, 50, "r")
        plt.hlines(medianFwhm, xlim[0], xlim[1])
        plt.hlines(bestFocusFwhm, xlim[0], xlim[1], "r", ls="--")

        plt.vlines(startOfLongList, 0, 50, "g")
        plt.vlines(endOfLongList, 0, 50, "g")

        plt.ylim(0, 200)
        plt.xlim(xlim)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(diffIndices)
        plt.vlines(startValue, 0, 50, "r")
        plt.vlines(endValue, 0, 50, "r")

        plt.vlines(startOfLongList, 0, 50, "g")
        plt.vlines(endOfLongList, 0, 50, "g")
        plt.ylim(0, 30)
        plt.xlim(xlim)
        plt.show()
        return startValue, endValue
