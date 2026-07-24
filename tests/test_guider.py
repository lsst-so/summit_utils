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

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import lsst.utils.tests
from lsst.daf.butler import Butler
from lsst.meas.algorithms.stamps import Stamps
from lsst.summit.utils.butlerUtils import makeDefaultButler
from lsst.summit.utils.guiders.metrics import GuiderMetricsBuilder
from lsst.summit.utils.guiders.plotting import GuiderPlotter
from lsst.summit.utils.guiders.reading import GuiderData, GuiderReader
from lsst.summit.utils.guiders.seeing import CorrelationAnalysis, GuiderSeeing
from lsst.summit.utils.guiders.tracking import GuiderStarTracker
from lsst.summit.utils.utils import getSite


class GuiderTestCase(unittest.TestCase):
    """Tests of the run method with fake data."""

    def setUp(self) -> None:
        try:
            if getSite() == "jenkins":
                raise unittest.SkipTest("Skip running butler-driven tests in Jenkins.")
            self.butler = makeDefaultButler("LSSTCam", embargo=False)
        except FileNotFoundError:
            raise unittest.SkipTest("Skipping tests that require the LSSTCam butler repo.")
        self.assertIsInstance(self.butler, Butler)

        self.dayObs = 20250629
        self.seqNum = 340
        self.expId = 2025062900340

        self.reader = GuiderReader(self.butler, view="dvcs")
        self.guiderData = self.reader.get(dayObs=self.dayObs, seqNum=self.seqNum)
        self.tracker = GuiderStarTracker(self.guiderData)
        self.stars = self.tracker.trackGuiderStars(refCatalog=None)
        self.plotter = GuiderPlotter(self.guiderData, starsDf=self.stars)
        self.metricsBuilder = GuiderMetricsBuilder(self.stars, self.guiderData.nMissingStamps)

    def test_types(self) -> None:
        self.assertIsInstance(self.guiderData.header, dict)
        self.assertIsInstance(self.guiderData.stampsMap, dict)

        expectedKeys = (
            "R00_SG0",
            "R00_SG1",
            "R04_SG0",
            "R04_SG1",
            "R40_SG0",
            "R40_SG1",
            "R44_SG0",
            "R44_SG1",
        )
        self.assertTrue(
            all(key in self.guiderData.stampsMap for key in expectedKeys),
            "Not all expected guider datasets are present in the data.",
        )

        detName = "R00_SG0"
        single = self.guiderData[detName, 0]
        self.assertIsInstance(single, np.ndarray)
        stack = self.guiderData.getStampArrayCoadd(detName=detName)
        self.assertIsInstance(stack, np.ndarray)
        self.assertEqual(single.shape, (400, 400))

        return

    def test_reading(self) -> None:
        """Test that the reader can read the expected data."""
        self.assertTrue(self.guiderData.isMedianSubtracted, "isMedianSubtracted not set correctly")
        for detName in self.guiderData.guiderNames:
            single = self.guiderData[detName, 0]
            self.assertIsInstance(single, np.ndarray)
            self.assertEqual(single.shape, (400, 400))
            self.assertLessEqual(abs(np.nanmedian(single)), 10, "median subtracted median is too high")

            stack = self.guiderData.getStampArrayCoadd(detName=detName)
            self.assertIsInstance(stack, np.ndarray)
            self.assertEqual(stack.shape, (400, 400))
            self.assertLessEqual(abs(np.nanmedian(stack)), 10, "stack median subtracted median is too high")

            fullStamps = self.guiderData[detName]
            self.assertIsInstance(fullStamps, Stamps)

        noMedianSubtracted = self.reader.get(dayObs=self.dayObs, seqNum=self.seqNum, doSubtractMedian=False)
        self.assertIsInstance(noMedianSubtracted, GuiderData)
        self.assertFalse(noMedianSubtracted.isMedianSubtracted, "isMedianSubtracted not set correctly")
        for detName in noMedianSubtracted.guiderNames:
            single = noMedianSubtracted[detName, 0]
            self.assertIsInstance(single, np.ndarray)
            self.assertEqual(single.shape, (400, 400))
            self.assertGreater(abs(np.nanmedian(single)), 500, "median un-subtracted median is too low")

            stack = noMedianSubtracted.getStampArrayCoadd(detName=detName)
            self.assertIsInstance(stack, np.ndarray)
            self.assertEqual(stack.shape, (400, 400))
            self.assertGreater(abs(np.nanmedian(single)), 500, "median un-subtracted median is too low")

            fullStamps = noMedianSubtracted[detName]
            self.assertIsInstance(fullStamps, Stamps)

    def test_detection(self) -> None:
        self.assertIsInstance(self.stars, pd.DataFrame)
        requiredColumns = (
            "xroi",
            "yroi",
            "dx",
            "dy",
            "dalt",
            "daz",
            "fwhm",
            "trackid",
            "expid",
        )
        self.assertTrue(
            all(col in self.stars.columns for col in requiredColumns),
            "Not all required columns are present in the stars DataFrame.",
        )

        maxStampIndex = max(self.stars["stamp"])
        nStamps = len(self.guiderData)  # we should make an attribute for this

        # we skip the first stamp, so the max index should be nStamps - 1
        self.assertEqual(maxStampIndex, nStamps - 1, "Did not get detections for all expected stamps")

    def testPlotMosaicFullView(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.plotMosaic(stampNum=-1, cutoutSize=-1, plo=50, phi=98, saveAs=tmp.name)
            os.fsync(tmp.fileno())  # be strict
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testPlotMosaicZoomView(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.plotMosaic(stampNum=-1, cutoutSize=12, plo=50, phi=98, saveAs=tmp.name)
            os.fsync(tmp.fileno())  # be strict
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testPlotMosaicStampZoomView(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.plotMosaic(stampNum=4, cutoutSize=12, plo=50, phi=98, saveAs=tmp.name)
            os.fsync(tmp.fileno())  # be strict
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testStripPlotPsf(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.stripPlot(plotType="psf", saveAs=tmp.name)
            os.fsync(tmp.fileno())
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testStripPlotCentroidAltAz(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.stripPlot(plotType="centroidAltAz", saveAs=tmp.name)
            os.fsync(tmp.fileno())
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testStripPlotFlux(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.stripPlot(plotType="flux", saveAs=tmp.name)
            os.fsync(tmp.fileno())
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testStripPlotShape(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.stripPlot(plotType="ellip", saveAs=tmp.name)
            os.fsync(tmp.fileno())
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testMakeGif(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=True) as tmp:
            # test the crop and zoom as a gif
            self.plotter.makeAnimation(cutoutSize=14, plo=50, phi=98, saveAs=tmp.name)
            os.fsync(tmp.fileno())
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testMakeMp4(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            # test the full frame as an mp4
            self.plotter.makeAnimation(cutoutSize=-1, plo=50, phi=98, saveAs=tmp.name)
            os.fsync(tmp.fileno())
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def test_metrics(self) -> None:
        # Check that metrics can be built without error
        metrics = self.metricsBuilder.buildMetrics(self.expId)
        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertGreater(len(metrics), 0, "Metrics DataFrame is empty")

        # These are what's currently there. Why is the 8th guider missing?
        expectedColumns = (
            "n_guiders",
            "n_stars",
            "n_missing_stamps",
            "n_measurements",
            "fraction_possible_measurements",
            "exptime",
            "R00_SG0",
            "R00_SG1",
            "R04_SG0",
            "R04_SG1",
            "R40_SG0",
            "R40_SG1",
            "R44_SG0",
            "R44_SG1",
            "az_drift_slope",
            "az_drift_intercept",
            "az_drift_trend_rmse",
            "az_drift_global_std",
            "az_drift_outlier_frac",
            "az_drift_slope_significance",
            "az_drift_nsize",
            "alt_drift_slope",
            "alt_drift_intercept",
            "alt_drift_trend_rmse",
            "alt_drift_global_std",
            "alt_drift_outlier_frac",
            "alt_drift_slope_significance",
            "alt_drift_nsize",
            "rotator_slope",
            "rotator_intercept",
            "rotator_trend_rmse",
            "rotator_global_std",
            "rotator_outlier_frac",
            "rotator_slope_significance",
            "rotator_nsize",
            "mag_slope",
            "mag_intercept",
            "mag_trend_rmse",
            "mag_global_std",
            "mag_outlier_frac",
            "mag_slope_significance",
            "mag_nsize",
            "psf_slope",
            "psf_intercept",
            "psf_trend_rmse",
            "psf_global_std",
            "psf_outlier_frac",
            "psf_slope_significance",
            "psf_nsize",
        )
        for col in expectedColumns:
            self.assertIn(col, metrics.columns, f"Column {col} is missing from metrics DataFrame")

        # check this runs without error
        self.metricsBuilder.printSummary()

    def testTomographicSeeing(self) -> None:
        analysis = CorrelationAnalysis(self.stars, self.expId)
        variance = np.nanmedian(analysis.measureVariance())
        self.assertGreater(variance, 0, "Variance should be positive and non-zero")
        self.assertIsInstance(variance, float)

        seeing = analysis.measureTomographicSeeing()
        self.assertIsInstance(seeing, GuiderSeeing)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: object) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
