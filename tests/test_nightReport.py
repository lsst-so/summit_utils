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

import datetime
import itertools
import os
import tempfile
import unittest
from unittest import mock

import matplotlib as mpl
from astro_metadata_translator import ObservationInfo
from numpy.random import shuffle

import lsst.utils.tests

mpl.use("Agg")

import lsst.summit.utils.butlerUtils as butlerUtils  # noqa: E402
from lsst.daf.butler import Butler  # noqa: E402
from lsst.summit.utils.nightReport import ColorAndMarker, NightReport  # noqa: E402
from lsst.summit.utils.utils import getSite  # noqa: E402


class NightReportTestCase(lsst.utils.tests.TestCase):
    # class attributes populated in setUpClass
    butler: Butler
    dayObs: int
    report: NightReport
    nImages: int
    seqNums: list[int]

    @classmethod
    def setUpClass(cls) -> None:
        try:
            if getSite() == "jenkins":
                raise unittest.SkipTest("Skip running butler-driven tests in Jenkins.")
            cls.butler = butlerUtils.makeDefaultLatissButler()
        except FileNotFoundError:
            raise unittest.SkipTest("Skipping tests that require the LATISS butler repo.")

        cls.dayObs = 20200314  # has 377 images and data also exists on the TTS & summit

        # Do the init in setUpClass because this takes about 35s for 20200314
        cls.report = NightReport(cls.butler, cls.dayObs)
        # number of images isn't necessarily the same as the number for the
        # the dayObs in the registry becacuse of the test stands/summit
        # having partial data, so get the number of images from the length
        # of the scraped data. Not ideal, but best that can be done due to
        # only having partial days in the test datasets.
        cls.nImages = len(cls.report.data.keys())
        cls.seqNums = list(cls.report.data.keys())

    def test_saveAndLoad(self) -> None:
        """Test that a NightReport can save itself, and be loaded back."""
        writeDir = tempfile.mkdtemp()
        saveFile = os.path.join(writeDir, f"testNightReport_{self.dayObs}.pickle")
        self.report.save(saveFile)
        self.assertTrue(os.path.exists(saveFile))

        loaded = NightReport(self.butler, self.dayObs, saveFile)
        self.assertIsInstance(loaded, lsst.summit.utils.nightReport.NightReport)
        self.assertGreaterEqual(len(loaded.data), 1)
        self.assertEqual(loaded.dayObs, self.dayObs)

        # TODO: add a self.assertRaises on a mismatched dayObs

    def test_getSortedData(self) -> None:
        """Test the _getSortedData returns the seqNums in order."""
        shuffledKeys = list(self.report.data.keys())
        shuffle(shuffledKeys)
        shuffledData = {k: self.report.data[k] for k in shuffledKeys}

        sortedData = self.report._getSortedData(shuffledData)
        sortedKeys = sorted(list(sortedData.keys()))
        self.assertEqual(sortedKeys, list(self.report.data.keys()))
        return

    def test_getExpRecordDictForDayObs(self) -> None:
        """Test getExpRecordDictForDayObs.

        Test it returns a dict of dicts, keyed by integer seqNums.
        """
        expRecDict = self.report.getExpRecordDictForDayObs(self.dayObs)
        self.assertIsInstance(expRecDict, dict)
        self.assertGreaterEqual(len(expRecDict), 1)

        # check all the keys are ints
        seqNums = list(expRecDict.keys())
        self.assertTrue(all(isinstance(s, int) for s in seqNums))

        # check all the values are dicts
        self.assertTrue(all(isinstance(expRecDict[s], dict) for s in seqNums))
        return

    def test_getObsInfoAndMetadataForSeqNum(self) -> None:
        """Test that getObsInfoAndMetadataForSeqNum returns the correct
        types.
        """
        seqNum = self.seqNums[0]
        obsInfo, md = self.report.getObsInfoAndMetadataForSeqNum(seqNum)
        self.assertIsInstance(obsInfo, ObservationInfo)
        self.assertIsInstance(md, dict)
        return

    def test_rebuild(self) -> None:
        """Test that rebuild does nothing, as no data will be being added.

        NB Do not call full=True on this, as it will double the length of the
        tests and they're already extremely slow.
        """
        lenBefore = len(self.report.data)
        self.report.rebuild()
        self.assertEqual(len(self.report.data), lenBefore)
        return

    def test_getExposureMidpoint(self) -> None:
        """Test the exposure midpoint calculation"""
        # we would like a non-zero exptime exposure really
        seqNumToUse = 0
        for seqNum in self.report.data.keys():
            expTime = self.report.data[seqNum]["exposure_time"]
            if expTime > 0:
                seqNumToUse = seqNum
                break

        midPoint = self.report.getExposureMidpoint(seqNumToUse)
        record = self.report.data[seqNumToUse]

        if expTime == 0:
            self.assertGreaterEqual(midPoint, record["datetime_begin"].to_datetime())
            self.assertLessEqual(midPoint, record["datetime_end"].to_datetime())
        else:
            self.assertGreater(midPoint, record["datetime_begin"].to_datetime())
            self.assertLess(midPoint, record["datetime_end"].to_datetime())
        return

    def test_getTimeDeltas(self) -> None:
        """Test the time delta calculation returns a dict."""
        dts = self.report.getTimeDeltas()
        self.assertIsInstance(dts, dict)
        return

    def test_makeStarColorAndMarkerMap(self) -> None:
        """Test the color map maker returns a dict of ColorAndMarker
        objects.
        """
        cMap = self.report.makeStarColorAndMarkerMap(self.report.stars)
        self.assertEqual(len(cMap), len(self.report.stars))
        self.assertIsInstance(cMap, dict)
        values = list(cMap.values())
        self.assertTrue(all(isinstance(value, ColorAndMarker) for value in values))
        return

    def test_printObsTable(self) -> None:
        """Test that a the printObsTable() method prints out the correct
        number of lines.
        """
        with mock.patch("sys.stdout") as fake_stdout:
            self.report.printObsTable()

        # newline for each row plus header line, plus the line with dashes
        self.assertEqual(len(fake_stdout.mock_calls), 2 * (self.nImages + 2))

    def test_plotPerObjectAirMass(self) -> None:
        """Test that a the per-object airmass plots runs."""
        # We assume matplotlib is making plots, so just check that these
        # don't crash.

        # Default plotting:
        self.report.plotPerObjectAirMass()
        # plot with only one object as a str not a list of str
        self.report.plotPerObjectAirMass(objects=self.report.stars[0])
        # plot with first two objects as a list
        self.report.plotPerObjectAirMass(objects=self.report.stars[0:2])
        # flip y axis option
        self.report.plotPerObjectAirMass(airmassOneAtTop=True)
        # flip and select stars
        self.report.plotPerObjectAirMass(objects=self.report.stars[0], airmassOneAtTop=True)  # both

    def test_makeAltAzCoveragePlot(self) -> None:
        """Test that a the polar coverage plotting code runs."""
        # We assume matplotlib is making plots, so just check that these
        # don't crash.

        # test the default case
        self.report.makeAltAzCoveragePlot()
        # plot with only one object as a str not a list of str
        self.report.makeAltAzCoveragePlot(objects=self.report.stars[0])
        # plot with first two objects as a list
        self.report.makeAltAzCoveragePlot(objects=self.report.stars[0:2])
        # test turning lines off
        self.report.makeAltAzCoveragePlot(objects=self.report.stars[0:2], withLines=False)

    def test_calcShutterTimes(self) -> None:
        timings = self.report.calcShutterTimes()
        if not timings:
            return  # if the day has no on-sky observations, this returns None
        efficiency = 100 * (timings["scienceTimeTotal"] / timings["nightLength"])
        self.assertGreater(efficiency, 0)
        self.assertLessEqual(efficiency, 100)

    def test_getDatesForSeqNums(self) -> None:
        dateTimeDict = self.report.getDatesForSeqNums()
        self.assertIsInstance(dateTimeDict, dict)
        self.assertTrue(all(isinstance(seqNum, int) for seqNum in dateTimeDict.keys()))
        self.assertTrue(all(isinstance(seqNum, datetime.datetime) for seqNum in dateTimeDict.values()))

    def test_doesNotRaise(self) -> None:
        """Tests for things which are hard to test, so just make sure they
        run.
        """
        self.report.printShutterTimes()
        for sample, includeRaw in itertools.product((True, False), (True, False)):
            self.report.printAvailableKeys(sample=sample, includeRaw=includeRaw)
        self.report.printObsTable()
        for threshold, includeCalibs in itertools.product((0, 1, 10), (True, False)):
            self.report.printObsGaps(threshold=threshold, includeCalibs=includeCalibs)

    def test_internals(self) -> None:
        startNum = self.report.getObservingStartSeqNum()
        self.assertIsInstance(startNum, int)
        self.assertGreater(startNum, 0)  # the day starts at 1, so zero would be an error of some sort

        starsFromGetter = self.report.getObservedObjects()
        self.assertIsInstance(starsFromGetter, list)
        self.assertSetEqual(set(starsFromGetter), set(self.report.stars))

        starsFromGetter = self.report.getObservedObjects(ignoreTileNum=True)
        self.assertLessEqual(len(starsFromGetter), len(self.report.stars))

        # check the internal color map has the right number of items
        self.assertEqual(len(self.report.cMap), len(starsFromGetter))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: object) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
