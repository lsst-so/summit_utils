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

import copy
import datetime
import os
import random
import unittest
from typing import Any, Iterable

import lsst.daf.butler as dafButler
import lsst.utils.tests
from lsst.daf.butler import DatasetRef, DimensionRecord
from lsst.daf.butler.direct_butler import DirectButler
from lsst.resources import ResourcePath
from lsst.summit.utils.butlerUtils import removeDataProduct  # noqa: F401
from lsst.summit.utils.butlerUtils import (
    RECENT_DAY,
    _assureDict,
    _dayobs_present,
    _expid_present,
    _get_dayobs_key,
    _get_expid_key,
    _get_seqnum_key,
    _seqnum_present,
    fillDataId,
    getDatasetRefForDataId,
    getDayObs,
    getDayObsSeqNumFromExposureId,
    getDaysWithData,
    getExpId,
    getExpIdFromDayObsSeqNum,
    getExpRecord,
    getExpRecordFromDataId,
    getLatissDefaultCollections,
    getLatissOnSkyDataIds,
    getMostRecentDataId,
    getMostRecentDayObs,
    getSeqNum,
    getSeqNumsForDayObs,
    getSite,
    makeDefaultLatissButler,
    sanitizeDayObs,
    sortRecordsByDayObsThenSeqNum,
    updateDataId,
    updateDataIdOrDataCord,
)


class ButlerUtilsTestCase(lsst.utils.tests.TestCase):
    """A test case for testing sky position offsets for exposures."""

    def setUp(self) -> None:
        # this also functions as test_makeDefaultLatissButler(), but we may as
        # well catch the butler once it's made so it can be reused if needed,
        # given how hard it is to made it robustly

        # butler stuff
        try:
            if getSite() == "jenkins":
                raise unittest.SkipTest("Skip running butler-driven tests in Jenkins.")
            butler = makeDefaultLatissButler()
        except FileNotFoundError:
            raise unittest.SkipTest("Skipping tests that require the LATISS butler repo.")
        # fillDataId() and getLatissOnSkyDataIds() use DirectButler-only APIs,
        # so pin the concrete type rather than fail deep inside them.
        assert isinstance(butler, DirectButler)
        self.butler = butler

        # dict-like dataIds
        self.rawDataId = getMostRecentDataId(self.butler)
        self.fullId = fillDataId(self.butler, self.rawDataId)
        self.assertIn("exposure", self.fullId)
        self.assertIn("day_obs", self.fullId)
        self.assertIn("seq_num", self.fullId)
        self.expIdOnly = {"exposure": self.fullId["exposure"], "detector": 0}
        self.dayObsSeqNumIdOnly = {
            "day_obs": getDayObs(self.fullId),
            "seq_num": getSeqNum(self.fullId),
            "detector": 0,
        }

        # expRecords
        self.expRecordNoDetector = getExpRecordFromDataId(self.butler, self.rawDataId)
        self.assertIsInstance(self.expRecordNoDetector, dafButler.DimensionRecord)
        self.assertFalse(hasattr(self.expRecordNoDetector, "detector"))
        self.assertFalse("detector" in self.expRecordNoDetector.dataId)
        # just a crosscheck on the above to make sure other things are correct
        self.assertTrue(hasattr(self.expRecordNoDetector, "instrument"))

        # data coordinates
        # popping here because butler.registry.expandDataId cannot have
        # day_obs or seq_num present right now
        rawDataIdNoDayObSeqNum = _assureDict(self.rawDataId)
        if dayObsKey := _get_dayobs_key(rawDataIdNoDayObSeqNum):
            rawDataIdNoDayObSeqNum.pop(dayObsKey)
        if seqNumKey := _get_seqnum_key(rawDataIdNoDayObSeqNum):
            rawDataIdNoDayObSeqNum.pop(seqNumKey)
        self.rawDataIdNoDayObSeqNum = rawDataIdNoDayObSeqNum
        self.dataCoordMinimal = self.butler.registry.expandDataId(self.rawDataIdNoDayObSeqNum, detector=0)
        self.assertIsInstance(self.dataCoordMinimal, dafButler.DataCoordinate)

    def test_getLatissDefaultCollections(self) -> None:
        defaultCollections = getLatissDefaultCollections()
        self.assertTrue(defaultCollections is not None)
        self.assertTrue(defaultCollections != [])
        self.assertTrue(len(defaultCollections) >= 1)

    def test_RECENT_DAY(self) -> None:
        todayInt = int(datetime.date.today().strftime("%Y%m%d"))
        self.assertTrue(RECENT_DAY <= todayInt)  # in the past
        self.assertTrue(RECENT_DAY >= 20200101)  # not too far in the past

        # check that the value of RECENT_DAY is before the end of the data.
        daysWithData = getDaysWithData(self.butler)
        self.assertLessEqual(RECENT_DAY, max(daysWithData))

        # no test here, but print a warning if it hasn't been updated recently
        recentDay_datetime = datetime.datetime.strptime(str(RECENT_DAY), "%Y%m%d")
        now = datetime.datetime.today()
        timeSinceUpdate = now - recentDay_datetime
        if timeSinceUpdate.days > 100:  # TODO:
            print(
                f"RECENT_DAY is now {timeSinceUpdate.days} days in the past. "
                "You might want to consider updating this to speed up butler queries."
            )

    def test_sanitizeDayObs(self) -> None:
        dayObs: str | int = "2020-01-02"
        self.assertEqual(sanitizeDayObs(dayObs), 20200102)
        dayObs = 20210201
        self.assertEqual(sanitizeDayObs(dayObs), dayObs)

        with self.assertRaises(ValueError):
            sanitizeDayObs(1.234)  # type: ignore[arg-type]
            sanitizeDayObs("Febuary 29th, 1970")

    def test_getMostRecentDayObs(self) -> None:
        # just a basic sanity check here as we can't know the value,
        # but at least check something is returned, and is plausible
        recentDay = getMostRecentDayObs(self.butler)
        self.assertIsInstance(recentDay, int)
        self.assertTrue(recentDay >= RECENT_DAY)
        # some test data might be set a millennium in the future, i.e.
        # the year wouldd be 2XXX+1000, so set to y4k just in case
        self.assertTrue(recentDay < 40000000)

    def test_getSeqNumsForDayObs(self) -> None:
        emptyDay = 19990101
        seqnums = getSeqNumsForDayObs(self.butler, emptyDay)
        self.assertIsInstance(seqnums, Iterable)
        self.assertEqual(len(list(seqnums)), 0)

        recentDay = getMostRecentDayObs(self.butler)
        seqnums = getSeqNumsForDayObs(self.butler, recentDay)
        self.assertIsInstance(seqnums, Iterable)
        self.assertTrue(len(list(seqnums)) >= 1)

    def test_getMostRecentDataId(self) -> None:
        # we can't know the values, but it should always return something
        # and the dict and int forms should always have certain keys and agree
        dataId = getMostRecentDataId(self.butler)
        self.assertIsInstance(dataId, dict)
        self.assertIn("day_obs", dataId)
        self.assertIn("seq_num", dataId)
        self.assertTrue("exposure" in dataId or "exposure.id" in dataId)

    def test_getDatasetRefForDataId(self) -> None:
        dRef = getDatasetRefForDataId(self.butler, "raw", self.rawDataId)
        self.assertIsInstance(dRef, DatasetRef)

        dRef = getDatasetRefForDataId(self.butler, "raw", self.rawDataIdNoDayObSeqNum)
        self.assertIsInstance(dRef, DatasetRef)
        dRef = getDatasetRefForDataId(self.butler, "raw", self.dataCoordMinimal)
        self.assertIsInstance(dRef, DatasetRef)

    def test__dayobs_present(self) -> None:
        goods = [{"day_obs": 123}, {"exposure.day_obs": 234}, {"day_obs": 345, "otherkey": -1}]
        bads = [{"different_key": 123}]
        for good in goods:
            self.assertTrue(_dayobs_present(good))
        for bad in bads:
            self.assertFalse(_dayobs_present(bad))

    def test__seqnum_present(self) -> None:
        goods = [{"seq_num": 123}, {"exposure.seq_num": 234}, {"seq_num": 345, "otherkey": -1}]
        bads = [{"different_key": 123}]
        for good in goods:
            self.assertTrue(_seqnum_present(good))
        for bad in bads:
            self.assertFalse(_seqnum_present(bad))

    def test__expid_present(self) -> None:
        goods = [{"exposure": 123}, {"exposure.id": 234}, {"exposure.id": 345, "otherkey": -1}]
        bads = [{"different_key": 123}]
        for good in goods:
            self.assertTrue(_expid_present(good))
        for bad in bads:
            self.assertFalse(_expid_present(bad))

    def test_getDayObs(self) -> None:
        dayVal = 98765
        goods = [{"day_obs": dayVal}, {"exposure.day_obs": dayVal}, {"day_obs": dayVal, "otherkey": -1}]
        bads = [{"different_key": 123}]
        for good in goods:
            self.assertTrue(getDayObs(good) == dayVal)
        for bad in bads:
            self.assertTrue(getDayObs(bad) is None)

    def test_getSeqNum(self) -> None:
        seqVal = 12345
        goods = [{"seq_num": seqVal}, {"exposure.seq_num": seqVal}, {"seq_num": seqVal, "otherkey": -1}]
        bads = [{"different_key": 123}]
        for good in goods:
            self.assertTrue(getSeqNum(good) == seqVal)
        for bad in bads:
            self.assertTrue(getSeqNum(bad) is None)

    def test_getExpId(self) -> None:
        expIdVal = 12345
        goods = [{"exposure": expIdVal}, {"exposure.id": expIdVal}, {"exposure": expIdVal, "otherkey": -1}]
        bads = [{"different_key": 123}]
        for good in goods:
            self.assertTrue(getExpId(good) == expIdVal)
        for bad in bads:
            self.assertTrue(getExpId(bad) is None)

    def test_datasetExists(self) -> None:
        self.assertTrue(self.butler.exists("raw", self.rawDataId))
        self.assertTrue(self.butler.exists("raw", self.expIdOnly))
        self.assertTrue(self.butler.exists("raw", self.dayObsSeqNumIdOnly))
        return

    def test_sortRecordsByDayObsThenSeqNum(self) -> None:
        where = "exposure.day_obs=dayObs"
        recordResults = self.butler.registry.queryDimensionRecords(
            "exposure", where=where, bind={"dayObs": RECENT_DAY}
        )
        expRecords = list(recordResults)
        self.assertGreaterEqual(len(expRecords), 1)  # just ensure we're not doing a no-op test
        random.shuffle(expRecords)  # they are often already in order, so make sure they're not
        sortedIds = sortRecordsByDayObsThenSeqNum(expRecords)
        for i, _id in enumerate(sortedIds[:-1]):
            self.assertTrue(_id.seq_num < sortedIds[i + 1].seq_num)

        # Check that ambiguous sorts raise as expected
        with self.assertRaises(ValueError):
            recordResults = self.butler.registry.queryDimensionRecords(
                "exposure", where=where, bind={"dayObs": RECENT_DAY}
            )
            expRecords = list(recordResults)
            self.assertGreaterEqual(len(expRecords), 1)  # just ensure we're not doing a no-op test
            expRecords.append(expRecords[0])  # add a duplicate
            sortedIds = sortRecordsByDayObsThenSeqNum(expRecords)
        return

    def test_getDaysWithData(self) -> None:
        days = getDaysWithData(self.butler)
        self.assertTrue(len(days) >= 0)
        self.assertIsInstance(days[0], int)
        return

    def test_getExpIdFromDayObsSeqNum(self) -> None:
        expId = getExpIdFromDayObsSeqNum(self.butler, self.dayObsSeqNumIdOnly)
        self.assertTrue(_expid_present(expId))
        return

    def test_updateDataIdOrDataCord(self) -> None:
        updateVals = {"testKey": "testValue"}

        ids: list[dafButler.DataId | DimensionRecord] = [
            self.rawDataId,
            self.expRecordNoDetector,
            self.dataCoordMinimal,
        ]
        for originalId in ids:
            testId = updateDataIdOrDataCord(originalId, **updateVals)
            for k, v in updateVals.items():
                self.assertTrue(testId[k] == v)
        return

    def test_fillDataId(self) -> None:
        self.assertFalse(_dayobs_present(self.expIdOnly))
        self.assertFalse(_seqnum_present(self.expIdOnly))

        fullId = fillDataId(self.butler, self.expIdOnly)
        self.assertTrue(_dayobs_present(fullId))
        self.assertTrue(_seqnum_present(fullId))

        ids: list[dafButler.DataId | DimensionRecord] = [
            self.rawDataId,
            self.expRecordNoDetector,
            self.dataCoordMinimal,
        ]
        for dataId in ids:
            fullId = fillDataId(self.butler, dataId)
            self.assertTrue(_dayobs_present(fullId))
            self.assertTrue(_seqnum_present(fullId))
            self.assertTrue(_expid_present(fullId))
        return

    def test_getExpRecordFromDataId(self) -> None:
        record = getExpRecordFromDataId(self.butler, self.rawDataId)
        self.assertIsInstance(record, dafButler.DimensionRecord)
        return

    def test_getDayObsSeqNumFromExposureId(self) -> None:
        dayObsSeqNum = getDayObsSeqNumFromExposureId(self.butler, self.expIdOnly)
        self.assertTrue(_dayobs_present(dayObsSeqNum))
        self.assertTrue(_seqnum_present(dayObsSeqNum))
        return

    def test_removeDataProduct(self) -> None:
        # Can't think of an easy or safe test for this
        return

    def test_getLatissOnSkyDataIds(self) -> None:
        # This is very slow, consider removing as it's the least import of all
        # the util functions. However, restricting it to only the most recent
        # day does help a lot, so probably OK like that, and should speed up
        # with middleware improvements in the future, and we should ensure
        # that they don't break this, so inclined to leave for now
        dayToUse = getDaysWithData(self.butler)[-1]
        # the most recent day with data might only be biases or flats so make
        # sure to override the default of skipping biases, darks & flats
        skipTypes = ()
        ids = getLatissOnSkyDataIds(self.butler, skipTypes=skipTypes, startDate=dayToUse, endDate=dayToUse)
        self.assertTrue(len(ids) > 0)
        self.assertTrue(ids[0] is not None)

        ids = getLatissOnSkyDataIds(
            self.butler, skipTypes=skipTypes, startDate=dayToUse, endDate=dayToUse, full=True
        )
        self.assertTrue(len(ids) > 0)
        self.assertTrue(ids[0] is not None)
        testId = ids[0]
        self.assertTrue(_dayobs_present(testId))
        self.assertTrue(_seqnum_present(testId))
        self.assertTrue(_expid_present(testId))
        return

    def test__assureDict(self) -> None:
        items: list[dafButler.DataId | DimensionRecord] = [
            self.rawDataId,
            self.fullId,
            self.expIdOnly,
            self.expRecordNoDetector,
            self.dataCoordMinimal,
            self.rawDataIdNoDayObSeqNum,
        ]
        for item in items:
            testId = _assureDict(item)
            self.assertIsInstance(testId, dict)
        return

    def test__get_dayobs_key(self) -> None:
        dataId = {"a_random_key": 321, "exposure.day_obs": 20200312, "z_random_key": "abc"}
        self.assertTrue(_get_dayobs_key(dataId) == "exposure.day_obs")
        dataId = {"day_obs": 20200312}
        self.assertTrue(_get_dayobs_key(dataId) == "day_obs")
        dataId = {"missing": 20200312}
        self.assertTrue(_get_dayobs_key(dataId) is None)
        return

    def test__get_seqnum_key(self) -> None:
        dataId = {"a_random_key": 321, "exposure.seq_num": 123, "z_random_key": "abc"}
        self.assertTrue(_get_seqnum_key(dataId) == "exposure.seq_num")
        dataId = {"seq_num": 123}
        self.assertTrue(_get_seqnum_key(dataId) == "seq_num")
        dataId = {"missing": 123}
        self.assertTrue(_get_seqnum_key(dataId) is None)
        return

    def test__get_expid_key(self) -> None:
        dataId = {"a_random_key": 321, "exposure.id": 123, "z_random_key": "abc"}
        self.assertTrue(_get_expid_key(dataId) == "exposure.id")
        dataId = {"a_random_key": 321, "exposure": 123, "z_random_key": "abc"}
        self.assertTrue(_get_expid_key(dataId) == "exposure")
        dataId = {"missing": 123}
        self.assertTrue(_get_expid_key(dataId) is None)
        return

    def test_updateDataId(self) -> None:
        # check with a dataCoordinate
        dataId: Any = copy.copy(self.expRecordNoDetector.dataId)
        self.assertTrue("detector" not in dataId)
        dataId = updateDataId(dataId, detector=123)
        self.assertTrue("detector" in dataId)
        self.assertEqual(dataId["detector"], 123)

        # check with a dict
        self.assertIsInstance(self.rawDataId, dict)
        dataId = copy.copy(self.rawDataId)
        dataId.pop("detector")
        self.assertTrue("detector" not in dataId)
        dataId = updateDataId(dataId, detector=321)
        self.assertTrue("detector" in dataId)
        self.assertEqual(dataId["detector"], 321)

    def test_getExpRecord(self) -> None:
        expId = self.expIdOnly["exposure"]
        dayObs = self.dayObsSeqNumIdOnly["day_obs"]
        seqNum = self.dayObsSeqNumIdOnly["seq_num"]
        assert seqNum is not None

        recordByExpId = getExpRecord(self.butler, "LATISS", expId=expId)
        self.assertIsInstance(recordByExpId, dafButler.DimensionRecord)

        recordByDayObsSeqNum = getExpRecord(self.butler, "LATISS", dayObs=dayObs, seqNum=seqNum)
        self.assertIsInstance(recordByDayObsSeqNum, dafButler.DimensionRecord)
        self.assertEqual(recordByExpId, recordByDayObsSeqNum)

        with self.assertRaises(ValueError):
            # because we need dayObs too, so immediate raise due to bad args
            _ = getExpRecord(self.butler, "LATISS", seqNum=seqNum)

        with self.assertRaises(RuntimeError):
            # (dayObs, seqNum) no longer matches the expId, so there are no
            # results, which is a RuntimeError
            _ = getExpRecord(self.butler, "LATISS", expId=expId, dayObs=dayObs, seqNum=seqNum + 1)


class ButlerInitTestCase(lsst.utils.tests.TestCase):
    """Separately test whether we can make a butler with the env var set
    and that the expected error type is raised and passed through when it is
    not, as this is relied upon to correctly skip tests when butler init is
    not possible.
    """

    def test_dafButlerRaiseTypes(self) -> None:
        # If DAF_BUTLER_REPOSITORY_INDEX is not set *at all* then
        # using an instrument label raises a FileNotFoundError
        with unittest.mock.patch.dict("os.environ"):
            if "DAF_BUTLER_REPOSITORY_INDEX" in os.environ:  # can't del unless it's already there
                del os.environ["DAF_BUTLER_REPOSITORY_INDEX"]
            with self.assertRaises(FileNotFoundError):
                dafButler.Butler("LATISS")  # type: ignore[abstract]

        # If DAF_BUTLER_REPOSITORY_INDEX is present but is just an empty
        # string then using a label raises a RuntimeError
        with unittest.mock.patch.dict(os.environ, {"DAF_BUTLER_REPOSITORY_INDEX": ""}):
            with self.assertRaises(FileNotFoundError):
                dafButler.Butler("LATISS")  # type: ignore[abstract]

        # If DAF_BUTLER_REPOSITORY_INDEX _is_ set, we can't rely on any given
        # camera existing, but we can check that we get the expected error
        # when trying to init an instrument which definitely won't be defined.
        if os.getenv("DAF_BUTLER_REPOSITORY_INDEX"):
            with self.assertRaises(FileNotFoundError):
                dafButler.Butler("NotAValidCameraName")  # type: ignore[abstract]

    def test_makeDefaultLatissButlerRaiseTypes(self) -> None:
        """makeDefaultLatissButler unifies the mixed exception types from
        butler inits, so test all available possibilities here.
        """
        if getSite() == "jenkins":
            raise unittest.SkipTest("Skip running butler-driven tests in Jenkins.")
        with unittest.mock.patch.dict("os.environ"):
            if "DAF_BUTLER_REPOSITORY_INDEX" in os.environ:  # can't del unless it's already there
                del os.environ["DAF_BUTLER_REPOSITORY_INDEX"]
            with self.assertRaises(FileNotFoundError):
                makeDefaultLatissButler()

        with unittest.mock.patch.dict(os.environ, {"DAF_BUTLER_REPOSITORY_INDEX": ""}):
            with self.assertRaises(FileNotFoundError):
                makeDefaultLatissButler()

        fakeFile = "/path/to/a/file/which/does/not_exist.yaml"
        with unittest.mock.patch.dict(os.environ, {"DAF_BUTLER_REPOSITORY_INDEX": fakeFile}):
            with self.assertRaises(FileNotFoundError):
                makeDefaultLatissButler()

    def test_DAF_BUTLER_REPOSITORY_INDEX_value(self) -> None:
        # If DAF_BUTLER_REPOSITORY_INDEX is truthy then we expect it to point
        # to an actual file
        repoFile = os.getenv("DAF_BUTLER_REPOSITORY_INDEX")
        if repoFile:
            self.assertTrue(ResourcePath(repoFile).exists())


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: object) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
