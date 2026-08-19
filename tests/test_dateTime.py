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

"""Test cases for utils."""

import datetime
import unittest

import astropy
from astropy.time import Time

import lsst.utils.tests
from lsst.summit.utils.dateTime import (  # TODO: add tests for efdTimestampToAstropy, astropyToEfdTimestamp
    calcDayOffset,
    getDayObsEndTime,
    getDayObsForTime,
    getDayObsStartTime,
)


class DateTimeTestCase(lsst.utils.tests.TestCase):

    def test_getDayObsAsTimes(self) -> None:
        """This tests getDayObsStartTime and getDayObsEndTime explicitly,
        but the days we loop over are chosen to test calcNextDay() which is
        called by getDayObsEndTime().
        """
        for dayObs in (
            20200228,  # day before end of Feb on a leap year
            20200229,  # end of Feb on a leap year
            20210227,  # day before end of Feb on a non-leap year
            20200228,  # end of Feb on a non-leap year
            20200430,  # end of a month with 30 days
            20200530,  # end of a month with 31 days
            20201231,  # year rollover
        ):
            dayStart = getDayObsStartTime(dayObs)
            self.assertIsInstance(dayStart, astropy.time.Time)

            dayEnd = getDayObsEndTime(dayObs)
            self.assertIsInstance(dayStart, astropy.time.Time)

            self.assertGreater(dayEnd, dayStart)
            self.assertEqual(dayEnd.jd, dayStart.jd + 1)

    def test_calcDayOffset(self) -> None:
        self.assertEqual(calcDayOffset(20230531, 20230601), 1)
        self.assertEqual(calcDayOffset(20230531, 20230530), -1)
        self.assertEqual(calcDayOffset(20230531, 20230531), 0)
        self.assertEqual(calcDayOffset(20231231, 20240101), 1)
        self.assertEqual(calcDayOffset(20240228, 20240301), 2)  # leap year
        self.assertEqual(calcDayOffset(20240229, 20240301), 1)  # leap year
        self.assertEqual(calcDayOffset(20240228, 20240302), 3)  # leap year

        self.assertNotEqual(calcDayOffset(20230531, 20230601), -1)

    def test_getDayObsForTime(self) -> None:
        pydate = datetime.datetime(2023, 2, 5, 13, 30, 1)
        time = Time(pydate)
        dayObs = getDayObsForTime(time)
        self.assertEqual(dayObs, 20230205)

        pydate = datetime.datetime(2023, 2, 5, 11, 30, 1)
        time = Time(pydate)
        dayObs = getDayObsForTime(time)
        self.assertEqual(dayObs, 20230204)
        return


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: object) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
