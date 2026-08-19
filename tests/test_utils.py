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

import copy
import datetime
import itertools
import unittest
from typing import Iterable

import astropy.time
import astropy.units as u
import numpy as np
from astro_metadata_translator import makeObservationInfo
from astropy.coordinates import HeliocentricEclipticIAU76, SkyCoord
from astropy.time import Time

import lsst.afw.detection as afwDetect
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.geom as geom
import lsst.utils.tests
from lsst.obs.base import createInitialSkyWcsFromBoresight
from lsst.obs.base.makeRawVisitInfoViaObsInfo import MakeRawVisitInfoViaObsInfo
from lsst.obs.lsst import Latiss
from lsst.obs.lsst.translators.latiss import AUXTEL_LOCATION
from lsst.summit.utils.dateTime import getCurrentDayObsDatetime, getCurrentDayObsHumanStr, getCurrentDayObsInt
from lsst.summit.utils.utils import getFilterSeeingCorrection  # deprecated
from lsst.summit.utils.utils import (
    calcEclipticCoords,
    computeCcdExposureId,
    computeExposureId,
    fluxesFromFootprints,
    getAirmassSeeingCorrection,
    getBandpassSeeingCorrection,
    getExpPositionOffset,
    getFieldNameAndTileNumber,
    getQuantiles,
    getSunAngle,
    quickSmooth,
)


class ExpSkyPositionOffsetTestCase(lsst.utils.tests.TestCase):
    """A test case for testing sky position offsets for exposures."""

    def setUp(self) -> None:
        camera = Latiss.getCamera()
        self.assertTrue(len(camera) == 1)
        self.detector = camera[0]

        self.viMaker = MakeRawVisitInfoViaObsInfo()
        self.mi = afwImage.MaskedImageF(0, 0)
        self.baseHeader = dict(
            boresight_airmass=1.5,
            temperature=15 * u.deg_C,
            observation_type="science",
            exposure_time=5 * u.ks,
            detector_num=32,
            location=AUXTEL_LOCATION,
        )

    def test_getExpPositionOffset(self) -> None:
        epsilon = 0.0001
        ra1s = [0, 45, 90]
        ra2s = copy.copy(ra1s)
        ra2s.extend([r + epsilon for r in ra1s])
        ra1s = np.deg2rad(ra1s)
        ra2s = np.deg2rad(ra2s)

        epsilon = 0.0001
        dec1s = [0, 45, 90]
        dec2s = copy.copy(dec1s)
        dec2s.extend([d + epsilon for d in dec1s[:-1]])  # skip last point as >90 not allowed for dec

        rotAngle1 = geom.Angle(43.2, geom.degrees)  # arbitrary non-zero
        rotAngle2 = geom.Angle(56.7, geom.degrees)

        t1 = astropy.time.Time("2021-09-15T12:00:00", format="isot", scale="utc")
        t2 = astropy.time.Time("2021-09-15T12:01:00", format="isot", scale="utc")
        expTime = astropy.time.TimeDelta(20, format="sec")

        header1 = copy.copy(self.baseHeader)
        header2 = copy.copy(self.baseHeader)
        header1["datetime_begin"] = astropy.time.Time(t1, format="isot", scale="utc")
        header2["datetime_begin"] = astropy.time.Time(t2, format="isot", scale="utc")

        header1["datetime_end"] = astropy.time.Time(t1 + expTime, format="isot", scale="utc")
        header2["datetime_end"] = astropy.time.Time(t2 + expTime, format="isot", scale="utc")

        obsInfo1 = makeObservationInfo(**header1)
        obsInfo2 = makeObservationInfo(**header2)

        vi1 = self.viMaker.observationInfo2visitInfo(obsInfo1)
        vi2 = self.viMaker.observationInfo2visitInfo(obsInfo2)
        expInfo1 = afwImage.ExposureInfo()
        expInfo1.setVisitInfo(vi1)
        expInfo2 = afwImage.ExposureInfo()
        expInfo2.setVisitInfo(vi2)

        for ra1, dec1, ra2, dec2 in itertools.product(ra1s, dec1s, ra2s, dec2s):
            pos1 = geom.SpherePoint(ra1, dec1, geom.degrees)
            pos2 = geom.SpherePoint(ra2, dec2, geom.degrees)

            wcs1 = createInitialSkyWcsFromBoresight(pos1, rotAngle1, self.detector, flipX=True)
            wcs2 = createInitialSkyWcsFromBoresight(pos2, rotAngle2, self.detector, flipX=True)

            exp1 = afwImage.ExposureF(self.mi, expInfo1)
            exp2 = afwImage.ExposureF(self.mi, expInfo2)

            exp1.setWcs(wcs1)
            exp2.setWcs(wcs2)

            result = getExpPositionOffset(exp1, exp2)

            deltaRa = ra1 - ra2
            deltaDec = dec1 - dec2

            self.assertAlmostEqual(result.deltaRa.asDegrees(), deltaRa, 6)
            self.assertAlmostEqual(result.deltaDec.asDegrees(), deltaDec, 6)


class MiscUtilsTestCase(lsst.utils.tests.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_getFieldNameAndTileNumber(self) -> None:
        field, num = getFieldNameAndTileNumber("simple")
        self.assertEqual(field, "simple")
        self.assertIsNone(num)

        field, num = getFieldNameAndTileNumber("_simple")
        self.assertEqual(field, "_simple")
        self.assertIsNone(num)

        field, num = getFieldNameAndTileNumber("simple_321")
        self.assertEqual(field, "simple")
        self.assertEqual(num, 321)

        field, num = getFieldNameAndTileNumber("_simple_321")
        self.assertEqual(field, "_simple")
        self.assertEqual(num, 321)

        field, num = getFieldNameAndTileNumber("test_321a_123")
        self.assertEqual(field, "test_321a")
        self.assertEqual(num, 123)

        field, num = getFieldNameAndTileNumber("test_321a_123_")
        self.assertEqual(field, "test_321a_123_")
        self.assertIsNone(num)

        field, num = getFieldNameAndTileNumber("test_321a_123a")
        self.assertEqual(field, "test_321a_123a")
        self.assertIsNone(num)

        field, num = getFieldNameAndTileNumber("test_321a:asd_asd-dsa_321")
        self.assertEqual(field, "test_321a:asd_asd-dsa")
        self.assertEqual(num, 321)

    def test_getAirmassSeeingCorrection(self) -> None:
        for airmass in (1.1, 2.0, 20.0):
            correction = getAirmassSeeingCorrection(airmass)
            self.assertGreater(correction, 0.01)
            self.assertLess(correction, 1.0)

        correction = getAirmassSeeingCorrection(1)
        self.assertEqual(correction, 1.0)

        with self.assertRaises(ValueError):
            getAirmassSeeingCorrection(0.5)

    def test_getFilterSeeingCorrection(self) -> None:
        for filterName in ("SDSSg_65mm", "SDSSr_65mm", "SDSSi_65mm"):
            with self.assertWarns(FutureWarning):
                correction = getFilterSeeingCorrection(filterName)
            self.assertGreater(correction, 0.5)
            self.assertLess(correction, 1.5)

    def test_getBandpassSeeingCorrection(self) -> None:
        for filterName in (
            "SDSSg_65mm",
            "SDSSr_65mm",
            "SDSSi_65mm",
            "u_02",
            "g_01",
            "r_03",
            "i_06",
            "z_03",
            "y_04",
        ):
            correction = getBandpassSeeingCorrection(filterName)
            self.assertGreater(correction, 0.5)
            self.assertLess(correction, 1.5)

    def test_quickSmooth(self) -> None:
        # just test that it runs and returns the right shape. It's a wrapper on
        # scipy.ndimage.gaussian_filter we can trust that it does what it
        # should, and we just test the interface hasn't bitrotted on either end
        data = np.zeros((100, 100), dtype=np.float32)
        data = quickSmooth(data, 5.0)
        self.assertEqual(data.shape, (100, 100))

    def test_getCurrentDayObsDatetime(self) -> None:
        """Just a type check and a basic sanity check on the range.

        Setting days=3 as the tolerance just because of timezones and who knows
        what really.
        """
        dt = getCurrentDayObsDatetime()
        self.assertIsInstance(dt, datetime.date)
        self.assertLess(dt, datetime.date.today() + datetime.timedelta(days=3))
        self.assertGreater(dt, datetime.date.today() - datetime.timedelta(days=3))

    def test_getCurrentDayObsInt(self) -> None:
        """Just a type check and a basic sanity check on the range."""
        dayObs = getCurrentDayObsInt()
        self.assertIsInstance(dayObs, int)
        self.assertLess(dayObs, 21000101)
        self.assertGreater(dayObs, 19700101)

    def test_getCurrentDayObsHumanStr(self) -> None:
        """Just a basic formatting check."""
        dateStr = getCurrentDayObsHumanStr()
        self.assertIsInstance(dateStr, str)
        self.assertEqual(len(dateStr), 10)
        self.assertRegex(dateStr, r"\d{4}-\d{2}-\d{2}")

    def test_getSunAngle(self) -> None:
        """Just a basic sanity check on the range."""
        testTime = Time("2021-09-15T16:00:00", format="isot", scale="utc")
        sunAngle = getSunAngle(testTime)
        self.assertIsInstance(sunAngle, float)
        self.assertLess(sunAngle, 90.01)
        self.assertGreater(sunAngle, -90.01)

        sunAngle = getSunAngle()
        self.assertIsInstance(sunAngle, float)
        self.assertLess(sunAngle, 90.01)
        self.assertGreater(sunAngle, -90.01)


class QuantileTestCase(lsst.utils.tests.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_quantiles(self) -> None:
        # We understand that our algorithm gives very large rounding error
        # compared to the generic numpy method. But still test it.
        np.random.seed(1234)
        dataRanges = [(50, 1, -1), (100_000, 5_000, -2), (5_000_000, 10_000, -2), (50_000, 100_000, -3)]
        colorRanges = [2, 256, 999]  # [very few, nominal, lots and an odd number]
        for nColors, (mean, width, decimal) in itertools.product(colorRanges, dataRanges):
            data = np.random.normal(mean, width, (100, 100))
            data[10, 10] = np.nan  # check we're still nan-safe
            if np.nanmax(data) - np.nanmin(data) > 300_000:
                with self.assertLogs(level="WARNING") as cm:
                    edges1 = getQuantiles(data, nColors)
                self.assertIn("Data range is very large", cm.output[0])
            else:
                with self.assertNoLogs(level="WARNING") as cm:
                    edges1 = getQuantiles(data, nColors)
            edges2 = np.nanquantile(data, np.linspace(0, 1, nColors + 1))  # must check with nanquantile
            np.testing.assert_almost_equal(edges1, edges2, decimal=decimal)


class ImageBasedTestCase(lsst.utils.tests.TestCase):
    def test_fluxFromFootprint(self) -> None:
        image = afwImage.Image(
            np.arange(8100, dtype=np.int32).reshape(90, 90), xy0=lsst.geom.Point2I(10, 12), dtype="I"
        )

        radius = 3
        spans = afwGeom.SpanSet.fromShape(radius, afwGeom.Stencil.CIRCLE, offset=(27, 30))
        footprint1 = afwDetect.Footprint(spans)

        # The extracted footprint should be the same as the product of the
        # spans and the overlapped bow with the image
        truth1 = spans.asArray() * image.array[15:22, 14:21]

        radius = 3
        spans = afwGeom.SpanSet.fromShape(radius, afwGeom.Stencil.CIRCLE, offset=(44, 49))
        footprint2 = afwDetect.Footprint(spans)
        truth2 = spans.asArray() * image.array[34:41, 31:38]

        allFootprints = [footprint1, footprint2]
        footprintSet = afwDetect.FootprintSet(image.getBBox())
        footprintSet.setFootprints(allFootprints)

        # check it can accept a footprintSet, and single and iterables of
        # footprints
        with self.assertRaises(TypeError):
            fluxesFromFootprints(10, image)

        with self.assertRaises(TypeError):
            fluxesFromFootprints([8, 6, 7, 5, 3, 0, 9], image)

        # check the footPrintSet
        fluxes = fluxesFromFootprints(footprintSet, image)
        expectedLength = len(footprintSet.getFootprints())
        self.assertEqual(len(fluxes), expectedLength)  # always one flux per footprint
        self.assertIsInstance(fluxes, Iterable)
        self.assertAlmostEqual(fluxes[0], np.sum(truth1))
        self.assertAlmostEqual(fluxes[1], np.sum(truth2))

        # check the list of footprints
        fluxes = fluxesFromFootprints(allFootprints, image)
        expectedLength = 2
        self.assertEqual(len(fluxes), expectedLength)  # always one flux per footprint
        self.assertIsInstance(fluxes, Iterable)
        self.assertAlmostEqual(fluxes[0], np.sum(truth1))
        self.assertAlmostEqual(fluxes[1], np.sum(truth2))

        # ensure that subtracting the image median from fluxes leave image
        # pixels untouched
        oldImageArray = copy.deepcopy(image.array)
        fluxes = fluxesFromFootprints(footprintSet, image, subtractImageMedian=True)
        np.testing.assert_array_equal(image.array, oldImageArray)


class IdTestCase(lsst.utils.tests.TestCase):
    def test_exposure_id(self) -> None:
        self.assertEqual(computeExposureId("latiss", "O", 20240402, 35), 2024040200035)
        self.assertEqual(computeExposureId("LATISS", "C", 20240402, 35), 2024040200035)
        self.assertEqual(computeExposureId("LSSTComCamSim", "S", 20240402, 35), 7024040200035)
        with self.assertRaises(ValueError):
            computeExposureId("bad_instrument", "O", 20240402, 35)

    def test_ccdexposure_id(self) -> None:
        self.assertEqual(computeCcdExposureId("latiss", 2024040200035, 0), 5205 * (2**23) + 35 * 256 + 0)
        with self.assertRaises(ValueError):
            computeCcdExposureId("latiss", 20240402000035, 1)
        with self.assertRaises(ValueError):
            computeCcdExposureId("LsstComCam", 20240402000035, 9)
        with self.assertRaises(ValueError):
            computeCcdExposureId("LsstCam", 20240402000035, 205)
        self.assertEqual(
            computeCcdExposureId("LSSTComCamSim", 7024040200035, 2),
            5 * (2**37) + 5205 * (2**23) + 35 * 256 + 2,
        )

    def test_calcEclipticCoords(self) -> None:
        ras = [0, 1, 45, 90, 180, 270, 359.9]
        decs = [-90, -80, -45, -1, 0, 1, 45, 90]

        for ra, dec in itertools.product(ras, decs):
            p = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=1 * u.au, frame="hcrs")
            eclCoords = p.transform_to(HeliocentricEclipticIAU76)
            _lambda, beta = calcEclipticCoords(ra, dec)
            self.assertAlmostEqual(eclCoords.lat.deg, beta, 6)
            self.assertAlmostEqual(eclCoords.lon.deg, _lambda, 6)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: object) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
