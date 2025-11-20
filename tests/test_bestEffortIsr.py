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

import unittest

import lsst.afw.image as afwImage
import lsst.summit.utils.butlerUtils as butlerUtils
import lsst.utils.tests
from lsst.summit.utils.bestEffort import BestEffortIsr
from lsst.summit.utils.quickLook import QuickLookIsrTask


class BestEffortIsrTestCase(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.bestEffortIsr = BestEffortIsr()
        except (FileNotFoundError, ValueError):
            raise unittest.SkipTest("Skipping tests that require the LATISS butler repo.")

        # chosen as this is available in the following locations - collections:
        # NCSA - LATISS/raw/all
        # TTS - LATISS-test-data-tts
        # summit - LATISS_test_data
        cls.dataId = {"day_obs": 20251021, "seq_num": 443, "detector": 0}

    def test_getExposure(self):
        # in most locations this will load a pre-made image
        exp = self.bestEffortIsr.getExposure(self.dataId)
        self.assertIsInstance(exp, afwImage.Exposure)

        # this will always actually run isr with whatever calibs are available
        exp = self.bestEffortIsr.getExposure(self.dataId, forceRemake=True)
        self.assertIsInstance(exp, afwImage.Exposure)

    def test_getExposureFromExpRecord(self):
        """Test getting with an expRecord. This requires also passing in
        the detector number as a kwarg.
        """
        expRecord = butlerUtils.getExpRecordFromDataId(self.bestEffortIsr.butler, self.dataId)

        exp = self.bestEffortIsr.getExposure(expRecord, detector=0)
        self.assertIsInstance(exp, afwImage.Exposure)

        # and then again with just the dataCoordinate
        exp = self.bestEffortIsr.getExposure(expRecord.dataId, detector=0)
        self.assertIsInstance(exp, afwImage.Exposure)

        # Try forceRemake with an expRecord and a detector as a kwarg
        # as forceRemake has a different code path, as it has to get a raw
        exp = self.bestEffortIsr.getExposure(expRecord.dataId, detector=0, forceRemake=True)
        self.assertIsInstance(exp, afwImage.Exposure)

    def test_raises(self):
        """Ensure function cannot be called without specifying a detector."""
        dataId = self.dataId
        dataId.pop("detector")
        with self.assertRaises(ValueError):
            self.bestEffortIsr.getExposure(dataId)

    def test_quicklook_connections(self):
        """Test that various QuickLookIsrConnections inputs are no longer
        required.
        """
        connections = QuickLookIsrTask.ConfigClass.ConnectionsClass(config=QuickLookIsrTask.ConfigClass())
        self.assertEqual(connections.bias.minimum, 0)
        self.assertEqual(connections.flat.minimum, 0)
        self.assertEqual(connections.ccdExposure.minimum, 1)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
