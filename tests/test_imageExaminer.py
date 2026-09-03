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

import lsst.utils.tests
from lsst.daf.butler import Butler
from lsst.summit.utils import ImageExaminer
from lsst.summit.utils.bestEffort import BestEffortIsr
from lsst.summit.utils.butlerUtils import makeDefaultLatissButler
from lsst.summit.utils.utils import getSite


class ImageExaminerTestCase(lsst.utils.tests.TestCase):
    # class attributes populated in setUpClass
    butler: Butler
    dataId: dict[str, int]
    bestEffort: BestEffortIsr
    outputDir: str
    outputFilename: str

    @classmethod
    def setUpClass(cls) -> None:
        try:
            if getSite() == "jenkins":
                raise unittest.SkipTest("Skip running butler-driven tests in Jenkins.")
            cls.butler = makeDefaultLatissButler()
        except FileNotFoundError:
            raise unittest.SkipTest("Skipping tests that require the LATISS butler repo.")

        # Chosen to work on the TTS, summit and NCSA
        cls.dataId = {"day_obs": 20200315, "seq_num": 120, "detector": 0}
        cls.bestEffort = BestEffortIsr()  # should always succeed if makeDefaultLatissButler works
        cls.outputDir = tempfile.mkdtemp()
        cls.outputFilename = os.path.join(cls.outputDir, "testImageExaminer.jpg")

    def test_imageExaminer(self) -> None:
        """Test that the animator produces a large file without worrying about
        the contents?
        """
        exp = self.bestEffort.getExposure(self.dataId)
        imExam = ImageExaminer(
            exp, doTweakCentroid=True, boxHalfSize=105, doForceCoM=False, savePlots=self.outputFilename
        )
        imExam.plot()

        self.assertTrue(os.path.isfile(self.outputFilename))
        self.assertTrue(os.path.getsize(self.outputFilename) > 100000)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: object) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
