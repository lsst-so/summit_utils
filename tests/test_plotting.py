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

import lsst.afw.image as afwImage
import lsst.utils
import lsst.utils.tests
from lsst.afw.detection import Footprint
from lsst.afw.geom import SpanSet
from lsst.summit.utils.plotting import plot
from lsst.summit.utils.utils import detectObjectsInExp


class PlottingTestCase(lsst.utils.tests.TestCase):
    outputDir: str

    try:
        afwDataDir = lsst.utils.getPackageDir("afwdata")
    except Exception:
        afwDataDir = None
    filename = "postISRCCD_2020021800224-EMPTY~EMPTY-det000.fits.fz"

    @classmethod
    def setUpClass(cls) -> None:
        cls.outputDir = tempfile.mkdtemp()

    @unittest.skipUnless(afwDataDir, "afwdata not available")
    def test_plot(self) -> None:
        """Test that the the plot is made and saved"""
        fullName = os.path.join(self.afwDataDir, "LATISS/postISRCCD", self.filename)
        exp = afwImage.ExposureF(fullName)
        centroids = [(567, 746), (576, 599), (678, 989)]

        foot1 = Footprint(SpanSet.fromShape(5, offset=(690, 710)))
        foot2 = Footprint(SpanSet.fromShape(6, offset=(159, 216)))
        fpset = detectObjectsInExp(exp)

        # Input is an exposure
        outputFilename = os.path.join(self.outputDir, "testPlotting_exp.jpg")
        plot(exp, centroids=centroids, footprints=fpset, addLegend=True, savePlotAs=outputFilename)
        self.assertTrue(os.path.isfile(outputFilename))
        self.assertTrue(os.path.getsize(outputFilename) > 10000)

        # Input is an image
        outputFilename = os.path.join(self.outputDir, "testPlotting_image.jpg")
        im = exp.image
        plot(im, footprints=[foot1, foot2], savePlotAs=outputFilename)
        self.assertTrue(os.path.isfile(outputFilename))
        self.assertTrue(os.path.getsize(outputFilename) > 10000)

        # Input is a masked image
        outputFilename = os.path.join(self.outputDir, "testPloting_mask.jpg")
        masked = exp.maskedImage
        plot(masked, savePlotAs=outputFilename)
        self.assertTrue(os.path.isfile(outputFilename))
        self.assertTrue(os.path.getsize(outputFilename) > 10000)

        # Input is a numpy array
        outputFilename = os.path.join(self.outputDir, "testPlotting_nparr.jpg")
        nparr = exp.image.array
        plot(nparr, footprints=foot1, savePlotAs=outputFilename)
        self.assertTrue(os.path.isfile(outputFilename))
        self.assertTrue(os.path.getsize(outputFilename) > 10000)

        # Nans in the image
        nparr[1200:1250, 1300:1345] = np.nan
        for stretch in ["ccs", "asinh", "power", "log", "linear", "sqrt"]:
            plot(nparr, showCompass=False, stretch=stretch)

        # Image consists of nans
        nparr[:, :] = np.nan
        plot(nparr)


def setup_module(module: object) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
