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

import contextlib
import tempfile
import unittest

import lsst.afw.cameraGeom.testUtils as afwTestUtils
import lsst.afw.image as afwImage
import lsst.daf.butler.tests as butlerTests
import lsst.ip.isr as ipIsr
import lsst.ip.isr.isrMockLSST as isrMock
import lsst.pex.exceptions
import lsst.pipe.base as pipeBase
import lsst.pipe.base.testUtils
from lsst.summit.utils.quickLook import QuickLookIsrTask, QuickLookIsrTaskConfig


class QuickLookIsrTaskTestCase(unittest.TestCase):
    """Tests of the run method with fake data."""

    def setUp(self):
        self.mockConfig = isrMock.IsrMockLSSTConfig()
        self.camera = isrMock.IsrMockLSST(config=self.mockConfig).getCamera()

        self.ccdExposure = isrMock.RawMockLSST(config=self.mockConfig).run()
        self.detector = self.ccdExposure.getDetector()
        amps = self.detector.getAmplifiers()
        ampNames = [amp.getName() for amp in amps]

        # # Mock other optional parameters
        self.bias = isrMock.BiasMockLSST(config=self.mockConfig).run()
        self.dark = isrMock.DarkMockLSST(config=self.mockConfig).run()
        self.flat = isrMock.FlatMockLSST(config=self.mockConfig).run()
        self.defects = isrMock.DefectMockLSST(config=self.mockConfig).run()
        self.ptc = ipIsr.PhotonTransferCurveDataset(ampNames=ampNames)  # Mock PTC dataset
        for amp, gain in self.mockConfig.gainDict.items():
            self.ptc.gain[amp] = 1.0
        self.bfKernel = isrMock.BfKernelMockLSST(config=self.mockConfig).run()
        self.task = QuickLookIsrTask(config=QuickLookIsrTaskConfig())

    def test_runQuickLook(self):
        # Execute the run method with the mock data
        result = self.task.run(
            self.ccdExposure,
            camera=self.camera,
            bias=self.bias,
            dark=self.dark,
            flat=self.flat,
            defects=self.defects,
            linearizer=None,
            crosstalk=None,
            bfKernel=self.bfKernel,
            ptc=self.ptc,
        )
        self.assertIsNotNone(result, "Result of run method should not be None")
        self.assertIsInstance(result, pipeBase.Struct, "Result should be of type lsst.pipe.base.Struct")
        self.assertIsInstance(
            result.exposure,
            afwImage.Exposure,
            "Resulting exposure should be an instance of lsst.afw.image.Exposure",
        )

    def test_runQuickLookMissingData(self):
        # Test without any inputs other than the exposure.  And the PTC.
        result = self.task.run(self.ccdExposure, ptc=self.ptc)
        self.assertIsInstance(result.exposure, afwImage.Exposure)

    def test_runQuickLookBadDark(self):
        # Test with an incorrect dark frame
        bbox = self.ccdExposure.getBBox()
        bbox.grow(-20)
        with self.assertRaises(lsst.pex.exceptions.wrappers.LengthError):
            self.task.run(
                self.ccdExposure,
                camera=self.camera,
                bias=self.bias,
                dark=self.dark[bbox],
                flat=self.flat,
                defects=self.defects,
            )


class QuickLookIsrTaskRunQuantumTests(lsst.utils.tests.TestCase):
    """Tests of ``QuickLookIsrTask.runQuantum``, which need a test butler,
    but do not need real images.

    Adapted from the unit tests of ``CalibrateImageTask.runQuantum``
    """

    def setUp(self):
        instrument = "testCam"
        exposureId = 100
        visit = 100101
        detector = 0
        physical_filter = "testCam_filter"
        band = "X"

        # Map the isrTask connection names to the names of the Butler dataset
        # inputs
        ccdExposure = "raw"
        camera = "camera"
        bias = "bias"
        dark = "dark"
        flat = "flat"
        defects = "defects"
        bfKernel = "bfk"
        ptc = "ptc"
        deferredChargeCalib = "cti"
        crosstalk = "crosstalk"
        linearizer = "linearizer"
        gainCorrection = "gain_correction"

        # outputs
        outputExposure = "postISRCCD"
        outputStatistics = "isrStatistics"

        # quickLook-only outputs
        exposure = "quickLookExp"

        # Create a and populate a test butler for runQuantum tests.
        self.repo_path = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.repo = butlerTests.makeTestRepo(self.repo_path.name)

        # dataIds for fake data
        butlerTests.addDataIdValue(self.repo, "instrument", instrument)
        butlerTests.addDataIdValue(self.repo, "physical_filter", physical_filter, band=band)
        butlerTests.addDataIdValue(self.repo, "detector", detector)
        butlerTests.addDataIdValue(self.repo, "exposure", exposureId, physical_filter=physical_filter)
        butlerTests.addDataIdValue(self.repo, "visit", visit)

        # inputs
        butlerTests.addDatasetType(self.repo, ccdExposure, {"instrument", "exposure", "detector"}, "Exposure")
        butlerTests.addDatasetType(self.repo, camera, {"instrument"}, "Camera")
        butlerTests.addDatasetType(self.repo, bias, {"instrument", "detector"}, "Exposure")
        butlerTests.addDatasetType(self.repo, dark, {"instrument", "detector"}, "Exposure")
        butlerTests.addDatasetType(self.repo, flat, {"instrument", "physical_filter", "detector"}, "Exposure")
        butlerTests.addDatasetType(self.repo, defects, {"instrument", "detector"}, "Defects")
        butlerTests.addDatasetType(self.repo, linearizer, {"instrument", "detector"}, "Linearizer")
        butlerTests.addDatasetType(self.repo, crosstalk, {"instrument", "detector"}, "CrosstalkCalib")
        butlerTests.addDatasetType(self.repo, bfKernel, {"instrument", "detector"}, "BrighterFatterKernel")
        butlerTests.addDatasetType(self.repo, ptc, {"instrument", "detector"}, "PhotonTransferCurveDataset")
        butlerTests.addDatasetType(self.repo, deferredChargeCalib, {"instrument", "detector"}, "IsrCalib")
        butlerTests.addDatasetType(self.repo, gainCorrection, {"instrument", "detector"}, "IsrCalib")

        # outputs
        butlerTests.addDatasetType(
            self.repo, outputExposure, {"instrument", "exposure", "detector"}, "Exposure"
        )
        butlerTests.addDatasetType(self.repo, exposure, {"instrument", "exposure", "detector"}, "Exposure")
        butlerTests.addDatasetType(self.repo, outputStatistics, {"instrument", "exposure", "detector"}, "StructuredDataDict")

        # dataIds
        self.exposure_id = self.repo.registry.expandDataId(
            {
                "instrument": instrument,
                "exposure": exposureId,
                "detector": detector,
                "physical_filter": physical_filter,
            }
        )
        self.instrument_id = self.repo.registry.expandDataId({"instrument": instrument})
        self.flat_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "physical_filter": physical_filter, "detector": detector}
        )
        self.detector_id = self.repo.registry.expandDataId({"instrument": instrument, "detector": detector})
        self.filter_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "physical_filter": physical_filter}
        )

        # put empty data
        self.butler = butlerTests.makeTestCollection(self.repo)
        self.butler.put(afwImage.ExposureF(), ccdExposure, self.exposure_id)
        self.butler.put(afwTestUtils.CameraWrapper().camera, camera, self.instrument_id)
        self.butler.put(afwImage.ExposureF(), bias, self.detector_id)
        self.butler.put(afwImage.ExposureF(), dark, self.detector_id)
        self.butler.put(afwImage.ExposureF(), flat, self.flat_id)
        self.butler.put(lsst.ip.isr.Defects(), defects, self.detector_id)
        self.butler.put(
            lsst.ip.isr.brighterFatterKernel.BrighterFatterKernel(), bfKernel, self.detector_id
        )
        self.butler.put(ipIsr.PhotonTransferCurveDataset(), ptc, self.detector_id)
        self.butler.put(lsst.ip.isr.calibType.IsrCalib(), deferredChargeCalib, self.detector_id)
        self.butler.put(lsst.ip.isr.crosstalk.CrosstalkCalib(), crosstalk, self.detector_id)
        self.butler.put(lsst.ip.isr.linearize.Linearizer(), linearizer, self.detector_id)
        self.butler.put(lsst.ip.isr.calibType.IsrCalib(), gainCorrection, self.detector_id)

    def tearDown(self):
        del self.repo_path  # this removes the temporary directory

    def test_runQuantum(self):
        config = ipIsr.IsrTaskConfig()
        # Remove some outputs
        config.doBinnedExposures = False
        config.doSaveInterpPixels = False
        config.qa.doThumbnailOss = False
        config.qa.doThumbnailFlattened = False
        config.doCalculateStatistics = False

        # Turn on all optional inputs
        config.doDeferredCharge = False  # There is no CTI calibration
                                         # available for LATISS.
        config.usePtcReadNoise = True
        config.doCrosstalk = True
        config.doBrighterFatter = True

        # Override a method in IsrTask that is executed early, to instead raise
        # a custom exception called ExitMock that we can catch and ignore.
        isrTask = ipIsr.IsrTask
        isrTask.ensureExposure = raiseExitMockError
        task = QuickLookIsrTask(isrTask=isrTask)
        lsst.pipe.base.testUtils.assertValidInitOutput(task)

        # Use the names of the connections here, not the Butler dataset name
        quantum = lsst.pipe.base.testUtils.makeQuantum(
            task,
            self.butler,
            self.exposure_id,
            {
                "ccdExposure": self.exposure_id,
                "camera": self.instrument_id,
                "bias": self.detector_id,
                "dark": self.detector_id,
                "flat": self.flat_id,
                "defects": self.detector_id,
                "bfKernel": self.detector_id,
                "ptc": self.detector_id,
                "deferredChargeCalib": self.detector_id,
                "crosstalk": self.detector_id,
                "linearizer": self.detector_id,
                "fringes": self.flat_id,
                "gainCorrection": self.detector_id,
                # outputs
                "outputExposure": self.exposure_id,
                "outputStatistics": self.exposure_id,
                "exposure": self.exposure_id,
            },
        )
        # Check that the proper kwargs are passed to run().
        with contextlib.suppress(ExitMockError):
            lsst.pipe.base.testUtils.runTestQuantum(task, self.butler, quantum, mockRun=False)


def raiseExitMockError(*args):
    """Raise a custom exception."""
    raise ExitMockError


class ExitMockError(Exception):
    """A custom exception to catch during a unit test."""

    pass


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
