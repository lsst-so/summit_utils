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

import dataclasses
import importlib.resources
from typing import Any


import lsst.afw.cameraGeom as camGeom
import lsst.afw.image as afwImage
import lsst.ip.isr as ipIsr
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.ip.isr import IsrTaskLSST
from lsst.ip.isr.isrTaskLSST import IsrTaskLSSTConnections
from lsst.meas.algorithms.installGaussianPsf import InstallGaussianPsfTask
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask

__all__ = ["QuickLookIsrTask", "QuickLookIsrTaskConfig"]


class QuickLookIsrTaskConnections(IsrTaskLSSTConnections):
    """Copy isrTask's connections, changing prereq min values to zero.

    Copy all the connections directly for IsrTask, keeping ccdExposure as
    required as non-zero, but changing all the other PrerequisiteInputs'
    minimum values to zero.
    """

    def __init__(self, *, config: Any = None):
        # programatically clone all of the connections from isrTask
        # setting minimum values to zero for everything except the ccdExposure
        super().__init__(
            config=IsrTaskLSST.ConfigClass()
        )  # need a dummy config, isn't used other than for ctor
        for name, connection in self.allConnections.items():
            if hasattr(connection, "minimum"):
                setattr(
                    self,
                    name,
                    dataclasses.replace(connection, minimum=(0 if name != "ccdExposure" else 1)),
                )

        exposure = cT.Output(  # called just "exposure" to mimic isrTask's return struct
            name="quickLookExp",
            doc="The quickLook output exposure.",
            storageClass="ExposureF",
            dimensions=("instrument", "exposure", "detector"),
        )
        # set like this to make it explicit that the outputExposure
        # and the exposure are identical. The only reason there are two is for
        # API compatibility.
        self.outputExposure = exposure


class QuickLookIsrTaskConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=QuickLookIsrTaskConnections  # type: ignore
):
    """Configuration parameters for QuickLookIsrTask."""

    doRepairCosmics: pexConfig.Field[bool] = pexConfig.Field(
        dtype=bool, doc="Interpolate over cosmic rays?", default=True
    )


class QuickLookIsrTask(pipeBase.PipelineTask):
    """Task automatically performing as much isr as possible. Should never fail

    Automatically performs as much isr as is possible, depending on the
    calibration products available. All calibration products that can be found
    are applied, and if none are found, the image is assembled, the overscan is
    subtracted and the assembled image is returned. Optionally, cosmic rays are
    interpolated over.
    """

    ConfigClass = QuickLookIsrTaskConfig
    config: QuickLookIsrTaskConfig
    _DefaultName = "quickLook"

    def __init__(self, isrTask: IsrTaskLSST = IsrTaskLSST, **kwargs: Any):
        super().__init__(**kwargs)
        # Pass in IsrTask so that we can modify it slightly for unit tests.
        # Note that this is not an instance of the IsrTask class, but the class
        # itself, which is then instantiated later on, in the run() method,
        # with the dynamically generated config.
        # import pdb; pdb.set_trace()
        if IsrTaskLSST._DefaultName != 'isrLSST':
            raise RuntimeError("QuickLookIsrTask should now always use IsrTaskLSST for processing.")
        self.isrTask = IsrTaskLSST

    def run(
        self,
        ccdExposure: afwImage.Exposure,
        *,
        camera: camGeom.Camera | None = None,
        bias: afwImage.Exposure | None = None,
        dark: afwImage.Exposure | None = None,
        flat: afwImage.Exposure | None = None,
        defects: ipIsr.Defects | None = None,
        linearizer: ipIsr.linearize.LinearizeBase | None = None,
        crosstalk: ipIsr.crosstalk.CrosstalkCalib | None = None,
        bfKernel: ipIsr.BrighterFatterKernel | None = None,
        ptc: ipIsr.PhotonTransferCurveDataset | None = None,
        isrBaseConfig: ipIsr.IsrTaskLSSTConfig | None = None,
        deferredChargeCalib: Any | None = None,
        gainCorrection: ipIsr.IsrCalib | None = None,
    ) -> pipeBase.Struct:
        """Run isr and cosmic ray repair using, doing as much isr as possible.

        Retrieves as many calibration products as are available, and runs isr
        with those settings enabled, but always returns an assembled image at
        a minimum. Then performs cosmic ray repair if configured to.

        Parameters
        ----------
        ccdExposure : `lsst.afw.image.Exposure`
            The raw exposure that is to be run through ISR.  The
            exposure is modified by this method.
        camera : `lsst.afw.cameraGeom.Camera`, optional
            The camera geometry for this exposure. Required if
            one or more of ``ccdExposure``, ``bias``, ``dark``, or
            ``flat`` does not have an associated detector.
        bias : `lsst.afw.image.Exposure`, optional
            Bias calibration frame.
        dark : `lsst.afw.image.Exposure`, optional
            Dark calibration frame.
        flat : `lsst.afw.image.Exposure`, optional
            Flat calibration frame.
        fringes : `lsst.afw.image.Exposure`, optional
            The fringe correction data.
            This input is slightly different than the `fringes` keyword to
            `lsst.ip.isr.IsrTask`, since the processing done in that task's
            `runQuantum` method is instead done here.
        defects : `lsst.ip.isr.Defects`, optional
            List of defects.
        linearizer : `lsst.ip.isr.linearize.LinearizeBase`, optional
            Functor for linearization.
        crosstalk : `lsst.ip.isr.crosstalk.CrosstalkCalib`, optional
            Calibration for crosstalk.
        bfKernel : `ipIsr.BrighterFatterKernel`, optional
            New Brighter-fatter kernel.
        ptc : `lsst.ip.isr.PhotonTransferCurveDataset`, optional
            Photon transfer curve dataset, with, e.g., gains
            and read noise.
        cti : `lsst.ip.isr.DeferredChargeCalib`, optional
            Charge transfer inefficiency correction calibration.
        isrBaseConfig : `lsst.ip.isr.IsrTaskLSSTConfig`, optional
            An isrTask config to act as the base configuration. Options which
            involve applying a calibration product are ignored, but this allows
            for the configuration of e.g. the number of overscan columns.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct with component:
            - ``exposure`` : `afw.image.Exposure`
                The ISRed and cosmic-ray-repaired exposure.
        """
        if not isrBaseConfig:
            isrConfig = IsrTaskLSST.ConfigClass()
            with importlib.resources.path("lsst.summit.utils", "resources/config/quickLookIsr.py") as cfgPath:
                isrConfig.load(cfgPath)
        else:
            isrConfig = isrBaseConfig

        isrConfig.doBias = False
        isrConfig.doDark = False
        isrConfig.doFlat = False
        isrConfig.doDefect = False
        isrConfig.doLinearize = False
        isrConfig.doCrosstalk = False
        isrConfig.doBrighterFatter = False
        isrConfig.doDeferredCharge = False
        isrConfig.doCorrectGains = False

        if bias:
            isrConfig.doBias = True
            self.log.info("Running with bias correction")

        if dark:
            isrConfig.doDark = True
            self.log.info("Running with dark correction")

        if flat:
            isrConfig.doFlat = True
            self.log.info("Running with flat correction")

        if defects:
            isrConfig.doDefect = True
            self.log.info("Running with defect correction")

        if linearizer:
            isrConfig.doLinearize = True
            self.log.info("Running with linearity correction")

        if crosstalk:
            isrConfig.doCrosstalk = True
            self.log.info("Running with crosstalk correction")

        if bfKernel is not None:
            isrConfig.doBrighterFatter = True
            self.log.info("Running with brighter-fatter correction")

        if deferredChargeCalib is not None:
            isrConfig.doDeferredCharge = True
            self.log.info("Running with CTI correction")

        if gainCorrection is not None:
            isrConfig.doCorrectGains = True
            self.log.info("Running with Gain corrections")

        if ptc is None:
            raise RuntimeError("IsrTaskLSST requires a PTC.")

        isrTask = self.isrTask(config=isrConfig)

        # DM-47959: TODO Add fringe correction to IsrTaskLSST.
        # if fringes:
        #     # Must be run after isrTask is instantiated.
        #     isrTask.fringe.loadFringes(
        #         fringes,
        #         expId=ccdExposure.info.id,
        #         assembler=isrTask.assembleCcd if isrConfig.doAssembleIsrExposures else None,
        #     )

        result = isrTask.run(
            ccdExposure,
            camera=camera,
            bias=bias,
            dark=dark,
            flat=flat,
            defects=defects,
            linearizer=linearizer,
            crosstalk=crosstalk,
            bfKernel=bfKernel,
            ptc=ptc,
            deferredChargeCalib=deferredChargeCalib,
            gainCorrection=gainCorrection,
        )

        postIsr = result.exposure

        if self.config.doRepairCosmics:
            try:  # can fail due to too many CRs detected, and we always want an exposure back
                self.log.info("Repairing cosmics...")
                if postIsr.getPsf() is None:
                    installPsfTask = InstallGaussianPsfTask()
                    installPsfTask.run(postIsr)

                # TODO: try adding a reasonably wide Gaussian as a temp PSF
                # and then just running repairTask on its own without any
                # imChar. It should work, and be faster.
                repairConfig = CharacterizeImageTask.ConfigClass()
                repairConfig.doMeasurePsf = False
                repairConfig.doApCorr = False
                repairConfig.doDeblend = False
                repairConfig.doWrite = False
                repairConfig.repair.cosmicray.nCrPixelMax = 200000
                repairTask = CharacterizeImageTask(config=repairConfig)

                repairTask.repair.run(postIsr)
            except Exception as e:
                self.log.warning(f"During CR repair caught: {e}")

        # exposure is returned for convenience to mimic isrTask's API.
        return pipeBase.Struct(exposure=postIsr, outputExposure=postIsr)
