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
from __future__ import annotations

__all__ = [
    "makeRotationTransform",
    "makeCcdToDvcsTransform",
    "makeRoiBbox",
    "stampToCcd",
    "getAmpNameFromMetadata",
    "roiImageToDvcs",
    "focalToPixel",
    "pixelToFocal",
    "ampToCcdView",
    "convertFocalToAltaz",
    "convertPixelsToAltaz",
    "convertToFocalPlane",
    "convertRoiToCcd",
    "convertCcdToRoi",
    "convertCcdToDvcs",
    "convertToAltaz",
    "convertPixelToRadec",
    "makeInitGuiderWcs",
    "getCamRotAngle",
    "DriftResult",
]

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from lsst.afw import cameraGeom
from lsst.afw.cameraGeom import Detector
from lsst.afw.image import ImageF, VisitInfo

# at top-level imports
from lsst.daf.base import PropertyList
from lsst.geom import AffineTransform, Angle, Box2D, Box2I, Extent2D, Point2D, degrees
from lsst.obs.base import createInitialSkyWcsFromBoresight
from lsst.obs.lsst import LsstCam
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms
from lsst.obs.lsst.translators.lsst import SIMONYI_LOCATION

if TYPE_CHECKING:
    from lsst.geom import SkyWcs

    from .reading import GuiderData

# build integer rotation matrices
ROTATION_MATRICES = {
    0: np.array([[1.0, 0.0], [0.0, 1.0]]),
    1: np.array([[0.0, -1], [1.0, 0.0]]),
    2: np.array([[-1.0, 0.0], [0.0, -1.0]]),
    3: np.array([[0.0, 1.0], [-1.0, 0.0]]),
}
# build inverse rotation matrices
INVERSE_ROTATION_MATRICES = {k: v.transpose() for k, v in ROTATION_MATRICES.items()}


def convertPixelToRadec(wcs: SkyWcs, x_flat: np.ndarray, y_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map detector-pixel coordinates to ICRS RA and DEC (in radians).

    Parameters
    ----------
    wcs : `SkyWcs`
        World Coordinate System object with pixelToSkyArray method.
    x_flat : `np.ndarray`
        Flattened array of x pixel coordinates.
    y_flat : `np.ndarray`
        Flattened array of y pixel coordinates.

    Returns
    -------
    raFlat : `np.ndarray`
        Array of right ascension values in radians.
    decFlat : `np.ndarray`
        Array of declination values in radians.
    """
    return wcs.pixelToSkyArray(x_flat, y_flat)


def convertPixelsToAltaz(
    wcs: SkyWcs, time: Time, xPix: np.ndarray, yPix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert detector-pixel coordinates to Altitude and Azimuth (degrees)
    in a vectorized manner.

    Parameters
    ----------
    wcs :lsst.afw.geom.SkyWcs
        World Coordinate System object with pixelToSkyArray method.
    time : astropy.time.Time
        Observation time.
    xPix : np.ndarray
        Array of x pixel coordinates.
    yPix : np.ndarray
        Array of y pixel coordinates.

    Returns
    -------
    az : np.ndarray
        Array of azimuth values in degrees (same shape as xPix/yPix).
    alt : np.ndarray
        Array of altitude values in degrees (same shape as xPix/yPix).
    """

    # 1) make sure we have numpy arrays, remember their shape
    x_arr = np.asarray(xPix)
    y_arr = np.asarray(yPix)
    shp = x_arr.shape

    # 2) flatten for the WCS call
    x_flat = x_arr.ravel()
    y_flat = y_arr.ravel()

    # 3) scalarSKyWcs → ICRS RA/Dec (radians) in bulk:
    #    (pixelToSkyArray expects floats and returns two 1D arrays)
    ra_flat, dec_flat = convertPixelToRadec(wcs, x_flat, y_flat)

    # 4) assemble a single SkyCoord and transform once
    sc_icrs = SkyCoord(
        ra=ra_flat * u.rad,
        dec=dec_flat * u.rad,
        frame="icrs",
        obstime=time,
        location=SIMONYI_LOCATION,
    )
    # Transform to AltAz
    sc_altaz = sc_icrs.transform_to(AltAz(obstime=time, location=SIMONYI_LOCATION))

    # 5) reshape back to original grid
    az = sc_altaz.az.deg.reshape(shp)
    alt = sc_altaz.alt.deg.reshape(shp)
    return az, alt


def convertFocalToAltaz(
    wcs: SkyWcs, time: Time, detector: Detector, xFocal: np.ndarray, yFocal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map focal-plane coordinates (mm) to Altitude/Azimuth (degrees) by
    chaining focal → pixel → AltAz transformations.

    Parameters
    ----------
    wcs : lsst.afw.geom.SkyWcs
        World Coordinate System object with pixelToSkyArray method.
    time : astropy.time.Time
        Observation time.
    detector : lsst.afw.cameraGeom.Detector
        Detector object.
    xFocal : np.ndarray
        Array of x focal-plane coordinates (mm).
    yFocal : np.ndarray
        Array of y focal-plane coordinates (mm).

    Returns
    -------
    az : np.ndarray
        Array of azimuth values in degrees.
    alt : np.ndarray
        Array of altitude values in degrees.
    """
    x_pix, y_pix = focalToPixel(xFocal, yFocal, detector)
    return convertPixelsToAltaz(wcs, time, x_pix, y_pix)


# Aaron's code to make transformations from ROI coordinates to Focal Plane and
# sky coordinates.
def makeRotationTransform(detNquarter: int, direction: int = 1) -> AffineTransform:
    """
    Create an AffineTransform representing a rotation by multiples of 90
    degrees.

    Parameters
    ----------
    detNquarter : `int`
        Number of 90 degree counter-clockwise rotations (modulo 4).
    direction : `int`, optional
        1 for forward rotation (default), -1 for reverse rotation (transpose)

    Returns
    -------
    rotation : `AffineTransform`
        The rotation transformation.

    Raises
    ------
    ValueError
        If direction is not 1 or -1.
    """
    rot = ROTATION_MATRICES
    irot = INVERSE_ROTATION_MATRICES

    nq = np.mod(detNquarter, 4)
    if direction == 1:
        rotation = AffineTransform(rot[nq])
    elif direction == -1:
        rotation = AffineTransform(irot[nq])
    else:
        raise ValueError(f"direction must be either +/- 1, got {direction}")
    return rotation


def makeCcdToDvcsTransform(
    bboxCcd: Box2I,
    detNquarter: int,
) -> tuple[AffineTransform, AffineTransform]:
    """
    Create forward and backward AffineTransforms for converting between CCD
    pixel coordinates and DVCS (focal plane) stamp coordinates.

    Parameters
    ----------
    bboxCcd : `Box2I`
        Bounding box for the ROI in CCD pixel coordinates.
    detNquarter : `int`
        Number of 90-degree CCW rotations to align CCD view to DVCS view.

    Returns
    -------
    forwards : `AffineTransform`
        Transform from CCD pixel coordinates (CCD view) to stamp pixel
        coordinates (DVCS view).
    backwards : `AffineTransform`
        Transform from stamp pixel coordinates (DVCS view) to CCD pixel
        coordinates (CCD view).

    Notes
    -----
    Useful for mapping between the raw detector coordinates and the orientation
    of the focal plane as used in DVCS.
    """
    # Use shared integer rotation matrices
    nq = np.mod(detNquarter, 4)
    rot = ROTATION_MATRICES
    irot = INVERSE_ROTATION_MATRICES
    # number of 90deg CCW rotations
    nq = np.mod(detNquarter, 4)

    # get LL,Size of the CCD view BBox
    lowerLeftCcd = Extent2D(bboxCcd.getCorners()[0])
    nx, ny = bboxCcd.getDimensions()

    # get translations to use for each NQ value
    boxtranslation = {}
    boxtranslation[0] = Extent2D(0, 0)
    boxtranslation[1] = Extent2D(ny - 1, 0)
    boxtranslation[2] = Extent2D(nx - 1, ny - 1)
    boxtranslation[3] = Extent2D(0, nx - 1)

    # build the forware Transform
    forwardTranslation = AffineTransform.makeTranslation(-lowerLeftCcd)
    forwardRotation = AffineTransform(rot[nq])
    forwardBoxTranslation = AffineTransform.makeTranslation(boxtranslation[nq])
    # ordering is third*second*first
    forwards = forwardBoxTranslation * forwardRotation * forwardTranslation

    backwardTranslation = AffineTransform.makeTranslation(lowerLeftCcd)
    backwardRotation = AffineTransform(irot[nq])
    backwardBoxTranslation = AffineTransform.makeTranslation(-boxtranslation[nq])
    backwards = backwardTranslation * backwardRotation * backwardBoxTranslation

    return forwards, backwards


def makeRoiBbox(md: PropertyList, camera: LsstCam) -> Box2I:
    """
    Construct the bounding box for a Guider ROI stamp in CCD coords.
    """
    # ROI metadata
    roiColumn: int = int(md["ROICOL"])
    roiRow: int = int(md["ROIROW"])
    roiColumns: int = int(md["ROICOLS"])
    roiRows: int = int(md["ROIROWS"])

    # detector and amp
    detector, ampName = getAmpNameFromMetadata(md, camera)

    # corner 0 (CCD view)
    lct = LsstCameraTransforms(camera, detector.getName())
    corner0CcdXFloat, corner0CcdYFloat = lct.ampPixelToCcdPixel(roiColumn, roiRow, ampName)
    corner0CcdX: int = int(corner0CcdXFloat)
    corner0CcdY: int = int(corner0CcdYFloat)

    # corner 2 (depends on flips)
    amp = detector[ampName]
    corner2CcdX: int = corner0CcdX - (roiColumns - 1) if amp.getRawFlipX() else corner0CcdX + (roiColumns - 1)
    corner2CcdY: int = corner0CcdY - (roiRows - 1) if amp.getRawFlipY() else corner0CcdY + (roiRows - 1)

    # bbox with LL/UR ordering
    llX: int = min(corner0CcdX, corner2CcdX)
    urX: int = max(corner0CcdX, corner2CcdX)
    llY: int = min(corner0CcdY, corner2CcdY)
    urY: int = max(corner0CcdY, corner2CcdY)

    llCcd: Point2D = Point2D(llX, llY)
    urCcd: Point2D = Point2D(urX, urY)
    ccdViewBbox: Box2I = Box2I(Box2D(llCcd, urCcd))
    return ccdViewBbox


def getAmpNameFromMetadata(md: PropertyList, camera: LsstCam) -> tuple[Detector, str]:
    """
    Unpack detector and amplifier information from metadata.

    Parameters
    ----------
    md : lsst.daf.base.PropertyList
        Metadata from one Guider CCD.
    camera : lsst.obs.lsst.LsstCam
        LsstCam object.

    Returns
    -------
    detector : lsst.afw.cameraGeom.Detector
        CCD Detector object.
    ampName : str
        Amplifier name, e.g., 'C00'.
    """
    raftBay: str = md["RAFTBAY"]
    ccdSlot: str = md["CCDSLOT"]

    segment: str = md["ROISEG"]
    ampName: str = "C" + segment[7:]
    detName: str = f"{raftBay}_{ccdSlot}"
    detector: Detector = camera[detName]

    return detector, ampName


def roiImageToDvcs(
    roi: np.ndarray,
    metadata: PropertyList,
    detector: Detector,
    ampName: str,
    camera: LsstCam,
    view: str = "dvcs",
) -> ImageF:
    """
    Convert ROI image from raw amplifier view to CCD or DVCS views,
    or embed in full CCD image.

    Parameters
    ----------
    roi : np.ndarray
        ROI array (raw, amplifier coordinates).
    metadata : lsst.daf.base.PropertyList
        Metadata from one Guider CCD.
    detector : lsst.afw.cameraGeom.Detector
        CCD detector object.
    ampName : str
        Amplifier name, e.g., 'C00'.
    camera : lsst.obs.lsst.LsstCam
        LsstCam object.
    view : str, optional
        Desired output view: 'dvcs' (default), 'ccd', or 'ccdfull'.

    Returns
    -------
    imf : lsst.afw.image.ImageF
        Image in the requested view.

    Notes
    -----
    - 'dvcs': Oriented as in the focal plane.
    - 'ccd': Oriented as in CCD coordinates.
    - 'ccdfull': ROI embedded in a full CCD-sized image.
    """

    # convert image to ccd view
    roiCcdView = ampToCcdView(roi, detector, ampName)

    # convert image to dvcs view (nb. np.rot90 works opposite to
    # afwMath.rotateImageBy90)
    roiDvcsView = np.rot90(roiCcdView, -detector.getOrientation().getNQuarter())

    ny, nx = roi.shape
    imf = ImageF(nx, ny, 0.0)

    # output ImageF
    if view == "dvcs":
        imf.array[:] = roiDvcsView
    elif view == "ccd":
        imf.array[:] = roiCcdView
    elif view == "ccdfull":
        # make the full CCD and place the ROI inside it
        nx, ny = detector.getBBox().getDimensions()
        imf = ImageF(nx, ny, 0.0)
        roiSegment = metadata["ROISEG"]
        ampName = "C" + roiSegment[7:]
        roiColumn = metadata["ROICOL"]
        roiRow = metadata["ROIROW"]
        imf.array[:] = stampToCcd(roi, imf.array[:], detector, camera, ampName, roiColumn, roiRow)

    return imf


def focalToPixel(focalX: np.ndarray, focalY: np.ndarray, det: Detector) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert focal plane coordinates (mm, DVCS) to detector pixel coordinates.

    Parameters
    ----------
    focalX : np.ndarray
        Focal plane x coordinates (mm).
    focalY : np.ndarray
        Focal plane y coordinates (mm).
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    x : np.ndarray
        Pixel x coordinates.
    y : np.ndarray
        Pixel y coordinates.
    """
    transform = det.getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS)
    x, y = transform.getMapping().applyForward(np.vstack((focalX, focalY)))
    return x.ravel(), y.ravel()


def pixelToFocal(pixelX: np.ndarray, pixelY: np.ndarray, det: Detector) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel coordinates to focal plane coordinates (mm, DVCS).

    Parameters
    ----------
    pixelX : np.ndarray
        Pixel x coordinates.
    pixelY : np.ndarray
        Pixel y coordinates.
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    focalX : np.ndarray
        Focal plane x coordinates (mm).
    focalY : np.ndarray
        Focal plane y coordinates (mm).
    """
    transform = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    focalX, focalY = transform.getMapping().applyForward(np.vstack((pixelX, pixelY)))
    return focalX.ravel(), focalY.ravel()


def stampToCcd(
    stamp: np.ndarray,
    ccdimg: np.ndarray,
    detector: Detector,
    camera: cameraGeom.Camera,
    ampName: str,
    ampCol: int,
    ampRow: int,
) -> np.ndarray:
    """
    Place ROI (stamp) into an existing full CCD array, all in CCD view.

    Parameters
    ----------
    stamp : np.ndarray
        Raw ROI array in amplifier coordinates.
    ccdimg : np.ndarray
        CCD image array in CCD coordinates (to be filled).
    detector : lsst.afw.cameraGeom.Detector
        Detector of interest.
    camera : lsst.afw.cameraGeom.Camera
        Camera object.
    ampName : str
        Amplifier name, e.g., 'C00'.
    ampCol : int
        Starting column number for ROI.
    ampRow : int
        Starting row number for ROI.

    Returns
    -------
    ccdimg : np.ndarray
        CCD image array with the ROI inserted in the appropriate location.

    Notes
    -----
    This function may print debugging information if placement fails.
    """

    stamp_ccd = ampToCcdView(stamp, detector, ampName)
    ampRows, ampCols = stamp_ccd.shape

    # get corner0 of the location for the ROI
    lct = LsstCameraTransforms(camera, detector.getName())
    corner0_CCDX, corner0_CCDY = lct.ampPixelToCcdPixel(ampCol, ampRow, ampName)
    corner0_CCDX = int(corner0_CCDX)
    corner0_CCDY = int(corner0_CCDY)

    # get opposite corner, corner2, of the ROI
    amp = detector[ampName]
    if amp.getRawFlipX():
        corner2_CCDX = corner0_CCDX - ampCols
    else:
        corner2_CCDX = corner0_CCDX + ampCols

    if amp.getRawFlipY():
        corner2_CCDY = corner0_CCDY - ampRows
    else:
        corner2_CCDY = corner0_CCDY + ampRows

    # now place the ROI
    yStart = min(corner0_CCDY, corner2_CCDY)
    yEnd = max(corner0_CCDY, corner2_CCDY)
    xStart = min(corner0_CCDX, corner2_CCDX)
    xEnd = max(corner0_CCDX, corner2_CCDX)

    if stamp_ccd.shape != (yEnd - yStart, xEnd - xStart):
        raise ValueError(
            f"ROI shape mismatch for detector '{detector.getName()}', amp '{ampName}'.\n"
            f"Expected shape: {(yEnd - yStart, xEnd - xStart)}, "
            f"Got: {stamp_ccd.shape}\n"
            f"ROI box X: {xStart}:{xEnd}, Y: {yStart}:{yEnd}"
        )

    ccdimg[yStart:yEnd, xStart:xEnd] = stamp_ccd

    return ccdimg


def ampToCcdView(stamp: np.ndarray, detector: Detector, ampName: str) -> np.ndarray:
    """
    Convert a Guider ROI stamp image from amplifier view to CCD view.

    Parameters
    ----------
    stamp : np.ndarray
        Raw ROI array in amplifier coordinates.
    detector : lsst.afw.cameraGeom.Detector
        Detector of interest.
    ampName : str
        Amplifier name, e.g., 'C00'.

    Returns
    -------
    ccdImage : np.ndarray
        ROI image array flipped to be in CCD coordinate orientation.
    """
    amplifier = detector[ampName]
    ccdImage = stamp.copy()
    if amplifier.getRawFlipX():
        ccdImage = np.fliplr(ccdImage)
    if amplifier.getRawFlipY():
        ccdImage = np.flipud(ccdImage)
    return ccdImage


def convertToFocalPlane(xccd: np.ndarray, yccd: np.ndarray, detNum: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert from CCD pixel coordinates to focal plane coordinates (mm).

    Parameters
    ----------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.
    detNum : int
        Detector number (e.g., 189 for R22_S11).

    Returns
    -------
    xfp : np.ndarray
        Focal plane x coordinates (mm).
    yfp : np.ndarray
        Focal plane y coordinates (mm).
    """
    if len(xccd) > 0:
        detector = LsstCam.getCamera()[detNum]
        # Convert the star positions to focal plane coordinates
        xfp, yfp = pixelToFocal(xccd, yccd, detector)
    else:
        xfp, yfp = np.array([]), np.array([])
    return xfp, yfp


def convertToAltaz(
    xccd: np.ndarray, yccd: np.ndarray, wcs: SkyWcs, obsTime: Time
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert CCD pixel coordinates to altitude and azimuth (degrees).

    Parameters
    ----------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.
    wcs : lsst.afw.image.Wcs
        WCS object for the guider detector.
    obsTime : astropy.time.Time
        Observation time.

    Returns
    -------
    alt : np.ndarray
        Array of altitude coordinates (degrees).
    az : np.ndarray
        Array of azimuth coordinates (degrees).
    """
    if len(xccd) > 0:
        az, alt = convertPixelsToAltaz(wcs, obsTime, xccd, yccd)
    else:
        alt, az = np.array([]), np.array([])

    return alt, az


def convertRoiToCcd(
    xroi: np.ndarray, yroi: np.ndarray, guiderData: GuiderData, guiderName: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ROI coordinates to CCD pixel coordinates.

    Parameters
    ----------
    xroi : np.ndarray
        Array of x ROI coordinates (pixels within the ROI).
    yroi : np.ndarray
        Array of y ROI coordinates (pixels within the ROI).
    guiderData : GuiderData
        GuiderData object containing the guider datasets and view information.
    guiderName : str
        Name of the guider (e.g., 'R44_SG0').

    Returns
    -------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.

    Raises
    ------
    ValueError
        If the view in GuiderData is not supported.
    """
    view = guiderData.view
    stamps = guiderData[guiderName]

    xroi = np.asarray(xroi)
    yroi = np.asarray(yroi)

    box, _, roi2ccd = stamps.getArchiveElements()[0]
    if view == "ccd":
        # convert roi coords to ccd coords
        # by adding the lower left corner of the box
        lower_left_corner = box.getMin()
        xmin, ymin = lower_left_corner.getX(), lower_left_corner.getY()
        xccd, yccd = xroi + xmin, yroi + ymin

    elif view == "dvcs":
        # convert roi coords to ccd coords using the roi2ccd transform
        # roi2ccd is an AffineTransform that converts
        # from roi coords to ccd coords
        xccd, yccd = roi2ccd(xroi, yroi)
    else:
        raise ValueError(f"Unsupported view '{view}' in convertRoiToCcd" "must be 'ccd' or 'dvcs'")

    return xccd, yccd


def convertCcdToRoi(
    xccd: np.ndarray, yccd: np.ndarray, guiderData: GuiderData, guiderName: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert CCD pixel coordinates to ROI pixel coordinates.

    Parameters
    ----------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.
    guiderData : GuiderData
        GuiderData object containing the guider datasets and view information.
    guiderName : str
        Name of the guider (e.g., 'R44_SG0').

    Returns
    -------
    xroi : np.ndarray
        Array of x ROI pixel coordinates.
    yroi : np.ndarray
        Array of y ROI pixel coordinates.

    Raises
    ------
    ValueError
        If the view in GuiderData is not supported.
    """
    view = guiderData.view
    stamps = guiderData[guiderName]

    if len(xccd) == 0:
        return np.array([]), np.array([])

    box, ccd2dvcs, _ = stamps.getArchiveElements()[0]

    if view == "ccd":
        lower_left_corner = box.getMin()
        xmin, ymin = lower_left_corner.getX(), lower_left_corner.getY()
        xroi, yroi = xccd - xmin, yccd - ymin

    elif view == "dvcs":
        xroi, yroi = ccd2dvcs(xccd, yccd)

    else:
        raise ValueError(f"Unsupported view '{view}' in convertCcdToRoi", "must be 'ccd' or 'dvcs'")

    return xroi, yroi


def convertCcdToDvcs(
    xccd: np.ndarray, yccd: np.ndarray, guiderData: GuiderData, guiderName: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert CCD pixel coordinates to DVCS (focal plane)
    coordinates (pixels or mm).

    Parameters
    ----------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.
    guiderData : GuiderData
        GuiderData object containing the guider datasets and view information.
    guiderName : str
        Name of the guider (e.g., 'R44_SG0').

    Returns
    -------
    xdvcs : np.ndarray
        Array of x DVCS focal plane coordinates.
    ydvcs : np.ndarray
        Array of y DVCS focal plane coordinates.
    """
    stamps = guiderData[guiderName]

    if len(xccd) == 0:
        return np.array([]), np.array([])

    _, ccd2dvcs, _ = stamps.getArchiveElements()[0]
    xdvcs, ydvcs = ccd2dvcs(xccd, yccd)
    return xdvcs, ydvcs


def getCamRotAngle(visitinfo: VisitInfo) -> float:
    """
    Compute the camera rotation angle in degrees from visit information.

    Parameters
    ----------
    visitinfo : lsst.afw.image.VisitInfo
        Visit information from an exposure.

    Returns
    -------
    float
        Camera rotation angle in degrees, wrapped near zero.
    """
    camRot = (
        visitinfo.getBoresightParAngle().asDegrees() - visitinfo.getBoresightRotAngle().asDegrees() - 90.0
    )
    camRotAngle = Angle(camRot, degrees)
    camRotAngleW = camRotAngle.wrapNear(Angle(0.0))
    return camRotAngleW.asDegrees()


# Putting former guiderwcs here


def makeInitGuiderWcs(camera: cameraGeom.Camera, visitInfo: VisitInfo) -> dict[str, SkyWcs]:
    """
    Create initial WCS for each guider detector in the camera.

    Parameters
    ----------
    camera : lsst.afw.cameraGeom.Camera
        Camera object containing detectors.
    visitInfo : lsst.afw.image.VisitInfo
        Visit information from an exposure.

    Returns
    -------
    dict[str, SkyWcs]
        Dictionary of WCS objects keyed by detector name for guider detectors.
    """
    orientation = visitInfo.getBoresightRotAngle()
    boresight = visitInfo.getBoresightRaDec()

    # Get WCS for each CCD in camera
    camWcs: dict[str, SkyWcs] = {}

    for det in camera:
        if det.getType() == cameraGeom.DetectorType.GUIDER:
            # get WCS
            args = boresight, orientation, det
            camWcs[det.getName()] = createInitialSkyWcsFromBoresight(*args, flipX=False)

    return camWcs


# Data class for drift results
@dataclass
class DriftResult:
    """
    Data class to store results of telescope drift calculations.

    Attributes
    ----------
    ra_point : float
        Telescope pointing right ascension in degrees.
    dec_point : float
        Telescope pointing declination in degrees.
    ra_real : float
        Actual right ascension in degrees.
    dec_real : float
        Actual declination in degrees.
    delta_ra_arcsec : float
        Pointing error in RA in arcseconds.
    delta_dec_arcsec : float
        Pointing error in Dec in arcseconds.
    az_drift_arcsec : float
        Azimuth drift in arcseconds.
    el_drift_arcsec : float
        Elevation drift in arcseconds.
    total_drift_arcsec : float
        Total drift in arcseconds.
    el_start : float
        Starting elevation in degrees.
    pixel_offset : tuple of float
        Pixel offset from raw WCS origin to calibrated WCS origin.
    """

    ra_point: float
    dec_point: float
    ra_real: float
    dec_real: float
    delta_ra_arcsec: float
    delta_dec_arcsec: float
    az_drift_arcsec: float
    el_drift_arcsec: float
    total_drift_arcsec: float
    el_start: float
    pixel_offset: tuple[float, float]

    def __str__(self, expid: int | None = None) -> str:
        """
        Return a formatted string summarizing the drift results.

        Parameters
        ----------
        expid : int, optional
            Exposure identifier to include in the summary. Defaults to None,
            which leaves the field blank, as it does when Python calls this
            implicitly via print() or str().

        Returns
        -------
        str
            Formatted summary string.
        """
        expidStr = "" if expid is None else str(expid)
        s = (
            f"Exposure summary: {expidStr}\n"
            f"Telescope pointing (RA, Dec): ({self.ra_point:.6f}, {self.dec_point:.6f})\n"
            f"Actual pointing   (RA, Dec) : ({self.ra_real:.6f}, {self.dec_real:.6f})\n"
            f"Pointing error (RA, Dec).   : ({self.delta_ra_arcsec:.1f},"
            f"{self.delta_dec_arcsec:.1f}) arcseconds\n"
            f"Starting elevation.         : {self.el_start:.2f} degrees\n"
            f"Pixel offset from origin   : ({self.pixel_offset[0]:.4f}, {self.pixel_offset[1]:.4f}) pixels\n"
            f"\n"
            f"-------------------- Drift Summary --------------------\n"
            f"Azimuth drift   : {self.az_drift_arcsec:.2f} arcseconds\n"
            f"Elevation drift : {self.el_drift_arcsec:.2f} arcseconds\n"
            f"Total drift.    : {self.total_drift_arcsec:.2f} arcseconds\n"
        )
        return s

    def summary(self, expid: int | None = None) -> None:
        """
        Print a summary of the drift results.

        Parameters
        ----------
        expid : int, optional
            Exposure identifier to include in the summary. Defaults to None,
            which leaves the field blank.
        """
        print(self.__str__(expid))


def getObsAltAz(
    ra: float,
    dec: float,
    pressure: u.Quantity,
    hum: float,
    temperature: u.Quantity,
    wl: u.Quantity,
    time: Time,
) -> SkyCoord:
    """
    Compute the observed AltAz coordinates for given RA/Dec and conditions.

    Parameters
    ----------
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    pressure : astropy.units.Quantity
        Atmospheric pressure with units.
    hum : float
        Relative humidity as a fraction (0-1).
    temperature : astropy.units.Quantity
        Temperature with units.
    wl : astropy.units.Quantity
        Wavelength of observation with units.
    time : astropy.time.Time
        Observation time.

    Returns
    -------
    astropy.coordinates.SkyCoord
        Altitude and azimuth coordinates of the object.
    """
    skyLocation = SkyCoord(ra * u.deg, dec * u.deg)
    altAz1 = AltAz(
        obstime=time,
        location=SIMONYI_LOCATION,
        pressure=pressure,
        temperature=temperature,
        relative_humidity=hum,
        obswl=wl,
    )
    obsAltAz1 = skyLocation.transform_to(altAz1)
    return obsAltAz1


def DeltaAltAz(
    ra: float,
    dec: float,
    pressure: u.Quantity,
    hum: float,
    temperature: u.Quantity,
    wl: u.Quantity,
    time1: Time,
    time2: Time,
) -> list[float]:
    """
    Calculate the change in AltAz coordinates between two times.

    Parameters
    ----------
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    pressure : astropy.units.Quantity
        Atmospheric pressure with units.
    hum : float
        Relative humidity as a fraction (0-1).
    temperature : astropy.units.Quantity
        Temperature with units.
    wl : astropy.units.Quantity
        Wavelength of observation with units.
    time1 : astropy.time.Time
        Start time of observation.
    time2 : astropy.time.Time
        End time of observation.

    Returns
    -------
    list of float
        List containing azimuth change and elevation change in arcseconds.
    """
    obsAltAz1 = getObsAltAz(ra, dec, pressure, hum, temperature, wl, time1)
    obsAltAz2 = getObsAltAz(ra, dec, pressure, hum, temperature, wl, time2)
    # 1 is at the beginning of the exposure, 2 is at the end
    # el, az are the actual values, prime values reflect the pointing model
    # These are all in degrees
    el1 = obsAltAz1.alt.deg
    az1 = obsAltAz1.az.deg
    el2 = obsAltAz2.alt.deg
    az2 = obsAltAz2.az.deg
    # Change values are the change from the beginning
    # to the end of the exposure,
    # in arcseconds
    azChange = (az2 - az1) * 3600.0
    elChange = (el2 - el1) * 3600.0
    return [azChange, elChange]


def computePointModelDrift(metadata: Mapping[str, Any], cWcs: SkyWcs, rWcs: SkyWcs) -> DriftResult:
    """
    Calculate Point Model drift from exposure metadata and
    the measured `calexp` WCS.

    Parameters
    ----------
    metadata : dict-like
        Must contain at least: 'FILTBAND', 'PRESSURE', 'AIRTEMP', 'HUMIDITY',
        'MJD-BEG', 'MJD-END', 'RASTART', 'DECSTART', 'ELSTART'.
    cWcs : lsst.afw.geom.SkyWcs
        Calibrated WCS object.
    rWcs : lsst.afw.geom.SkyWcs
        Raw WCS object.

    Returns
    -------
    DriftResult
        Object containing drift calculation results.

    Example
    -------
    expId = 2025072300537
    rawExp = butler.get('raw', detector=94,
                        exposure=expId, instrument=instrument)
    calExp = butler.get('preliminary_visit_image',
                        detector=94, visit=expId, instrument=instrument)

    md = rawExp.getMetadata()
    cWcs = calExp.getWcs()
    rWcs = rawExp.getWcs()

    drift547 = calculate_drift(md, cWcs, rWcs)
    drift547.summary(2025072300537)
    """
    wavelengths = {"u": 3671, "g": 4827, "r": 6223, "i": 7546, "z": 8691, "y": 9712}
    filter = metadata["FILTBAND"]
    wl = wavelengths[filter] * u.angstrom
    pressure = metadata["PRESSURE"] * u.pascal
    temperature = metadata["AIRTEMP"] * u.Celsius
    hum = metadata["HUMIDITY"]
    time1 = Time(metadata["MJD-BEG"], format="mjd", scale="tai")
    time2 = Time(metadata["MJD-END"], format="mjd", scale="tai")
    ra_point = metadata["RASTART"]
    dec_point = metadata["DECSTART"]
    el_start = metadata["ELSTART"]

    azChangePoint, elChangePoint = DeltaAltAz(
        ra_point, dec_point, pressure, hum, temperature, wl, time1, time2
    )

    # Use WCS to get actual (ra, dec) at image center
    calExpSkyCenter = cWcs.pixelToSky(rWcs.getPixelOrigin())
    ra_real = calExpSkyCenter.getRa().asDegrees()
    dec_real = calExpSkyCenter.getDec().asDegrees()
    delta_ra = (ra_real - ra_point) * 3600.0
    delta_dec = (dec_real - dec_point) * 3600.0

    # Calculate pixel offset from raw WCS origin to calibrated WCS origin
    pixel_offset = tuple(cWcs.getPixelOrigin() - rWcs.getPixelOrigin())

    azChangeReal, elChangeReal = DeltaAltAz(ra_real, dec_real, pressure, hum, temperature, wl, time1, time2)

    az_drift = azChangeReal - azChangePoint
    el_drift = elChangeReal - elChangePoint
    total_drift = np.sqrt(el_drift**2 + (az_drift * np.cos(np.radians(el_start))) ** 2)

    return DriftResult(
        ra_point=ra_point,
        dec_point=dec_point,
        ra_real=ra_real,
        dec_real=dec_real,
        delta_ra_arcsec=delta_ra,
        delta_dec_arcsec=delta_dec,
        az_drift_arcsec=az_drift,
        el_drift_arcsec=el_drift,
        total_drift_arcsec=total_drift,
        el_start=el_start,
        pixel_offset=pixel_offset,
    )
