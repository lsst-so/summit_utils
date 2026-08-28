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

from collections.abc import Iterable
from typing import Any

import matplotlib
import numpy as np
import pandas
from scipy.optimize import curve_fit  # type: ignore

from lsst.afw.cameraGeom import FIELD_ANGLE, Detector
from lsst.daf.butler import Butler, DatasetRef
from lsst.geom import Box2I, Extent2I, Point2I
from lsst.summit.utils.utils import getCameraFromInstrumentName
from lsst.utils.plotting.figures import make_figure


def gaussian2dFitFunction(
    xy: tuple[np.ndarray, np.ndarray],
    peak: float,
    fwhm: float,
    x0: float,
    y0: float,
    baseline: float = 0.0,
) -> np.ndarray:
    """Gaussian distribution with centroid.

    Parameters
    ----------
    xy: `tuple` of `np.ndarray`
        Points coordinates.
    peak: `float`
        Values of the intesity peak.
    fwhm: `float`
        Full Width at Half Maximum fo the distribution.
    x0: `float`
        The x position of the 2d Guassian function.
    y0: `float`
        The y position of the 2d Guassian function.
    baseline: `float`
        Offset to apply. Default 0.

    Returns
    -------
    pdf: `np.ndarray`
        Probability density function of the distribution.
    """
    x, y = xy
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return baseline + peak * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def moffat2dFitFunction(
    xy: tuple[np.ndarray, np.ndarray],
    peak: float,
    alpha: float,
    beta: float,
    x0: float,
    y0: float,
    baseline: float,
) -> np.ndarray:
    """Moffat distribution with centroid.

    Parameters
    ----------
    xy: `tuple` of `np.ndarray`
        Points coordinates.
    peak: `float`
        Values of the intesity peak.
    alpha: `float`
        The alpha parameter for the Moffat distribution.
    beta: `float`
        The beta parameter for the Moffat distribution.
    x0: `float`
        x coordinate of the distribution.
    y0: `float`
        y coordinate of the distribution.
    baseline: `float`
        Offset to apply. Default 0.

    Returns
    -------
    pdf: `np.ndarray`
        Probability density function of the distribution.
    """
    x, y = xy
    return baseline + peak * (1 + (((x - x0) ** 2 + (y - y0) ** 2)) / alpha**2) ** (-beta)


def doRadialAnalysis(
    data: np.ndarray, fitModel: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Perform the radial analysis on a star cutout

    Parameters
    ----------
    data: `np.ndarray`
        2D array containing the star cutout.
    fitModel: `str`
        Model used for the fit ('moffat' or 'gauss').

    Returns
    -------
    x: `np.ndarray`
        1d array with the radial from the centroid.
    yScatter: `np.ndarray`
        The flattened radial distribution of the start intensity.
        Usefull for plotting purpose.
    yFit: `np.ndarray`
        The fitted radial distribution.
    fwhmFit: `float`
        The Full Width at Half Maximum of the fitted distribution.
    eE50Diameter: `float`
        The Encircled Energy diameter at 50%.
    eE80Diameter: `float`
        The Encircled Energy diameter at 80%.
    """
    # Create meshgrid for fitting with x and y positions
    xGrid, yGrid = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    xy = (xGrid.ravel(), yGrid.ravel())
    radialValues = data.ravel()

    match fitModel.lower():
        case "moffat":
            # Initial guess for Moffat fitting
            initialGuess = [
                np.max(radialValues),
                4,
                2,
                data.shape[1] / 2,
                data.shape[0] / 2,
                np.median(radialValues),
            ]
            params, _ = curve_fit(moffat2dFitFunction, xy, radialValues, p0=initialGuess, maxfev=10000)
            _, alphaFit, betaFit, x0Fit, y0Fit, baselineFit = params
            fwhmFit = np.abs(2.0 * alphaFit * np.sqrt(2 ** (1 / betaFit) - 1.0))
        case "gauss":
            # Initial guess for Gaussian fitting
            initialGuess = [
                np.max(radialValues),
                10,
                data.shape[1] / 2,
                data.shape[0] / 2,
                np.median(radialValues),
            ]
            params, _ = curve_fit(gaussian2dFitFunction, xy, radialValues, p0=initialGuess, maxfev=10000)
            _, fwhmFit, x0Fit, y0Fit, baselineFit = params
        case _:
            raise ValueError(f"The model {fitModel} is not among the available ones (gauss, moffat)")

    # Compute the curve of growth (cumulative energy)
    radii = np.sqrt((xGrid - x0Fit) ** 2 + (yGrid - y0Fit) ** 2).ravel()

    sortedIndices = np.argsort(radii)
    sortedradii = radii[sortedIndices]
    sortedValues = radialValues[sortedIndices] - baselineFit
    cumulativeEnergy = np.cumsum(sortedValues)

    # Determine 50% and 80% encircled energy diameters
    eE50Diameter = 2 * sortedradii[np.searchsorted(cumulativeEnergy, 0.5 * cumulativeEnergy[-1])]
    eE80Diameter = 2 * sortedradii[np.searchsorted(cumulativeEnergy, 0.8 * cumulativeEnergy[-1])]

    x = 0.2 * sortedradii
    yScatter = sortedValues + baselineFit  # added back the backgroud.

    match fitModel.lower():
        case "moffat":
            yFit = moffat2dFitFunction(xy, *params)[sortedIndices]
        case "gauss":
            yFit = gaussian2dFitFunction(xy, *params)[sortedIndices]
        case _:
            raise ValueError(f"The model {fitModel} is not among the available ones (gauss, moffat)")

    return (
        x,
        yScatter,
        yFit,
        fwhmFit,
        eE50Diameter,
        eE80Diameter,
    )


def makeLayerPlot(
    ax: matplotlib.axes.Axes,
    data: np.ndarray,
    e1: float,
    e2: float,
    e: float,
    fitModel: str,
    layers: list[str] | str = "all",
    levels: np.ndarray | Iterable[float] | None = None,
) -> tuple[float, float, float]:
    """Make per axes layer plot.

    Create a plot with three possible layers:
        - Background image with the star cutout
        - The contour level of the star
        - The radial profile with Gaussian and Moffat fit
    The value of FWHM and Encircled Energy Radii (EE)
    at 50% and 80% are reported if the 'radial' layer has been chosen.

    Parameters
    ----------
    ax: `matplotlib.axes.Axes`
        The axes to use
    data: `np.ndarray`
        2D array containing the star cutout
    fitModel: `str`
        Model used for the fit ('moffat' or 'gauss')
    layers: list[str] | str = "all",
        List of layers to be displayed
        ('background', 'radial', 'contour', 'ellipticity').
    levels: `np.ndarray` or `Iterable` of `float` or `None`, optional
        The levels value for the contour layer.
        If None, is set to `np.linspace(1.5*np.std(data), data.max(), 5)`

    Returns
    -------
    fwhmFit: `float`
        The FWHM of the fitted distribution.
    eE50Diameter: `float`
        The encircled energy diameter at 50%.
    eE80Diameter: `float`
        The encircled energy diameter at 80%.
    """
    if layers == "all":
        layers = ["background", "radial", "contours", "ellipticity"]

    (
        x,
        yScatter,
        yFit,
        fwhmFit,
        eE50Diameter,
        eE80Diameter,
    ) = doRadialAnalysis(data, fitModel)

    # get figure and axes position on figure
    # to create multiple axes on that position
    fig = ax.get_figure()
    bbox = ax.get_position()
    bboxArray: tuple[float, float, float, float] = bbox.bounds

    assert fig is not None

    # plot the background layer if present
    axBkg = None
    if "background" in layers:
        axBkg = fig.add_axes(bboxArray, frameon=False)
        axBkg.imshow(data, cmap="gray", origin="lower", zorder=1)
        axBkg.set(zorder=1, xticks=[], yticks=[])

    # plot the contour layer if present
    if "contour" in layers:
        axCtr = fig.add_axes(bboxArray, frameon=False)
        xGrid, yGrid = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
        levels = levels if levels is not None else np.linspace(1.5 * np.std(data), data.max(), 5)
        axCtr.contour(xGrid, yGrid, data, cmap="spring", levels=levels, alpha=0.7)
        axCtr.set(zorder=2, facecolor=(1, 1, 1, 0), xticks=[], yticks=[])

    # plot the radial analysis layer if present
    if "radial" in layers:
        axRad = fig.add_axes(bboxArray, frameon=False)
        axRad.scatter(
            x,
            yScatter,
            marker="o",
            s=5,
            facecolor="none",
            edgecolor="cyan",
            label="Data",
        )

        axRad.plot(
            x,
            yFit,
            ls="-",
            color="y",
            label=f"{fitModel} Fit",
        )

        axRad.set(zorder=3, facecolor=(1, 1, 1, 0), xticks=[], yticks=[])

    if "ellipticity" in layers:
        axEll = fig.add_axes(bboxArray, frameon=False)
        axEll.set(zorder=4, facecolor=(1, 1, 1, 0), xticks=[], yticks=[])

        quiverScale = 1
        eU = e * np.cos(0.5 * np.arctan2(e2, e1))
        eV = e * np.sin(0.5 * np.arctan2(e2, e1))
        peak = np.argmax(data)
        centerY, centerX = np.unravel_index(peak, data.shape)

        # better to use the background axes limits
        if "background" in layers:
            assert axBkg is not None, "Logic error: axBkg should have been defined above"
            axEll.set(xlim=axBkg.get_xlim(), ylim=axBkg.get_ylim())
        else:  # otherwise use the cutout limits
            axEll.set(xlim=(0, data.shape[1]), ylim=(0, data.shape[0]))

        axEll.quiver(
            centerX,
            centerY,
            eU,
            eV,
            headlength=0,
            headaxislength=0,
            scale=quiverScale,
            pivot="mid",
            color="fuchsia",
            width=0.015,
        )

    return fwhmFit, eE50Diameter, eE80Diameter


def compactifyLayout(
    rectDict: dict[str, tuple[float, float, float, float]],
) -> dict[str, tuple[float, float, float, float]]:
    """Compact the layout of the rectangles to fit in a smaller area.
    This function rescales the rectangles to fit within a specified area
    while maintaining their aspect ratios.

    Parameters
    ----------
    rectDict: `dict`
        Dictionary with the detector boxes.

    Returns
    -------
    rectDict: `dict`
        Dictionary with the rescaled rectangles.
    """

    margin = 0.02

    lefts = [r[0] for r in rectDict.values()]
    bottoms = [r[1] for r in rectDict.values()]

    min_left = min(lefts)
    max_left = max(lefts)
    min_bottom = min(bottoms)
    max_bottom = max(bottoms)

    range_x = max_left - min_left if max_left != min_left else 1
    range_y = max_bottom - min_bottom if max_bottom != min_bottom else 1

    scale_x = (1 - 2 * margin) / range_x
    scale_y = (1 - 2 * margin) / range_y

    newRectDict: dict[str, tuple[float, float, float, float]] = {}
    for name, (left, bottom, width, height) in rectDict.items():
        new_left = (left - min_left) * scale_x + margin
        new_bottom = (bottom - min_bottom) * scale_y + margin
        newRectDict[name] = (new_left, new_bottom, width, height)

    return newRectDict


def computeColorbarRect(
    rectDict: dict[str, tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Compute the colorbar rectangle based on the detector rectangles

    Parameters
    ----------
    rectDict: `dict`
        Dictionary with the detector boxes.

    Returns
    -------
    rect: `tuple`
        The box in which the colorbar will be drawn.
    """

    width = 0.015
    padding = 0.01

    rects = list(rectDict.values())

    min_bottom = min(r[1] for r in rects)
    max_top = max(r[1] + r[3] for r in rects)
    max_right = max(r[0] + r[2] for r in rects)

    height = max_top - min_bottom
    left = max_right + padding
    bottom = min_bottom

    return (left, bottom, width, height)


def createFigWithInstrumentLayout(
    fig: matplotlib.figure.Figure,
    instrument: str,
    onlyS11: bool = False,
) -> dict[str, matplotlib.axes.Axes]:
    """Create a figure with the requested instrument layout

    Parameters
    ----------
    fig: `matplotlib.figure.Figure`
        The figure to use
    instrument: `str`
        The instrument name.
    onlyS11: `bool`, optional
        If True, only S11 detectors are shown. Default False.

    Returns
    -------
    axsDict: `dict[str, matplotlib.axes.Axes]`
        A dictionary with the detector names as keys and the axes as values.
    """

    camera = getCameraFromInstrumentName(instrument)
    detectors = [detector.getId() for detector in camera]

    rectDict: dict[str, tuple[float, float, float, float]] = {}
    for name in detectors:
        detector = camera.get(name)
        detName = detector.getName()

        # Get the corners of the detector in FIELD_ANGLE
        corners = detector.getCorners(FIELD_ANGLE)
        corners_deg = np.rad2deg(corners)  # Convert corners to degrees

        # turn the cornsers coords in (letf, bottom, width, height)
        detRect = (
            corners_deg[:, 0].min(),
            corners_deg[:, 1].min(),
            corners_deg[:, 0].max() - corners_deg[:, 0].min(),
            corners_deg[:, 1].max() - corners_deg[:, 1].min(),
        )

        if onlyS11 and "S11" not in detName:
            continue
        rectDict[detName] = detRect

    if onlyS11:
        rectDict = compactifyLayout(rectDict)

    axsDict = {}
    for detName, rect in rectDict.items():
        # Create an axes for each detector
        ax = fig.add_axes(rect)
        ax.set(xticks=[], yticks=[])
        axsDict[detName] = ax

    cbraRect = computeColorbarRect(rectDict)
    cbarAx = fig.add_axes(cbraRect)
    axsDict["cbar"] = cbarAx

    return axsDict


def makePsfPanel(
    cutouts: dict[str, tuple[np.ndarray, tuple[float, float, float]]],
    instrument: str = "LSSTComCam",
    onlyS11: bool = False,
    layers: list[str] | str = "all",
    fitModel: str = "moffat",
    levels: np.ndarray | Iterable[float] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Make a per-detector PSF radial analysis.

    Each subplot shows for a detector a PSF cutout, a radial analysis and the
    morphology contour, a custom selection of this layer is possible.

    Parameters
    ----------
    cutouts: `dict[str, np.ndarray]`
        A detector's name key dictionary containing
        the 2D array of the star cutouts.
    instrument: `str`, optional
        Detector type. Default 'LSSTComCam'.
    onlyS11: `bool`, optional
        If True, only S11 detectors are shown. Default False.
    layers: `str` or `list` of `str`, optional
        List of layers to be displayed ('background', 'radial', 'contour').
        It is possible to pass also the string 'all' as a shortcut for
        ['background', 'radial', 'contour']. Default 'all'.
    fitModel: `str`, optional
        Model used for the radial fit ('moffat' or 'gauss').
        Default 'moffat'.
    levels: `np.ndarray` or `Iterable` of `float` or `None`, optional
        The levels value for the contour layer.
        If None, is set to `np.linspace(1.5*np.std(data), data.max(), 5)`.
    **kwargs:
        Additional keyword arguments passed to matplotlib Figure constructor.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
        The figure.
    """
    fig = make_figure(**kwargs)
    axsDict = createFigWithInstrumentLayout(fig, instrument, onlyS11=onlyS11)

    if layers == "all":
        layers = ["background", "radial", "contours", "ellipticity"]

    fwhmDict: dict[str, float] = {}
    ee50Dict: dict[str, float] = {}
    ee80Dict: dict[str, float] = {}
    for detName, (cutout, (e1, e2, e)) in cutouts.items():
        fwhm, ee50, ee80 = makeLayerPlot(axsDict[detName], cutout, e1, e2, e, fitModel, layers, levels)
        fwhmDict[detName] = fwhm
        ee50Dict[detName] = ee50
        ee80Dict[detName] = ee80

    cmap = matplotlib.colormaps["RdYlGn_r"]
    for detName, (_, (e1, e2, _)) in cutouts.items():
        bbox = axsDict[detName].get_position()
        bboxArray = bbox.bounds

        axText = fig.add_axes(bboxArray, frameon=True)
        axText.set(zorder=4, facecolor=(1, 1, 1, 0), xticks=[], yticks=[])

        if detName == "cbar":
            continue

        val = (fwhmDict[detName] - min(fwhmDict.values())) / (max(fwhmDict.values()) - min(fwhmDict.values()))
        color = cmap(val)
        axText.patch.set(lw=7, ec=color)
        for _, spine in axText.spines.items():
            spine.set_color(color)

        # Text with FWHM and EE50|80
        text = (
            f'FWHM: {fwhmDict[detName] * 0.2:.2f}" '
            f'EE80: {ee80Dict[detName] * 0.2:.2f}"\n'
            f"E1|2: {e1:.2f}|{e2:.2f}"
        )
        axText.text(
            0.97,
            0.95,
            text,
            color=color,
            fontsize=12,
            fontweight="bold",
            transform=axText.transAxes,
            ha="right",
            va="top",
        )

        axText.text(
            0.3,
            0.1,
            detName,
            color="silver",
            fontsize=11,
            fontweight="bold",
            transform=axText.transAxes,
            ha="right",
            va="top",
        )

    # set colorbar
    vmin = None
    vmax = None
    if fwhmDict:
        vmin = min(fwhmDict.values()) * 0.2
        vmax = max(fwhmDict.values()) * 0.2
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(
            cmap=cmap,
            norm=matplotlib.pyplot.Normalize(vmin=vmin, vmax=vmax),
        ),
        cax=axsDict["cbar"],
    )
    cbar.ax.set_ylabel(ylabel='FWHM"', fontsize=30)
    cbar.ax.tick_params(labelsize=30)

    return fig


def generateCutout(
    butler: Butler,
    imgRef: DatasetRef,
    detector: Detector,
    target: np.ndarray | list[float] | tuple[float, float],
) -> np.ndarray:
    """Generate the cutout around a target position

    Parameters
    ----------
    butler: `lsst.daf.butler.Butler`
        The butler to use to get the image
    imgRef: `lsst.daf.butler.DatasetRef`
        The dataset reference to use to get the image
    detector: `lsst.afw.cameraGeom.Detector`
        The detector to use to get calculate the region of interest
    target: `np.ndarray` or `list` of `float` or `tuple` of `float`
        The coordinates of the cutout center

    Returns
    -------
    cutout: `np.ndarray`
        The square cutout around the center position.
    """

    pad = 20
    detBbox = detector.getBBox()
    start = Point2I(target[0] - (pad // 2), target[1] - (pad // 2))
    dim = Extent2I(pad, pad)
    roiBbox = detBbox.clippedTo(Box2I(start, dim))
    cutout = butler.get(imgRef, parameters={"bbox": roiBbox}).image.array

    return cutout


def findNearestStarToCenter(
    tab: pandas.DataFrame,
    detector: Detector,
    instrument: str,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Find the nearest star w.r.t to the detector center
    N.B. The seacrh is done in PIXEL coordinates.

    Parameters
    ----------
    tab: `pandas.DataFrame`
        pandas.DataFrame with the in focus stars positions.
    detector: `lsst.afw.cameraGeom.Detector`
        The detector realted to the sourceTable's positions.
    instrument: `str`
        Instrument name.

    Returns
    -------
    `np.ndarray`
        The x and y coordinates of the nearest star.
    `tuple`
        The ellipticity parameters (e1, e2, e).
    """

    if instrument == "LSSTComCam":
        xCol = "slot_Centroid_x"
        yCol = "slot_Centroid_y"
    else:  # for now just work with src file that has the same column.
        xCol = "slot_Centroid_x"  # "x"
        yCol = "slot_Centroid_y"  # "y"

    target = (detector.getBBox().centerX, detector.getBBox().centerY)

    tab["center_sep"] = np.sqrt((tab[xCol] - target[0]) ** 2 + (tab[yCol] - target[1]) ** 2)
    most_close = tab.sort_values(by=["center_sep"]).iloc[0].name
    nearest = tab.loc[most_close, [xCol, yCol]].values

    # from makeTableFromSourceCatalogs on lsst.summit_extras.plotting
    iXX = tab.loc[most_close, "slot_Shape_xx"] * (0.2) ** 2
    iYY = tab.loc[most_close, "slot_Shape_yy"] * (0.2) ** 2
    iXY = tab.loc[most_close, "slot_Shape_xy"] * (0.2) ** 2
    T = iXX + iYY
    e1 = (iXX - iYY) / T
    e2 = 2 * iXY / T
    e = np.hypot(e1, e2)

    return nearest, (e1, e2, e)


def makePanel(
    butler: Butler,
    visit: int,
    onlyS11: bool = False,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Create the panel with the in focus stars.
    See the documentation of `makePsfPanel` for more information.

    Parameters
    ----------
    butler: `lsst.daf.butler.Butler`
        The butler to use to get the image and the source table
    imgRefs: `lsst.daf.butler.DatasetRef`
        The dataset reference to use to get the image
    srcRefs: `lsst.daf.butler.DatasetRef`
        The dataset reference to use to get the source table
    onlyS11: `bool`, optional
        If True, only S11 detectors are shown. Default False.
    **kwargs:
        Parameters for the `makePsfPanel` method.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
        The figure.

    Raises
    ------
    ValueError
        If no image or source table datasets are found for the given visit.
    """

    # retrieve the image and source table dataset references
    imgRefs = butler.query_datasets("post_isr_image", where=f"exposure={visit}", explain=False)
    srcRefs = butler.query_datasets("single_visit_psf_star", where=f"exposure={visit}", explain=False)
    if not imgRefs or not srcRefs:
        raise ValueError(f"No image and source tables found for visit {visit}")

    # grab the instrument name from one of the imgRefs
    instrument = imgRefs[0].dataId["instrument"]
    assert isinstance(instrument, str), f"Instrument name {instrument} is not a string"
    camera = getCameraFromInstrumentName(instrument)
    instrumentName = camera.getName()

    # if only S11 then keep only the S11 detectors
    if onlyS11:
        imgRefs = [dr for dr in imgRefs if "S11" in camera[dr.dataId["detector"]].getName()]
        srcRefs = [dr for dr in srcRefs if "S11" in camera[dr.dataId["detector"]].getName()]

    # retrieve the detector object for each detector
    detNameDict = {detector.getName(): detector for detector in camera}

    # interesct the detNum for images and tables
    imgDetName = {camera[dr.dataId["detector"]].getName() for dr in imgRefs}
    srcDetName = {camera[dr.dataId["detector"]].getName() for dr in srcRefs}
    commonDetName = imgDetName.intersection(srcDetName)

    # first retrieve the srcTable datasets from butler
    filterColumn = "calib_psf_used"
    sourceTableDict = {
        camera[dr.dataId["detector"]].getName(): butler.get(dr).to_pandas()
        for dr in srcRefs
        if camera[dr.dataId["detector"]].getName() in commonDetName
    }
    sourceTableDict = {detName: tab[tab[filterColumn]] for detName, tab in sourceTableDict.items()}

    # filter commoDetName to keep only srcTable with non zero rows
    filterDetName = []
    for detName in commonDetName:
        if sourceTableDict[detName].shape[0] > 0:
            filterDetName.append(detName)

    # find the most center star in the srcTables
    candidates = {
        detName: findNearestStarToCenter(sourceTableDict[detName], detNameDict[detName], instrumentName)
        for detName in filterDetName
    }

    # filter the imgRefs
    filterImgRefDict = {}
    for imgRef in imgRefs:
        detName = camera[imgRef.dataId["detector"]].getName()
        if detName in filterDetName:
            filterImgRefDict[detName] = imgRef

    # generate the cutouts
    cutouts = {
        detName: (
            generateCutout(butler, filterImgRefDict[detName], detNameDict[detName], candidates[detName][0]),
            candidates[detName][1],
        )
        for detName in filterDetName
    }

    fig = makePsfPanel(cutouts, instrumentName, onlyS11=onlyS11, **kwargs)
    return fig
