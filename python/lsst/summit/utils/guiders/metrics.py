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

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.stats import mad_std

from lsst.summit.utils.utils import RobustFitResult, RobustFitter

__all__ = ["GuiderMetricsBuilder"]


class GuiderMetricsBuilder:
    """
    Measure and organize guider performance metrics for a given exposure.

    This class wraps the computation of both exposure-level counts (number of
    guiders, stars, measurements, fraction of valid stamps) and per-quantity
    trend metrics (ALT drift, AZ drift, rotator, photometry, PSF). Trend
    metrics include slope, intercept, trend RMSE, global scatter, outlier
    fraction, slope significance, and sample size.

    Parameters
    ----------
    starCatalog : `pandas.DataFrame`
        Catalog of guider star measurements, containing at least the columns
        required for the counts and trend metrics: ``expid``, ``elapsed_time``,
        and the measurement columns for each metric (e.g., ``dalt``, ``daz``,
        ``dtheta``, ``magoffset``, ``fwhm``).
    """

    def __init__(self, starCatalog: pd.DataFrame, nMissingStamps: int) -> None:
        self.starCatalog = starCatalog
        self.log = logging.getLogger(__name__)
        self.nMissingStamps = nMissingStamps

        # Store the basic variable names for metrics
        self.baseVarsCols = {
            "altDrift": "dalt",
            "azDrift": "daz",
            "mag": "magoffset",
            "rotator": "dtheta",
            "psf": "fwhm",
        }
        self.baseVars = list(self.baseVarsCols.keys())

        # keep track if the metrics were build
        self.isBuilt = False

    def buildMetrics(self, expid: int) -> pd.DataFrame:
        """
        Compute all metrics for the specified exposure ID.

        Parameters
        ----------
        expid : `int`
            Exposure ID to compute metrics for.

        Returns
        -------
        metricsDf : `pandas.DataFrame`
            Single-row DataFrame with all computed metrics for the specified
            exposure ID. Columns include exposure counts (e.g., ``n_guiders``,
            ``n_stars``, per-guider flags) and each metric prefix
            (``alt_drift``, ``az_drift``, ``rotator``, ``mag``, ``psf``)
            expanded with its statistic names.
        """
        self.expid = expid
        stars = self.starCatalog

        # early exit if no data
        mask = stars["expid"].eq(expid)
        if not mask.any():
            self.isBuilt = False
            self.log.warning(f"No data found for expid={expid}. Returning empty metrics DataFrame.")
            return pd.DataFrame(columns=self.metricsColumns)

        # build metrics
        self.countsDf = computeExposureCounts(stars, self.nMissingStamps, expid)
        self.altDriftData: GuiderDriftResult = computeTrendMetrics(stars, "elapsed_time", "dalt", expid)
        self.azDriftData: GuiderDriftResult = computeTrendMetrics(stars, "elapsed_time", "daz", expid)
        self.rotatorData: GuiderDriftResult = computeTrendMetrics(stars, "elapsed_time", "dtheta", expid)
        self.magData: GuiderDriftResult = computeTrendMetrics(stars, "elapsed_time", "magoffset", expid)
        self.psfData: GuiderDriftResult = computeTrendMetrics(stars, "elapsed_time", "fwhm", expid)

        # Set the built state to true
        self.isBuilt = True

        # build MetricResult objects to a DataFrame
        return self.toDataFrame()

    def detrendStars(self) -> pd.DataFrame:
        """
        Detrend the star catalog using the computed slopes.

        This method modifies the `starCatalog` in place, adding new columns
        with detrended values for each input column. The detrended columns
        are named with a '_corr' suffix (e.g., 'dalt_corr', 'daz_corr').

        Returns
        -------
        stars : `pandas.DataFrame`
            The modified star catalog with detrended measurement columns.
        """
        if not self.isBuilt:
            raise RuntimeError("Metrics have not been built. Call buildMetrics(expid) first.")
        # Create a copy to avoid modifying the original catalog
        stars = self.starCatalog.copy()
        # Get the metrics DataFrame
        metrics = self.toDataFrame()

        _prefixCol = list(self.baseVarsCols.keys())
        prefixCol = [toSnakeCase(p) for p in _prefixCol]

        starsDetrended = detrendBaseVariables(stars, metrics, prefixCol)
        starsDetrended = detrendFocalPlaneVariables(starsDetrended, metrics)
        return starsDetrended

    def toDataFrame(self) -> pd.DataFrame:
        """
        Assemble all computed counts and trend metrics into a single row.

        Returns
        -------
        metricsDf : `pandas.DataFrame`
            DataFrame with one row for the currently set ``expid``. Columns
            include exposure counts (e.g., ``n_guiders``, ``n_stars``,
            per-guider flags) and each metric prefix (``alt_drift``,
            ``az_drift``, ``rotator``, ``mag``, ``psf``) expanded with its
            statistic names.
        """
        if not self.isBuilt:
            raise RuntimeError("Metrics have not been built. Call buildMetrics(expid) first.")

        listDf = [
            self.countsDf,
            self.azDriftData.toDataFrame("az_drift", index=self.expid),
            self.altDriftData.toDataFrame("alt_drift", index=self.expid),
            self.rotatorData.toDataFrame("rotator", index=self.expid),
            self.magData.toDataFrame("mag", index=self.expid),
            self.psfData.toDataFrame("psf", index=self.expid),
        ]
        return pd.concat(listDf, axis=1)

    @property
    def metricsColumns(self) -> list[str]:
        """
        List of expected output column names for the metrics DataFrame.

        Combines the base count columns and each metric prefix with all
        statistic suffixes.

        Returns
        -------
        columns : `list` of `str`
            All column names in the order they will appear in the DataFrame
            returned by ``toDataFrame()`` or ``buildMetrics()``.
        """
        baseCols = ["n_guiders", "n_stars", "n_measurements", "fraction_possible_measurements"]
        statVars = [
            "slope",
            "intercept",
            "trend_rmse",
            "global_std",
            "outlier_frac",
            "slope_significance",
            "nsize",
        ]
        columns = baseCols[:]
        for var in self.baseVars:
            for stat in statVars:
                varSnakeCase = toSnakeCase(var)
                columns.append(f"{varSnakeCase}_{stat}")
        return columns

    def printSummary(self) -> None:
        """
        Print a human-readable summary of all metrics.

        Each metric's slope is scaled by the exposure time.

        """
        # Guard if buildMetrics found no data
        if not self.isBuilt:
            raise RuntimeError("Metrics have not been built. Call buildMetrics(expid) first.")

        # set units (ensure consistency with y units!)
        self.altDriftData.units = "arcsec"
        self.azDriftData.units = "arcsec"
        self.rotatorData.units = "arcsec"
        self.magData.units = "mag"
        self.psfData.units = "arcsec"

        exptime = self.countsDf["exptime"].values[0]

        # Print summaries
        header1 = makeHeader("Guider Metrics Summary")
        print("\n".join(header1))
        print("Exposure ID:", self.expid)
        print(f"Exposure time: {exptime:.2f} sec")
        printExposureCounts(self.countsDf)

        self.azDriftData.pprint("Az")
        self.altDriftData.pprint("Alt")
        self.rotatorData.pprint("Rotator")
        self.magData.pprint("Mag")
        self.psfData.pprint("PSF FWHM")


def computeTrendMetrics(
    stars: pd.DataFrame,
    timeCol: str,
    yCol: str,
    expid: int,
) -> GuiderDriftResult:
    """
    Compute robust linear trend metrics for a given measurement column versus
    time within a single exposure.

    The function fits a robust linear model to the specified `yCol` as a
    function of `timeCol` for rows matching the given `expid`. It returns a
    `MetricResult` containing the slope, intercept, trend RMSE, robust global
    scatter, outlier fraction, slope significance, and sample size.

    Parameters
    ----------
    stars : `pandas.DataFrame`
        Table of star measurements containing at least the columns `timeCol`,
        `yCol`, and `expid`.
    timeCol : `str`
        Name of the time column (e.g., ``"elapsed_time"``).
    yCol : `str`
        Name of the dependent variable column to fit (e.g., ``"dalt"``,
        ``"magoffset"``).
    expid : `int`
        Exposure identifier used to filter the rows.

    Returns
    -------
    metrics : `MetricResult`
        Dataclass containing the computed trend metrics. If there are no data
        in `yCol` after filtering, all fields are set to NaN/None and `nsize`
        is zero.

    Raises
    ------
    KeyError
        If `timeCol` or `yCol` are not present in `stars`.
    """
    s = stars.loc[stars["expid"].eq(expid), [timeCol, yCol, "exptime"]].dropna()
    if s.empty or s[yCol].nunique() < 2:
        empty_mask = np.zeros((0,), dtype=bool)
        return GuiderDriftResult(
            fit=RobustFitResult(
                slope=np.nan,
                intercept=np.nan,
                scatter=np.nan,
                outlierMask=empty_mask,
                slopePValue=np.nan,
                slopeStdErr=np.nan,
                slopeTValue=np.nan,
                interceptPValue=np.nan,
                interceptStdErr=np.nan,
                interceptTValue=np.nan,
            ),
            globalStd=np.nan,
            nsize=0,
            units="",
            exptime=np.nan,
        )

    x = s[timeCol].to_numpy()
    y = s[yCol].to_numpy()
    exptime = s["exptime"].max()
    global_std = float(mad_std(y))

    fitter = RobustFitter()
    fit_res = fitter.fit(x, y, residualThreshold=3.0)

    return GuiderDriftResult(
        fit=fit_res,
        globalStd=global_std,
        nsize=int(y.size),
        units="",
        exptime=exptime,
    )


@dataclass(slots=True)
class GuiderDriftResult:
    """
    Metrics for guider data derived from a robust linear trend fit.

    This dataclass wraps a `RobustFitResult` with guider-specific
    fields. It stores the global scatter, number of valid points,
    and domain metadata such as units and exposure time. Properties
    provide easy access to slope, intercept, trend RMSE, outlier
    fraction, and slope significance.

    Parameters
    ----------
    fit : `RobustFitResult`
        The result of the robust fit containing slope and intercept.
    globalStd : `float`
        Robust global standard deviation of the dependent values.
    nsize : `int`
        Number of valid points used in the fit.
    units : `str`, optional
        Units of the dependent variable. Default is empty string.
    exptime : `float`, optional
        Exposure time in seconds. Default is 1.0.
    """

    fit: RobustFitResult  # composition, not duplication
    globalStd: float  # robust global std of y
    nsize: int
    units: str = ""
    exptime: float = 1.0

    def __post_init__(self) -> None:
        assert self.fit is not None, "fit must be provided"

    @property
    def slope(self) -> float:
        return self.fit.slope

    @property
    def intercept(self) -> float:
        return self.fit.intercept

    @property
    def trendRmse(self) -> float:
        return self.fit.scatter

    @property
    def outlierFrac(self) -> float:
        m = self.fit.outlierMask
        return float(np.count_nonzero(m)) / float(m.size) if m.size else np.nan

    @property
    def slopeSignificance(self) -> float | None:
        t = self.fit.slopeTValue
        return abs(float(t)) if t is not None else None

    def toDataFrame(self, prefix: str, index: int = 0) -> pd.DataFrame:
        """
        Convert the stored metrics into a single-row DataFrame.

        Parameters
        ----------
        prefix : `str`
            Prefix to add to each metric's column name in the output.
        index : `int`, optional
            Index value for the returned DataFrame row.

        Returns
        -------
        metrics : `pandas.DataFrame`
            Single-row DataFrame containing the numeric/statistical fields of
            this result, prefixed with `prefix`.
        """
        row = {
            f"{prefix}_slope": self.slope,
            f"{prefix}_intercept": self.intercept,
            f"{prefix}_trend_rmse": self.trendRmse,
            f"{prefix}_global_std": self.globalStd,
            f"{prefix}_outlier_frac": self.outlierFrac,
            f"{prefix}_slope_significance": self.slopeSignificance,
            f"{prefix}_nsize": self.nsize,
        }
        return pd.DataFrame([row], index=[index])

    def pprint(self, title: str) -> None:
        """
        Print the stored metrics in a formatted, human-readable block.

        Parameters
        ----------
        title : `str`
            Title to display for the metric block.
        """
        if not title:
            title = "Metric"
        units = self.units
        exptime = self.exptime

        header = makeHeader(f"Metrics Summary: {title}", nchar=40)
        print("\n".join(header))
        slope_per_exp = self.slope * exptime
        print(f"  Slope          : {slope_per_exp:.3f} {units} per exposure")
        sig = "—" if self.slopeSignificance is None else f"{self.slopeSignificance:.1f}"
        print(f"  Slope signif.  : {sig} sigma")
        print(f"  Intercept      : {self.intercept:.3f} {units}")
        print(f"  Trend RMSE     : {self.trendRmse:.3f} {units}")
        print(f"  Global std     : {self.globalStd:.3f} {units}")
        print(f"  Outlier frac   : {self.outlierFrac:.2%}")
        print(f"  N (points)     : {self.nsize:d}\n")


def detrendFocalPlaneVariables(
    stars: pd.DataFrame,
    metricsDf: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detrend focal plane measurement columns in the star catalog using slopes
    from Alt/Az drift metrics. Project the Alt/Az slopes onto the focal plane
    to correct the dX and dyfp measurements.

    This function modifies the input `stars` DataFrame in place, subtracting
    the projected linear trend (slope * elapsed_time) from the dX and dyfp
    columns. The slopes are taken from the `metricsDf` for the corresponding
    `expid`.

    Parameters
    ----------
    stars : `pandas.DataFrame`
        DataFrame containing star measurements with at least the columns
        ``expid``, ``elapsed_time``, ``dX``, ``dyfp``, and guider position
        columns ``alt``, ``az``.
    metricsDf : `pandas.DataFrame`
        DataFrame containing computed metrics with slope columns named as
        ``alt_drift_slope`` and ``az_drift_slope``.

    Returns
    -------
    stars : `pandas.DataFrame`
        The modified input DataFrame with detrended dX and dyfp columns.
    """
    # Validate metrics columns
    alt_slope_col = "alt_drift_slope"
    az_slope_col = "az_drift_slope"

    # Extract slopes (Alt/Az per second)
    s_alt = float(metricsDf[alt_slope_col].values[0])
    s_az = float(metricsDf[az_slope_col].values[0])

    # Validate required columns in stars
    required_cols = ["dxfp", "dyfp", "elapsed_time", "dalt", "daz"]
    for col in required_cols:
        if col not in stars.columns:
            raise KeyError(f"Required column '{col}' not found in stars DataFrame.")

    # Build design matrices using only finite rows
    A = stars[["dalt", "daz"]].to_numpy(dtype=float)
    B = stars[["dxfp", "dyfp"]].to_numpy(dtype=float)
    finite_mask = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    A = A[finite_mask]
    B = B[finite_mask]

    if A.shape[0] < 3:
        # Not enough information to estimate a stable 2x2 mapping
        raise ValueError("Insufficient finite samples to estimate focal-plane projection (need >= 3 rows).")

    # Solve for the best-fit 2x2 linear mapping M such that A @ M ≈ B
    # Uses least squares across both X and Y simultaneously
    # np.linalg.lstsq returns the matrix that minimizes ||A M - B||_F
    M, *_ = np.linalg.lstsq(A, B, rcond=None)

    # Project Alt/Az slopes into focal plane slopes
    s_fp = M @ np.array([s_alt, s_az], dtype=float)
    s_xfp = float(s_fp[0])
    s_yfp = float(s_fp[1])

    # Apply detrending:
    # new_value = original_value - projected_slope * elapsed_time
    t = stars["elapsed_time"].to_numpy(dtype=float)
    dxfp = stars["dxfp"].to_numpy(dtype=float) - s_xfp * t
    stars["dxfp_corr"] = dxfp - np.nanmedian(dxfp)

    dyfp = stars["dyfp"].to_numpy(dtype=float) - s_yfp * t
    stars["dyfp_corr"] = dyfp - np.nanmedian(dyfp)

    return stars


def detrendBaseVariables(
    stars: pd.DataFrame,
    metricsDf: pd.DataFrame,
    prefixCol: list[str],
) -> pd.DataFrame:
    """
    Detrend specified measurement columns in the star catalog using slopes
    from the metrics DataFrame.

    This function modifies the input `stars` DataFrame in place, subtracting
    the linear trend (slope * elapsed_time) from each specified measurement
    column. The slopes are taken from the `metricsDf` for the corresponding
    `expid`.

    Parameters
    ----------
    stars : `pandas.DataFrame`
        DataFrame containing star measurements with at least the columns
        ``expid``, ``elapsed_time``, and the measurement columns to detrend.
    metricsDf : `pandas.DataFrame`
        DataFrame containing computed metrics with slope columns named as
        ``{prefix}_slope`` for each prefix in `prefixCol`.
    prefixCol : `list` of `str`
        List of prefixes corresponding to measurement columns to detrend.
        For example, if ``"alt_drift"`` is in this list, the function will
        look for a column named ``"alt_drift_slope"`` in `metricsDf` and a
        column named ``"dalt"`` in `stars`.

    Returns
    -------
    stars : `pandas.DataFrame`
        The modified input DataFrame with detrended measurement columns.
    """
    for prefix in prefixCol:
        slopeCol = f"{prefix}_slope"
        if slopeCol not in metricsDf.columns:
            raise KeyError(f"Slope column '{slopeCol}' not found in metricsDf.")
        slope = metricsDf[slopeCol].values[0]

        # Determine the corresponding measurement column in stars
        if prefix == "alt_drift":
            measCol = "dalt"
        elif prefix == "az_drift":
            measCol = "daz"
        elif prefix == "rotator":
            measCol = "dtheta"
        elif prefix == "mag":
            measCol = "magoffset"
        elif prefix == "psf":
            measCol = "fwhm"
        else:
            raise ValueError(f"Unknown prefix '{prefix}'. Cannot determine measurement column.")

        if measCol not in stars.columns:
            raise KeyError(f"Measurement column '{measCol}' not found in stars.")

        # Apply detrending: new_value = original_value - slope * elapsed_time
        det = stars[measCol] - slope * stars["elapsed_time"]
        stars[measCol + "_corr"] = det - np.nanmedian(det)

    return stars


def toSnakeCase(name: str) -> str:
    """
    Convert a camelCase or PascalCase string to snake_case.

    Parameters
    ----------
    name : `str`
        Input string in camelCase or PascalCase.

    Returns
    -------
    snake_case : `str`
        Converted string in snake_case.
    """
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


def computeExposureCounts(stars: pd.DataFrame, nMissingStamps: int, expid: int) -> pd.DataFrame:
    """
    Compute guider/star/measurement counts for the given expid.

    Parameters
    ----------
    stars : `pandas.DataFrame`
        Star measurement rows.
    nMissingStamps : `int`
        Number of missing stamps for the exposure.
    expid : `int`
        Exposure ID to filter on.

    Returns
    -------
    countsDf : `pandas.DataFrame`
        Single-row DataFrame with counts for the exposure.
    """
    s = stars.loc[stars["expid"].eq(expid)]
    exptime = s["exptime"].max() if not s.empty else np.nan

    # Guiders and stars per guider
    nGuiders = s["detector"].nunique()
    nUnique = s["detid"].nunique()
    nMissing = nMissingStamps
    counts = s.groupby("detector")["detid"].nunique().to_dict()
    guiderNames = sorted(s["detector"].unique())
    guidersPresent = {f"{det}": (counts.get(det, 0) > 0) for det in guiderNames}

    # Valid measurements
    maskValid = (s["stamp"] >= 0) & (s["xccd"].notna())
    nMeas = int(maskValid.sum())

    # Fraction of valid stamps (protect against div-by-zero)
    nStamps = s["stamp"].nunique()
    totalPossible = nGuiders * nStamps
    fracValid = (nMeas / totalPossible) if totalPossible > 0 else np.nan

    row = {
        "n_guiders": nGuiders,
        "n_stars": nUnique,
        "n_missing_stamps": int(nMissing),
        "n_measurements": nMeas,
        "fraction_possible_measurements": fracValid,
        "exptime": exptime,
    }
    row.update(guidersPresent)
    return pd.DataFrame([row], index=[expid])


def printExposureCounts(countsDf: pd.DataFrame, precision: int = 3) -> None:
    """
    Print exposure-level counts from a single-row counts DataFrame.

    Parameters
    ----------
    countsDf : `pandas.DataFrame`
        DataFrame with a single row containing exposure counts (e.g.,
        n_guiders, n_stars, per-guider flags).
    precision : `int`, optional
        Number of decimal places for fractional values.

    """
    row = countsDf.iloc[0].to_dict()

    lines = makeHeader("Exposure Counts")
    lines += [
        f"Tracked stars: {int(row.get('n_stars', 0))}",
        f"Missing stamps: {int(row.get('n_missing_stamps', 0))}",
        f"Measurements: {int(row.get('n_measurements', 0))}",
    ]
    frac = row.get("fraction_possible_measurements")
    if isinstance(frac, (int, float, np.floating)):
        lines.append(f"Possible meas. frac: {float(frac):.{precision}f}")

    # Per-guider boolean flags if present
    guider_flags = [k for k in row.keys() if k.startswith("R") and "_SG" in k]
    if guider_flags:
        present = [g for g in sorted(guider_flags) if bool(row.get(g))]
        lines.append(f"Guiders used: {', '.join(present) if present else '—'}")

    print("\n".join(lines))
    print()


def makeHeader(title: str, nchar: int = 40) -> list[str]:
    """
    Create a formatted header block with horizontal lines.

    The header consists of a top line, a centered title line with padding, and
    a bottom line. The line width is the maximum of `nchar` and the title
    length plus 10 characters.

    Parameters
    ----------
    title : `str`
        Text to display in the header.
    nchar : `int`, optional
        Minimum width of the horizontal lines. Increased automatically if the
        title requires more space.

    Returns
    -------
    header_lines : `list` of `str`
        List of three strings: the top line, the title line, and the bottom
        line.
    """
    width = max(nchar, len(title) + 10)
    line = "─" * width
    header = f"{' ' * 5}{title}{' ' * 5}"
    return [line, header, line]


def makeLine(nchar: int = 40) -> str:
    """
    Create a horizontal line of box-drawing characters.

    Parameters
    ----------
    nchar : `int`, optional
        Number of characters in the line. Default is 40.

    Returns
    -------
    line : `str`
        String consisting of `nchar` repetitions of the '─' character.
    """
    return "─" * nchar
