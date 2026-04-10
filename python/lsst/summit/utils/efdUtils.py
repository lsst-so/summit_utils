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

import asyncio
import datetime
import logging
import re
from typing import TYPE_CHECKING, Any, Callable

import astropy
import pandas as pd
from astropy import units as u
from astropy.time import Time, TimeDelta
from deprecated.sphinx import deprecated

from lsst.utils.iteration import ensure_iterable

if TYPE_CHECKING:
    from .tmaUtils import TMAEvent
    from lsst.daf.butler import DimensionRecord
    from pandas import DataFrame, Series, Timestamp

from . import dateTime as newLocation
from .utils import getSite

HAS_EFD_CLIENT = True
try:
    from lsst_efd_client import EfdClient
except ImportError:
    HAS_EFD_CLIENT = False


__all__ = [
    "getEfdData",
    "getMostRecentRowWithDataBefore",
    "makeEfdClient",
    "expRecordToTimespan",
    "efdTimestampToAstropy",
    "astropyToEfdTimestamp",
    "clipDataToEvent",
    "calcNextDay",
    "calcDayOffset",
    "getDayObsStartTime",
    "getDayObsEndTime",
    "getDayObsForTime",
    "getTopics",
    "getCommands",
]


COMMAND_ALIASES = {
    "raDecTarget": "lsst.sal.MTPtg.command_raDecTarget",
    "moveToTarget": "lsst.sal.MTMount.command_moveToTarget",
    "startTracking": "lsst.sal.MTMount.command_startTracking",
    "stopTracking": "lsst.sal.MTMount.command_stopTracking",
    "trackTarget": "lsst.sal.MTMount.command_trackTarget",  # issued at 20Hz - don't plot
}

# When looking backwards in time to find the most recent state event, look back
# in chunks of this size. Too small, and there will be too many queries, too
# large and there will be too much data returned unnecessarily, as we only need
# one row by definition. Will tune this parameters in consultation with SQuaRE.
TIME_CHUNKING = datetime.timedelta(minutes=15)


def _getBeginEnd(
    dayObs: int | None = None,
    begin: Time | None = None,
    end: Time | None = None,
    timespan: TimeDelta | None = None,
    event: TMAEvent | None = None,
    expRecord: DimensionRecord | None = None,
) -> tuple[Time, Time]:
    """Calculate the begin and end times to pass to _getEfdData, given the
    kwargs passed to getEfdData.

    Parameters
    ----------
    dayObs : `int`
        The dayObs to query. If specified, this is used to determine the begin
        and end times.
    begin : `astropy.Time`
        The begin time for the query. If specified, either an end time or a
        timespan must be supplied.
    end : `astropy.Time`
        The end time for the query. If specified, a begin time must also be
        supplied.
    timespan : `astropy.TimeDelta`
        The timespan for the query. If specified, a begin time must also be
        supplied.
    event : `lsst.summit.utils.efdUtils.TMAEvent`
        The event to query. If specified, this is used to determine the begin
        and end times, and all other options are disallowed.
    expRecord : `lsst.daf.butler.dimensions.DimensionRecord`
        The exposure record containing the timespan to query. If specified, all
        other options are disallowed.

    Returns
    -------
    begin : `astropy.Time`
        The begin time for the query.
    end : `astropy.Time`
        The end time for the query.
    """
    if expRecord is not None:
        forbiddenOpts = [event, begin, end, timespan, dayObs]
        if any(x is not None for x in forbiddenOpts):
            raise ValueError("You can't specify both an expRecord and a begin/end or timespan or dayObs")
        begin = expRecord.timespan.begin
        end = expRecord.timespan.end
        return begin, end

    if event is not None:
        forbiddenOpts = [begin, end, timespan, dayObs]
        if any(x is not None for x in forbiddenOpts):
            raise ValueError("You can't specify both an event and a begin/end or timespan or dayObs")
        begin = event.begin
        end = event.end
        return begin, end

    # check for dayObs, and that other options aren't inconsistently specified
    if dayObs is not None:
        forbiddenOpts = [begin, end, timespan]
        if any(x is not None for x in forbiddenOpts):
            raise ValueError("You can't specify both a dayObs and a begin/end or timespan")
        begin = newLocation.getDayObsStartTime(dayObs)
        end = newLocation.getDayObsEndTime(dayObs)
        return begin, end
    # can now disregard dayObs entirely

    if begin is None:
        raise ValueError("You must specify either a dayObs or a begin/end or begin/timespan")
    # can now rely on begin, so just need to deal with end/timespan

    if end is None and timespan is None:
        raise ValueError("If you specify a begin, you must specify either a end or a timespan")
    if end is not None and timespan is not None:
        raise ValueError("You can't specify both a end and a timespan")
    if end is None:
        assert timespan is not None
        if timespan > datetime.timedelta(minutes=0):
            end = begin + timespan  # the normal case
        else:
            end = begin  # the case where timespan is negative
            begin = begin + timespan  # adding the negative to the start, i.e. subtracting it to bring back

    assert begin is not None
    assert end is not None
    return begin, end


def getEfdData(
    client: EfdClient,
    topic: str,
    *,
    columns: list[str] | None = None,
    prePadding: float = 0,
    postPadding: float = 0,
    dayObs: int | None = None,
    begin: Time | None = None,
    end: Time | None = None,
    timespan: TimeDelta | None = None,
    event: TMAEvent | None = None,
    expRecord: DimensionRecord | None = None,
    warn: bool = True,
    raiseIfTopicNotInSchema: bool = True,
) -> DataFrame:
    """Get one or more EFD topics over a time range, synchronously.

    The time range can be specified as either:
        * a dayObs, in which case the full 24 hour period is used,
        * a begin point and a end point,
        * a begin point and a timespan.
        * a mount event
        * an exposure record
    If it is desired to use an end time with a timespan, just specify it as the
    begin time and use a negative timespan.

    The results from all topics are merged into a single dataframe.

    `raiseIfTopicNotInSchema` should only be set to `False` when running on the
    summit or in utility code for topics which might have had no data taken
    within the last <data_retention_period> (nominally 30 days). Once a topic
    is in the schema at USDF it will always be there, and thus users there
    never need worry about this, always using `False` will be fine. However, at
    the summit things are a little less predictable, so something missing from
    the schema doesn't necessarily mean a typo, and utility code shouldn't
    raise when data has been expunged.

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    topic : `str`
        The topic to query.
    columns : `list` of `str`, optional
        The columns to query. If not specified, all columns are queried.
    prePadding : `float`
        The amount of time before the nominal start of the query to include, in
        seconds.
    postPadding : `float`
        The amount of extra time after the nominal end of the query to include,
        in seconds.
    dayObs : `int`, optional
        The dayObs to query. If specified, this is used to determine the begin
        and end times.
    begin : `astropy.Time`, optional
        The begin time for the query. If specified, either a end time or a
        timespan must be supplied.
    end : `astropy.Time`, optional
        The end time for the query. If specified, a begin time must also be
        supplied.
    timespan : `astropy.TimeDelta`, optional
        The timespan for the query. If specified, a begin time must also be
        supplied.
    event : `lsst.summit.utils.efdUtils.TMAEvent`, optional
        The event to query. If specified, this is used to determine the begin
        and end times, and all other options are disallowed.
    expRecord : `lsst.daf.butler.dimensions.DimensionRecord`, optional
        The exposure record containing the timespan to query. If specified, all
        other options are disallowed.
    warn : bool, optional
        If ``True``, warn when no data is found. Exists so that utility code
        can disable warnings when checking for data, and therefore defaults to
        ``True``.
    raiseIfTopicNotInSchema : `bool`, optional
        Whether to raise an error if the topic is not in the EFD schema.

    Returns
    -------
    data : `pd.DataFrame`
        The merged data from all topics.

    Raises
    ------
    ValueError:
        If the topics are not in the EFD schema.
    ValueError:
        If both a dayObs and a begin/end or timespan are specified.
    ValueError:
        If a begin time is specified but no end time or timespan.

    """
    # TODO: DM-40100 ideally should calls mpts as necessary so that users
    # needn't care if things are packed

    # supports aliases so that you can query with them. If there is no entry in
    # the alias dict then it queries with the supplied key. The fact the schema
    # is now being checked means this shouldn't be a problem now.

    # TODO: RFC-948 Move this import back to top of file once is implemented.
    import nest_asyncio

    begin, end = _getBeginEnd(dayObs, begin, end, timespan, event, expRecord)
    begin -= TimeDelta(prePadding, format="sec")
    end += TimeDelta(postPadding, format="sec")

    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    ret = loop.run_until_complete(
        _getEfdData(
            client=client,
            topic=topic,
            begin=begin,
            end=end,
            columns=columns,
            raiseIfTopicNotInSchema=raiseIfTopicNotInSchema,
        )
    )
    if ret.empty and warn:
        log = logging.getLogger(__name__)
        msg = ""
        if raiseIfTopicNotInSchema:
            f"Topic {topic} is in the schema, but "
        msg += "no data was returned by the query for the specified time range"
        log.warning(msg)
    return ret


async def _getEfdData(
    client: EfdClient,
    topic: str,
    begin: Time,
    end: Time,
    columns: list[str] | None = None,
    raiseIfTopicNotInSchema: bool = True,
) -> DataFrame:
    """Get data for a topic from the EFD over the specified time range.

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    topic : `str`
        The topic to query.
    begin : `astropy.Time`
        The begin time for the query.
    end : `astropy.Time`
        The end time for the query.
    columns : `list` of `str`, optional
        The columns to query. If not specified, all columns are returned.
    raiseIfTopicNotInSchema : `bool`, optional
        Whether to raise an error if the topic is not in the EFD schema.

    Returns
    -------
    data : `pd.DataFrame`
        The data from the query.
    """
    if columns is None:
        columns = ["*"]
    columns = list(ensure_iterable(columns))

    availableTopics = await client.get_topics()

    if topic not in availableTopics:
        if raiseIfTopicNotInSchema:
            raise ValueError(f"Topic {topic} not in EFD schema")
        else:
            log = logging.getLogger(__name__)
            log.debug(f"Topic {topic} not in EFD schema, returning empty DataFrame")
            return pd.DataFrame()

    data = await client.select_time_series(topic, columns, begin.utc, end.utc)

    return data


def getMostRecentRowWithDataBefore(
    client: EfdClient,
    topic: str,
    timeToLookBefore: Time,
    warnStaleAfterNMinutes: float | int = 60 * 12,
    maxSearchNMinutes: float | int | None = None,
    where: Callable[[DataFrame], list[bool]] | None = None,
    raiseIfTopicNotInSchema: bool = True,
) -> Series:
    """Get the most recent row of data for a topic before a given time.

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    topic : `str`
        The topic to query.
    timeToLookBefore : `astropy.Time`
        The time to look before.
    warnStaleAfterNMinutes : `float`, optional
        The number of minutes after which to consider the data stale and issue
        a warning.
    maxSearchNMinutes: `float` or None, optional
        Maximum number of minutes to search before raising ValueError.
    where: `Callable` or None, optional
        A callable taking a single pd.Dataframe argument and returning a
        boolean list indicating rows to consider.
    raiseIfTopicNotInSchema : `bool`, optional
        Whether to raise an error if the topic is not in the EFD schema.

    Returns
    -------
    row : `pd.Series`
        The row of data from the EFD containing the most recent data before the
        specified time.

    Raises
    ------
    ValueError:
        If the topic is not in the EFD schema.
    """
    staleAge = datetime.timedelta(warnStaleAfterNMinutes)

    earliest = newLocation.getDayObsStartTime(20190101)

    if timeToLookBefore < earliest:
        raise ValueError(f"Requested time {timeToLookBefore} is before any data was put in the EFD")

    if maxSearchNMinutes is not None:
        earliest = max(earliest, timeToLookBefore - maxSearchNMinutes * u.min)

    df = pd.DataFrame()
    beginTime = timeToLookBefore
    while df.empty and beginTime > earliest:
        df = getEfdData(
            client,
            topic,
            begin=beginTime,
            timespan=-TIME_CHUNKING,
            warn=False,
            raiseIfTopicNotInSchema=raiseIfTopicNotInSchema,
        )
        beginTime -= TIME_CHUNKING
        if where is not None and not df.empty:
            df = df[where(df)]

    if beginTime < earliest and df.empty:  # search ended early
        out = f"EFD searched backwards from {timeToLookBefore} to {earliest} and no data "
        if where is not None:
            out += "consistent with `where` predicate "
        out += f"was found in {topic=}"
        raise ValueError(out)

    lastRow = df.iloc[-1]
    commandTime = newLocation.efdTimestampToAstropy(lastRow["private_efdStamp"])

    commandAge = timeToLookBefore - commandTime
    if commandAge > staleAge:
        log = logging.getLogger(__name__)
        log.warning(
            f"Component {topic} was last set {commandAge.sec / 60:.1} minutes before the requested time"
        )

    return lastRow


def makeEfdClient(testing: bool | None = False, databaseName: str | None = None) -> EfdClient:
    """Automatically create an EFD client based on the site.

    Parameters
    ----------
    testing : `bool`
        Set to ``True`` if running in a test suite. This will default to using
        the USDF EFD, for which data has been recorded for replay by the ``vcr`
        package. Note data must be re-recorded to ``vcr`` from both inside and
        outside the USDF when the package/data changes, due to the use of a
        proxy meaning that the web requests are different depending on whether
        the EFD is being contacted from inside and outside the USDF.
    databaseName : `str`, optional
        Name of the database within influxDB to query. If not provided, the
        default specified by EfdClient() is used.

    Returns
    -------
    efdClient : `lsst_efd_client.efd_helper.EfdClient`, optional
        The EFD client to use for the current site.
    """
    efdKwargs: dict[str, Any] = {}
    if databaseName is not None:
        efdKwargs["db_name"] = databaseName

    if not HAS_EFD_CLIENT:
        raise RuntimeError("Could not create EFD client because importing lsst_efd_client failed.")

    if testing:
        return EfdClient("usdf_efd", **efdKwargs)

    site = getSite()
    if site == "local":
        raise RuntimeError(
            "Could not create EFD client: getSite() returned 'local', meaning none of the known"
            " sites could be detected. EFD clients are only available from real Rubin sites."
        )

    if site == "summit":
        return EfdClient("summit_efd", **efdKwargs)
    if site == "tucson":
        return EfdClient("tucson_teststand_efd", **efdKwargs)
    if site == "base":
        return EfdClient("base_efd", **efdKwargs)
    if site in ["staff-rsp", "rubin-devl", "usdf-k8s"]:
        return EfdClient("usdf_efd", **efdKwargs)
    if site == "jenkins":
        return EfdClient("usdf_efd", **efdKwargs)

    raise RuntimeError(f"Could not create EFD client as the {site=} is not recognized")


@deprecated(
    reason="expRecordToTimespan() has moved to lsst.summit.utils.dateTime. The function is unchanged, "
    "but you should change to import from there. This function will be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def expRecordToTimespan(expRecord: DimensionRecord) -> dict:
    """Get the timespan from an exposure record.

    Returns the timespan in a format where it can be used to directly unpack
    into a efdClient.select_time_series() call.

    Parameters
    ----------
    expRecord : `lsst.daf.butler.DimensionRecord`
        The exposure record.

    Returns
    -------
    timespanDict : `dict`
        The timespan in a format that can be used to directly unpack into a
        efdClient.select_time_series() call.
    """
    return newLocation.expRecordToTimespan(expRecord)


@deprecated(
    reason="efdTimestampToAstropy() has moved to lsst.summit.utils.dateTime. The function is unchanged, "
    "but you should change to import from there. This function will be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def efdTimestampToAstropy(timestamp: float) -> Time:
    """Get an efd timestamp as an astropy.time.Time object.

    Parameters
    ----------
    timestamp : `float`
        The timestamp, as a float.

    Returns
    -------
    time : `astropy.time.Time`
        The timestamp as an astropy.time.Time object.
    """
    return newLocation.efdTimestampToAstropy(timestamp)


@deprecated(
    reason="astropyToEfdTimestamp() has moved to lsst.summit.utils.dateTime. The function is unchanged, "
    "but you should change to import from there. This function will be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def astropyToEfdTimestamp(time: Time) -> float:
    """Get astropy Time object as an efd timestamp

    Parameters
    ----------
    time : `astropy.time.Time`
        The time as an astropy.time.Time object.

    Returns
    -------
    timestamp : `float`
        The timestamp, in UTC, in unix seconds.
    """

    return newLocation.astropyToEfdTimestamp(time)


def clipDataToEvent(
    df: DataFrame,
    event: TMAEvent,
    prePadding: float = 0,
    postPadding: float = 0,
    logger: logging.Logger | None = None,
) -> DataFrame:
    """Clip a padded dataframe to an event.

    Parameters
    ----------
    df : `pd.DataFrame`
        The dataframe to clip.
    event : `lsst.summit.utils.efdUtils.TMAEvent`
        The event to clip to.
    prePadding : `float`, optional
        The amount of time before the nominal start of the event to include, in
        seconds.
    postPadding : `float`, optional
        The amount of extra time after the nominal end of the event to include,
        in seconds.
    logger : `logging.Logger`, optional
        The logger to use. If not specified, a new one is created.

    Returns
    -------
    clipped : `pd.DataFrame`
        The clipped dataframe.
    """
    begin = event.begin.value - prePadding
    end = event.end.value + postPadding

    if logger is None:
        logger = logging.getLogger(__name__)

    if begin < df["private_efdStamp"].min():
        logger.warning(f"Requested begin time {begin} is before the start of the data")
    if end > df["private_efdStamp"].max():
        logger.warning(f"Requested end time {end} is after the end of the data")

    mask = (df["private_efdStamp"] >= begin) & (df["private_efdStamp"] <= end)
    clipped_df = df.loc[mask].copy()
    return clipped_df


@deprecated(
    reason="offsetDayObs() has moved to lsst.summit.utils.dateTime. The function is unchanged, "
    "but you should change to import from there. This function will be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def offsetDayObs(dayObs: int, nDays: int) -> int:
    """Offset a dayObs by a given number of days.

    Parameters
    ----------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231225
    nDays : `int`
        The number of days to offset the dayObs by.

    Returns
    -------
    newDayObs : `int`
        The new dayObs, as an integer, e.g. 20231225
    """
    return newLocation.offsetDayObs(dayObs, nDays)


@deprecated(
    reason="calcNextDay() has moved to lsst.summit.utils.dateTime. The function is unchanged, "
    "but you should change to import from there. This function will be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def calcNextDay(dayObs: int) -> int:
    """Given an integer dayObs, calculate the next integer dayObs.

    Integers are used for dayObs, but dayObs values are therefore not
    contiguous due to month/year ends etc, so this utility provides a robust
    way to get the integer dayObs which follows the one specified.

    Parameters
    ----------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231231

    Returns
    -------
    nextDayObs : `int`
        The next dayObs, as an integer, e.g. 20240101
    """
    return newLocation.calcNextDay(dayObs)


@deprecated(
    reason="calcPreviousDay() has moved to lsst.summit.utils.dateTime. The function is unchanged, "
    "but you should change to import from there. This function will be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def calcPreviousDay(dayObs: int) -> int:
    """Given an integer dayObs, calculate the next integer dayObs.

    Integers are used for dayObs, but dayObs values are therefore not
    contiguous due to month/year ends etc, so this utility provides a robust
    way to get the integer dayObs which follows the one specified.

    Parameters
    ----------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231231

    Returns
    -------
    nextDayObs : `int`
        The next dayObs, as an integer, e.g. 20240101
    """
    return newLocation.calcPreviousDay(dayObs)


@deprecated(
    reason="calcDayOffset() has moved to lsst.summit.utils.dateTime. The function is unchanged, "
    "but you should change to import from there. This function will be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def calcDayOffset(startDay: int, endDay: int) -> int:
    """Calculate the number of days between two dayObs values.

    Positive if endDay is after startDay, negative if before, zero if equal.

    Parameters
    ----------
    startDay : `int`
        The starting dayObs, e.g. 20231225.
    endDay : `int`
        The ending dayObs, e.g. 20240101.

    Returns
    -------
    offset : `int`
        The number of days from startDay to endDay.
    """
    return newLocation.calcDayOffset(startDay, endDay)


@deprecated(
    reason="getDayObsStartTime() has moved to lsst.summit.utils.dateTime. The function is unchanged, "
    "but you should change to import from there. This function will be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def getDayObsStartTime(dayObs: int) -> astropy.time.Time:
    """Get the start of the given dayObs as an astropy.time.Time object.

    The observatory rolls the date over at UTC-12.

    Parameters
    ----------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231225

    Returns
    -------
    time : `astropy.time.Time`
        The start of the dayObs as an astropy.time.Time object.
    """
    return newLocation.getDayObsStartTime(dayObs)


@deprecated(
    reason="getDayObsEndTime() has moved to lsst.summit.utils.dateTime. The function is unchanged, "
    "but you should change to import from there. This function will be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def getDayObsEndTime(dayObs: int) -> Time:
    """Get the end of the given dayObs as an astropy.time.Time object.

    Parameters
    ----------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231225

    Returns
    -------
    time : `astropy.time.Time`
        The end of the dayObs as an astropy.time.Time object.
    """
    return newLocation.getDayObsEndTime(dayObs)


@deprecated(
    reason="getDayObsForTime() has moved to lsst.summit.utils.dateTime. The function is unchanged, "
    "but you should change to import from there. This function will be removed after w_2026_01.",
    version="w_2026_01",
    category=FutureWarning,
)
def getDayObsForTime(time: Time) -> int:
    """Get the dayObs in which an astropy.time.Time object falls.

    Parameters
    ----------
    time : `astropy.time.Time`
        The time.

    Returns
    -------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231225
    """
    return newLocation.getDayObsForTime(time)


def getTopics(client: EfdClient, toFind: str, caseSensitive: bool = False) -> list[str]:
    """Return all the strings in topics which match the topic query string.

    Supports wildcards, which are denoted as `*``, as per shell globs.

    Example:
    >>> # assume topics are ['apple', 'banana', 'grape']
    >>> getTopics(, 'a*p*')
    ['apple', 'grape']

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    toFind : `str`
        The query string, with optional wildcards denoted as *.
    caseSensitive : `bool`, optional
        If ``True``, the query is case sensitive. Defaults to ``False``.

    Returns
    -------
    matches : `list` of `str`
        The list of matching topics.
    """
    loop = asyncio.get_event_loop()
    topics = loop.run_until_complete(client.get_topics())

    # Replace wildcard with regex equivalent
    pattern = toFind.replace("*", ".*")
    flags = re.IGNORECASE if not caseSensitive else 0

    matches = []
    for topic in topics:
        if re.match(pattern, topic, flags):
            matches.append(topic)

    return matches


def getCommands(
    client: EfdClient,
    commands: list[str],
    begin: Time,
    end: Time,
    prePadding: float,
    postPadding: float,
    timeFormat: str = "python",
) -> dict[Time | Timestamp | datetime.datetime, str]:
    """Retrieve the commands issued within a specified time range.

    Parameters
    ----------
    client : `EfdClient`
        The client object used to retrieve EFD data.
    commands : `list`
        A list of commands to retrieve.
    begin : `astropy.time.Time`
        The start time of the time range.
    end : `astropy.time.Time`
        The end time of the time range.
    prePadding : `float`
        The amount of time to pad before the begin time.
    postPadding : `float`
        The amount of time to pad after the end time.
    timeFormat : `str`
        One of 'pandas' or 'astropy' or 'python'. If 'pandas', the dictionary
        keys will be pandas timestamps, if 'astropy' they will be astropy times
        and if 'python' they will be python datetimes.

    Returns
    -------
    commandTimes : `dict` [`time`, `str`]
        A dictionary of the times at which the commands where issued. The type
        that `time` takes is determined by the format key, and defaults to
        python datetime.

    Raises
    ------
    ValueError
        Raise if there is already a command at a timestamp in the dictionary,
        i.e. there is a collision.
    """
    if timeFormat not in ["pandas", "astropy", "python"]:
        raise ValueError(f"format must be one of 'pandas', 'astropy' or 'python', not {timeFormat=}")

    commands = list(ensure_iterable(commands))

    commandTimes: dict[Time | Timestamp | datetime.datetime, str] = {}
    for command in commands:
        data = getEfdData(
            client,
            command,
            begin=begin,
            end=end,
            prePadding=prePadding,
            postPadding=postPadding,
            warn=False,  # most commands will not be issue so we expect many empty queries
            raiseIfTopicNotInSchema=False,
        )
        for time, _ in data.iterrows():
            # this is much the most simple data structure, and the chance
            # of commands being *exactly* simultaneous is minimal so try
            # it like this, and just raise if we get collisions for now. So
            # far in testing this seems to be just fine.

            timeKey = None
            match timeFormat:
                case "pandas":
                    timeKey = time
                case "astropy":
                    timeKey = Time(time)
                case "python":
                    assert isinstance(time, pd.Timestamp)
                    timeKey = time.to_pydatetime()

            if timeKey in commandTimes:
                msg = f"There is already a command at {timeKey=} - make a better data structure!"
                msg += f"Colliding commands = {commandTimes[timeKey]} and {command}"
                raise ValueError(msg)
            commandTimes[timeKey] = command
    return commandTimes
