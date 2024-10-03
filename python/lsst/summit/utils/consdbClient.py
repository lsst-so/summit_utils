# This file is part of summit_utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import os

from collections.abc import Mapping
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse

import numpy as np
import requests
from astropy.table import Column, Table, join

__all__ = ["ConsDbClient", "FlexibleMetadataInfo", "getCcdVisitTableForDay", "getWideQuicklookTableForDay"]


logger = logging.getLogger(__name__)


def _urljoin(*args: str) -> str:
    """Join parts of a URL with slashes.

    Does not do any quoting.  Mostly to remove a level of list-making.

    Parameters
    ----------
    *args : `str`
        Each parameter is a URL part.

    Returns
    -------
    url : `str`
        The joined URL.
    """
    return "/".join(args)


def _check_status(r: requests.Response) -> None:
    """Check the status of an HTTP response and raise if an error.

    Adds additional response information to the raise_for_status exception.

    Parameters
    ----------
    r : `requests.Response`
        The response to check.

    Raises
    ------
    requests.HTTPError
        Raised if a non-successful status is returned.
    """
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        try:
            json_data = e.response.json()
            e.add_note(str(json_data))
            if "message" in json_data:
                e.add_note(f"\n\n{json_data['message']}")
        except requests.JSONDecodeError:
            pass
        raise e


def clean_url(resp: requests.Response, *args, **kwargs) -> requests.Response:
    """Parse url from response and remove netloc portion.

    Set new url in response and return response

    Parameters
    ----------
    resp : `requests.Response`
        The response that could contain a URL with tokens
    """
    url = urlparse(resp.url)
    short_user = f"{url.username[:2]}***" if url.username is not None else ""
    short_pass = f":{url.password[:2]}***" if url.password is not None else ""
    netloc = f"{short_user}{short_pass}@{url.hostname}"
    resp.url = url._replace(netloc=netloc).geturl()
    return resp


@dataclass
class FlexibleMetadataInfo:
    """Description of a flexible metadata value.

    Parameters
    ----------
    dtype : `str`
        Data type of the flexible metadata value.
        One of ``bool``, ``int``, ``float``, or ``str``.
    doc : `str`
        Documentation string.
    unit : `str`, optional
        Unit of the value.
    ucd : `str`, optional
        IVOA Unified Content Descriptor
        (https://www.ivoa.net/documents/UCD1+/).
    """

    dtype: str
    doc: str
    unit: str | None = None
    ucd: str | None = None


class ConsDbClient:
    """A client library for accessing the Consolidated Database.

    This library provides a basic interface for using flexible metadata
    (key/value pairs associated with observation ids from an observation
    type table), determining the schema of ConsDB tables, querying the
    ConsDB using a general SQL SELECT statement, and inserting into
    ConsDB tables.

    Parameters
    ----------
    url : `str`, optional
        Base URL of the Web service, defaults to the value of environment
        variable ``LSST_CONSDB_PQ_URL`` (the location of the publish/query
        service).
    token : `str`, optional
        Authentication token for the RSP. The token must begin with "gt-".

    Notes
    -----
    This client is a thin layer over the publish/query Web service, which
    avoids having a dependency on database drivers.

    It enforces the return of query results as Astropy Tables.
    """

    def __init__(self, url: str | None = None, token: str | None = None):
        self.session = requests.Session()
        self.session.hooks["response"].append(clean_url)

        if token is not None:
            if not token.startswith("gt-"):
                raise ValueError("token must start with `gt-`.")

            self.session.headers.update({"Authorization": f"Bearer {token}"})

        if url is None:
            self.url = os.environ["LSST_CONSDB_PQ_URL"]
        else:
            self.url = url
        self.url = self.url.rstrip("/")

    def _handle_get(self, url: str, query: dict[str, str | list[str]] | None = None) -> Any:
        """Submit GET requests to the server.

        Parameters
        ----------
        url : `str`
            URL to GET.
        query : `dict` [`str`, `str` | `list` [`str`]], optional
            Query parameters to attach to the URL.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        requests.JSONDecodeError
            Raised if the result does not decode as JSON.

        Returns
        -------
        result : `Any`
            Result of decoding the Web service result content as JSON.
        """
        logger.debug(f"GET {url}")
        response = self.session.get(url, params=query)
        _check_status(response)
        return response.json()

    def _handle_post(self, url: str, data: dict[str, Any]) -> requests.Response:
        """Submit POST requests to the server.

        Parameters
        ----------
        url : `str`
            URL to POST.
        data : `dict` [`str`, `Any`]
            Key/value pairs of data to POST.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.

        Returns
        -------
        result : `requests.Response`
            The raw Web service result object.
        """
        logger.debug(f"POST {url}: {data}")
        response = self.session.post(url, json=data)
        _check_status(response)
        return response

    @staticmethod
    def compute_flexible_metadata_table_name(instrument: str, obs_type: str) -> str:
        """Compute the name of a flexible metadata table.

        Each instrument and observation type made with that instrument can
        have a flexible metadata table.  This function is useful when
        issuing SQL queries, and it avoids a round-trip to the server.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        obs_type : `str`
            Name of the observation type (e.g. ``Exposure``).

        Returns
        -------
        table_name : `str`
            Name of the appropriate flexible metadata table.
        """
        return f"cdb_{instrument}.{obs_type}_flexdata"

    @staticmethod
    def compute_fixed_metadata_namespace(instrument: str) -> str:
        """Compute the namespace for a fixed metadata table.

        Each instrument has its own namespace in the ConsDB.
        This function is useful when issuing SQL queries, and it avoids a
        round-trip to the server.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).

        Returns
        -------
        namespace_name : `str`
            Name of the appropriate namespace
        """
        return f"cdb_{instrument}"

    def add_flexible_metadata_key(
        self,
        instrument: str,
        obs_type: str,
        key: str,
        dtype: str,
        doc: str,
        unit: str | None = None,
        ucd: str | None = None,
    ) -> requests.Response:
        """Add a key to a flexible metadata table.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        obs_type : `str`
            Name of the observation type (e.g. ``Exposure``).
        key : `str`
            Name of the key to be added (must not already exist).
        dtype : `str`
            One of ``bool``, ``int``, ``float``, or ``str``.
        doc : `str`
            Documentation string for the key.
        unit : `str`, optional
            Unit for the value. Should be from the IVOA
            (https://www.ivoa.net/documents/VOUnits/) or astropy.
        ucd : `str`, optional
            IVOA Unified Content Descriptor
            (https://www.ivoa.net/documents/UCD1+/).

        Returns
        -------
        response : `requests.Response`
            HTTP response from the server, with 200 status for success.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        data = {"key": key, "dtype": dtype, "doc": doc}
        if unit is not None:
            data["unit"] = unit
        if ucd is not None:
            data["ucd"] = ucd
        url = _urljoin(self.url, "flex", quote(instrument), quote(obs_type), "addkey")
        return self._handle_post(url, data)

    def get_flexible_metadata_keys(self, instrument: str, obs_type: str) -> dict[str, FlexibleMetadataInfo]:
        """Retrieve the valid keys for a flexible metadata table.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        obs_type : `str`
            Name of the observation type (e.g. ``Exposure``).

        Returns
        -------
        key_info : `dict` [ `str`, `FlexibleMetadataInfo` ]
            Dict of keys and information values.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        url = _urljoin(self.url, "flex", quote(instrument), quote(obs_type), "schema")
        result = self._handle_get(url)
        return {key: FlexibleMetadataInfo(*value) for key, value in result.items()}

    def get_flexible_metadata(
        self, instrument: str, obs_type: str, obs_id: int, keys: list[str] | None = None
    ) -> dict[str, Any]:
        """Get the flexible metadata for an observation.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        obs_type : `str`
            Name of the observation type (e.g. ``Exposure``).
        obs_id : `int`
            Unique observation id.
        keys : `list` [ `str` ], optional
            List of keys to be retrieved; all if not specified.

        Returns
        -------
        result_dict : `dict` [ `str`, `Any` ]
            Dictionary of key/value pairs for the observation.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        url = _urljoin(
            self.url,
            "flex",
            quote(instrument),
            quote(obs_type),
            "obs",
            quote(str(obs_id)),
        )
        return self._handle_get(url, {"k": keys} if keys else None)

    def get_all_metadata(
        self, instrument: str, obs_type: str, obs_id: int, flex: bool = False
    ) -> dict[str, Any]:
        """Get all metadata for an observation.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        obs_type : `str`
            Name of the observation type (e.g. ``Exposure``).
        obs_id : `int`
            Unique observation id.
        flex : `bool`
            Include flexible metadata.

        Returns
        -------
        result_dict : `dict` [ `str`, `Any` ]
            Dictionary of key/value pairs for the observation.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        url = _urljoin(
            self.url,
            "query",
            quote(instrument),
            quote(obs_type),
            "obs",
            quote(str(obs_id)),
        )
        return self._handle_get(url, {"flex": "1"} if flex else None)

    def insert_flexible_metadata(
        self,
        instrument: str,
        obs_type: str,
        obs_id: int,
        values: dict[str, Any] | None = None,
        *,
        allow_update: bool = False,
        **kwargs,
    ) -> requests.Response:
        """Set flexible metadata values for an observation.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        obs_type : `str`
            Name of the observation type (e.g. ``Exposure``).
        obs_id : `int`
            Unique observation id.
        values : `dict` [ `str`, `Any` ], optional
            Dictionary of key/value pairs to add for the observation.
        allow_update : `bool`, optional
            If ``True``, allow replacement of values of existing keys.
        **kwargs : `dict`
            Additional key/value pairs, overriding ``values``.

        Returns
        -------
        response : `requests.Response`
            HTTP response from the server, with 200 status for success.

        Raises
        ------
        ValueError
            Raised if no values are provided in ``values`` or kwargs.
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        if values:
            values.update(kwargs)
        else:
            values = kwargs
        if not values:
            raise ValueError(f"No values to set for {instrument} {obs_type} {obs_id}")
        data = {"values": values}
        url = _urljoin(
            self.url,
            "flex",
            quote(instrument),
            quote(obs_type),
            "obs",
            quote(str(obs_id)),
        )
        if allow_update:
            url += "?u=1"
        return self._handle_post(url, data)

    def insert(
        self,
        instrument: str,
        table: str,
        obs_id: tuple[int, int] | int,
        values: Mapping[str, Any],
        *,
        allow_update: bool = False,
        **kwargs,
    ) -> requests.Response:
        """Insert values into a single ConsDB fixed metadata table.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        table : `str`
            Name of the table to insert into.
        obs_id : `tuple` [ `int`, `int`] or `int`
            Unique observation id or day_obs and seq_num.
        values : `Mapping` [ `str`, `Any` ]
            Dictionary-like mapping of column/value pairs to add for the
            observation.
        allow_update : `bool`, optional
            If ``True``, allow replacement of values of existing columns.
        **kwargs : `dict`
            Additional column/value pairs, overriding ``values``.

        Returns
        -------
        response : `requests.Response`
            HTTP response from the server, with 200 status for success.

        Raises
        ------
        ValueError
            Raised if no values are provided in ``values`` or kwargs.
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        # Build a new merged dict to avoid mutating the incoming Mapping.
        merged_values: dict[str, Any] = {**(dict(values) if values else {}), **kwargs}
        if not merged_values:
            raise ValueError(f"No values to insert for {instrument} {table} {obs_id}")

        data: dict[str, Any]
        if isinstance(obs_id, tuple):
            data = {"table": table, "values": merged_values}
            url = _urljoin(
                self.url,
                "insert",
                quote(instrument),
                quote(table),
                "by_seq_num",
                quote(str(obs_id[0])),
                quote(str(obs_id[1])),
            )
        else:
            data = {"table": table, "obs_id": obs_id, "values": merged_values}
            url = _urljoin(
                self.url,
                "insert",
                quote(instrument),
                quote(table),
                "obs",
                quote(str(obs_id)),
            )
        if allow_update:
            url += "?u=1"
        return self._handle_post(url, data)

    def insert_multiple(
        self,
        instrument: str,
        table: str,
        obs_dict: dict[int, dict[str, Any]],
        *,
        allow_update=False,
    ) -> requests.Response:
        """Insert values into a single ConsDB fixed metadata table.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        table : `str`
            Name of the table to insert into.
        obs_dict : `dict` [ `int`, `dict` [ `str`, `Any` ] ]
            Dictionary of observation ids, each with a dictionary of
            column/value pairs to add for each observation.
        allow_update : `bool`, optional
            If ``True``, allow replacement of values of existing columns.

        Returns
        -------
        response : `requests.Response`
            HTTP response from the server, with 200 status for success.

        Raises
        ------
        ValueError
            Raised if no values are provided in ``obs_dict``.
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        if not obs_dict:
            raise ValueError(f"No values to insert for {instrument} {table}")
        data = {"table": table, "obs_dict": obs_dict}
        url = _urljoin(
            self.url,
            "insert",
            quote(instrument),
            quote(table),
        )
        if allow_update:
            url += "?u=1"
        return self._handle_post(url, data)

    def query(self, query: str) -> Table:
        """Query the ConsDB database.

        Parameters
        ----------
        query : `str`
            A SQL query (currently) to the database.

        Returns
        -------
        result : `Table`
            An ``astropy.Table`` containing the query results.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.

        Notes
        -----
        This is a very general query interface because it is expected that
        a wide variety of types of queries will be needed. If some types prove
        to be common, syntactic sugar could be added to make them simpler.
        """
        url = _urljoin(self.url, "query")
        data = {"query": query}
        result = self._handle_post(url, data).json()

        columns = result.get("columns", [])
        if not columns:
            # No result columns
            return Table(rows=[])

        rows = result.get("data", [])
        if not rows:
            # No result rows
            return Table(names=columns)

        return Table(rows=rows, names=columns)

    def wait_for_row_to_exist(self, query: str, timeout: float, poll_frequency_hz: float = 2) -> Table:
        """Returns a row once it exists, or an empty table if it times out.

        The supplied ``query`` must be expected to return exactly a single row,
        (once it exists), e.g. it should be something like
        'select * from cdb_latiss.exposure where exposure_id = 2024100200541'
        or similar. If the query were like
        'select * from cdb_latiss.exposure where exposure_id in (2024100200541,
        2024100200542)', then the query would return multiple rows and an error
        would be raised. The user is expected to check that the query meets
        this criterion, because if 2024100200541 existed but 2024100200542 was
        about to be created the error would not be raised, and downstream
        beaviour would be undefined.

        Parameters
        ----------
        query : `str`
            A SQL query (currently) to the database.
        timeout : `float`
            Maximum time to wait for a non-empty result, in seconds.
        poll_frequency_hz : `float`, optional
            Frequency to poll the database for results, in Hz.

        Returns
        -------
        result : `Table`
            An ``astropy.Table`` containing the query results, or an empty
            if the row was not inserted before the timeout.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        ValueError
            Raised if the query returns more than one row.
        """
        sleep_duration = 1 / poll_frequency_hz
        t0 = time.time()
        while time.time() - t0 < timeout:
            result = self.query(query)
            if len(result) > 1:
                raise ValueError(f"Query {query} returned more than one row")
            elif len(result) == 1:
                return result
            time.sleep(sleep_duration)

        logger.warning(f"Query {query} did not return any results within {timeout}s")
        return Table(rows=[])

    def wait_for_item_in_row(
        self, query: str, item: str, timeout: float, poll_frequency_hz: float = 2
    ) -> float | None:
        """Returns the value of an item in a row once it exists, or ``None``
        if it times out.

        If the item is not in the schema of the table, an error will be raised.

        The supplied ``query`` must be expected to return exactly a single row,
        (once it exists), e.g. it should be something like
        'select * from cdb_latiss.exposure where exposure_id = 2024100200541'
        or similar. If the query were like
        'select * from cdb_latiss.exposure where exposure_id in (2024100200541,
        2024100200542)', then the query would return multiple rows and an error
        would be raised. The user is expected to check that the query meets
        this criterion, because if 2024100200541 existed but 2024100200542 was
        about to be created the error would not be raised, and downstream
        beaviour would be undefined.

        Parameters
        ----------
        query : `str`
            A SQL query (currently) to the database.
        item : `str`
            The item to check for in the query results.
        timeout : `float`
            Maximum time to wait for a non-empty result, in seconds.
        poll_frequency_hz : `float`, optional
            Frequency to poll the database for results, in Hz.

        Returns
        -------
        value : `float` or `None`
            The corresponding value of the item in the row in the table, or
            ``None`` if the item was not found before the timeout.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        ValueError
            Raised if the query returns more than one row, or if the requested
            item is not in the schema of the table.
        """

        row = self.wait_for_row_to_exist(query, timeout, poll_frequency_hz)
        if len(row) == 0:
            # wait_for_row_to_exist already logged a warning if table is empty
            return None

        # we know the row now exists but the required item may not be there yet
        sleep_duration = 1 / poll_frequency_hz
        t0 = time.time()
        while time.time() - t0 < timeout:
            result = self.query(query)
            if len(result) > 1:
                raise ValueError(f"Query {query} returned more than one row")
            assert len(result) == 1, "Somehow no rows came back, which should be impossible"
            row = result[0]
            if item not in row.columns:
                raise ValueError(f"Query {query} did not return a column named {item} - check the schema")
            value = result[0][item]
            if value is not None:
                return value
            time.sleep(sleep_duration)

        logger.warning(
            f"The row returned by {query} did not end up containing a value for {item} within {timeout}s"
        )
        return None

    def schema(
        self, instrument: str | None = None, table: str | None = None
    ) -> dict[str, tuple[str, str]] | list[str]:
        """Retrieve information about ConsDB.

        If ``instrument`` and ``table`` are given, return the schema of a
        fixed metadata table in ConsDB.

        If only ``instrument`` is given, return the names of all tables
        for that instrument.

        If no arguments are given, return the names of all instruments.

        Parameters
        ----------
        instrument : `str`, optional
            Name of the instrument (e.g. ``LATISS``).
        table : `str`, optional
            Name of the table to insert into.

        Returns
        -------
        info : `list` [ `str` ] or `dict` [ `str`, `tuple` [ `str`, `str` ] ]
            A list of instrument strings or table names, or else a dict of
            columns with values that are tuples containing a data type string
            and a documentation string.

        Raises
        ------
        ValueError
            Raised if only ``table`` is given.
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.

        Notes
        -----
        Fixed metadata data types may use the full database vocabulary,
        unlike flexible metadata data types.
        """
        if instrument is None:
            if table is not None:
                raise ValueError("Must specify instrument if table is given")
            url = _urljoin(self.url, "schema")
        elif table is None:
            url = _urljoin(self.url, "schema", quote(instrument))
        else:
            url = _urljoin(self.url, "schema", quote(instrument), quote(table))
        result = self._handle_get(url)
        if instrument is not None and table is not None:
            return {key: (str(value[0]), str(value[1])) for key, value in result.items()}
        else:
            return [str(value) for value in result]


def getCcdVisitTableForDay(
    client: ConsDbClient,
    dayObs: int,
    visitTableItems: list[str] | None = None,
    detectors: list[int] | None = None,
    withZeropoint: bool = False,
) -> Table:
    """Get the ccdvisit1_quicklook table for a given dayObs.

    Parameters
    ----------
    client : `ConsDbClient`
        The ConsDbClient to use.
    dayObs : `int`
        The dayObs to query for.
    visitTableItems : `list` of `str`, optional
        Additional items from the visit1 table to include.
    detectors : `list` of `int`, optional
        If given, only return rows for these detectors.
    withZeropoint : `bool`, optional
        If ``True``, only return rows with a non-null zeropoint.

    Returns
    -------
    table : `astropy.table.Table`
        The resulting table.
    """
    extraVisit: str = ", " + ", ".join(f"v.{item}" for item in visitTableItems) if visitTableItems else ""
    query = (
        "SELECT cvq.*, "
        "cv.detector, cv.visit_id, "
        f"v.band, v.exp_time, v.seq_num, v.day_obs, v.img_type{extraVisit} "
        "FROM cdb_LSSTCam.ccdvisit1_quicklook as cvq, "
        "cdb_LSSTCam.ccdvisit1 as cv, "
        "cdb_LSSTCam.visit1 as v "
    )
    where = f"WHERE cvq.ccdvisit_id=cv.ccdvisit_id and cv.visit_id=v.visit_id and v.day_obs={dayObs}"
    if detectors:
        where += f" and detector in ({','.join([str(d) for d in detectors])})"
    if withZeropoint:
        where += " and cvq.zero_point is not null"

    table = client.query(query + where)
    return table


def columnsEqual(a: Column, b: Column) -> bool:
    """Check if two columns are equal, taking masks into account.

    Parameters
    ----------
    a : `Column`
        First column to compare.
    b : `Column`
        Second column to compare.

    Returns
    -------
    equal : `bool`
        True if the columns are equal, False otherwise.
    """
    aArr = np.asanyarray(a)
    bArr = np.asanyarray(b)
    if aArr.shape != bArr.shape:
        return False

    aMask = getattr(a, "mask", None)
    bMask = getattr(b, "mask", None)

    if aMask is None and bMask is None:
        return bool(np.all(aArr == bArr))

    if aMask is None:
        aMask = np.zeros(aArr.shape, dtype=bool)
    if bMask is None:
        bMask = np.zeros(bArr.shape, dtype=bool)

    aMaskArr = np.asanyarray(aMask)
    bMaskArr = np.asanyarray(bMask)

    if np.any(aMaskArr ^ bMaskArr):
        return False  # one masked where the other isn't

    present = ~aMaskArr
    return bool(np.all(aArr[present] == bArr[present]))


def getWideQuicklookTableForDay(client: ConsDbClient, dayObs: int) -> Table:
    """Get a wide quicklook table for a given dayObs.

    Joins all columns from the visit1 table to the visit1_quicklook table. Note
    that the visit1 table already contains all the columns from the exposure
    table, and is just keyed by the exposure_id instead of the visit_id.

    Parameters
    ----------
    client : `ConsDbClient`
        The ConsDbClient to use.
    dayObs : `int`
        The dayObs to query for.

    Returns
    -------
    table : `astropy.table.Table`
        The resulting wide quicklook table.
    """
    vq = client.query(f"SELECT * FROM cdb_LSSTCam.visit1_quicklook WHERE day_obs = {dayObs}")
    v = client.query(f"SELECT * FROM cdb_LSSTCam.visit1 WHERE day_obs = {dayObs}")

    wide: Table = join(
        vq,
        v,
        keys="visit_id",
        join_type="inner",
        table_names=("vq", "v"),
        uniq_col_name="{col_name}_{table_name}",  # only duplicates get suffixed
    )

    duplicated = sorted((set(vq.colnames) & set(v.colnames)) - {"visit_id"})
    for name in duplicated:
        left = f"{name}_vq"
        right = f"{name}_v"
        if left not in wide.colnames or right not in wide.colnames:
            continue

        if not columnsEqual(wide[left], wide[right]):
            raise ValueError(f"Column '{name}' differs between tables for dayObs={dayObs}")

        wide[name] = wide[left]
        wide.remove_columns([left, right])

    return wide
