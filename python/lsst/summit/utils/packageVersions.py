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

"""The science-package version data model, and reading it back from ConsDB.

The Rapid Analysis backend records, for every image it dispatches, the exact
versions of the handful of science packages that determine the AOS results. A
set of versions is identified by a content hash (a stable, order-independent
SHA-256 of the versions) - two pods with the same versions hash identically.

This module is the public home of that data model. It lives here, rather than
in Rapid Analysis, because most users cannot import Rapid Analysis code but do
need to consume the data: everything needed to read a record back is carried in
ConsDB and this package, so a consumer can recover the full versions with only
this module and a ConsDB client.

The JSON blob stored in the ConsDB ``package_versions`` column is::

    {"hash": "<sha256hex>", "versions": {"ts_wep": "v17.6.1-alpha", ...}}

``versions`` is inline (self-contained; no dereference needed) and ``hash`` is
the stable identity, handy as a SQL-side join/dedup key. The blob shape is
`PackageVersions.toDict` / `fromDict`, pinned by tests in both this package and
Rapid Analysis; change it only with a migration plan for the rows already
written.

This module is deliberately read-only with respect to ConsDB: it is the data
model plus `readPackageVersionsFromConsDb`, so an ordinary user can recover the
versions with nothing but a ConsDB client. Scraping the versions and writing
them to ConsDB both live in Rapid Analysis, which reuses the shape (`toDict`)
and the table/column constants here, so the two sides cannot drift.
"""

from __future__ import annotations

__all__ = [
    "UNKNOWN_VERSION",
    "PACKAGE_VERSIONS_TABLE",
    "PACKAGE_VERSIONS_COLUMN",
    "PackageVersions",
    "readPackageVersionsFromConsDb",
    "readPackageVersionsForExposure",
    "readPackageVersionsByHash",
]

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lsst.daf.butler import DimensionRecord

    from .consdbClient import ConsDbClient

# Sentinel recorded when a package's version couldn't be determined at
# processing time. Readers must handle it, but it is not expected from a
# healthy pod: Rapid Analysis' CI treats a missing package directory as a hard
# failure, but it could show up if e.g. rows are written from a non-git
# environment (e.g. a conda-installed stack, where the package directory isn't
# a checkout) and in rows backfilled from older metadata. Note that it takes
# part in `PackageVersions.versionHash`, so a set containing it is a distinct
# identity from the same set with the real version, and the two will not join
# or dedup.
UNKNOWN_VERSION = "unknown"

# The anticipated home of the package-versions JSONB column in ConsDB, as
# ``cdb_<instrument>.<PACKAGE_VERSIONS_TABLE>.<PACKAGE_VERSIONS_COLUMN>``.
# The column does not exist yet; these defaults are parameterised on every
# function that uses them so they can be corrected without code changes if
# the schema lands elsewhere.
PACKAGE_VERSIONS_TABLE = "exposure_quicklook"
PACKAGE_VERSIONS_COLUMN = "package_versions"

# Everything interpolated into raw SQL by readPackageVersionsFromConsDb must
# look like a plain identifier; anything else is rejected.
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _checkIsSqlIdentifier(value: str, description: str) -> None:
    """Raise if ``value`` is not usable as a bare SQL identifier.

    Parameters
    ----------
    value : `str`
        The value to check.
    description : `str`
        What the value is, for the error message, e.g. ``"table name"``.
    """
    if not _SQL_IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid {description}: {value!r}")


@dataclass
class PackageVersions:
    """The versions of the tracked science packages at processing time.

    The versions are held in a dict keyed by package name rather than as named
    fields so that the set of recorded packages can grow without changing the
    wire format. A set of versions is identified by its `versionHash`, which is
    also the key under which Rapid Analysis stores it.

    Parameters
    ----------
    versions : `dict` [`str`, `str`]
        Mapping of package name to version (a git tag/SHA for git checkouts, an
        installed distribution version for installed packages, or
        `UNKNOWN_VERSION`).
    """

    versions: dict[str, str]

    def versionHash(self) -> str:
        """A stable, order-independent hash of the package versions.

        This is the identity of a version set: two pods with the same package
        versions hash identically regardless of the order the packages happen
        to be recorded in, so the hash is a stable dedup/join key in ConsDB.

        Returns
        -------
        hexdigest : `str`
            The SHA-256 hex digest of the canonicalised versions.
        """
        canonical = json.dumps(self.versions, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def toDict(self) -> dict[str, Any]:
        """Render as the JSON blob dict stored in the ConsDB column.

        Returns
        -------
        blobDict : `dict` [`str`, `Any`]
            The pinned wire shape: ``{"hash": ..., "versions": {...}}``.
            ``versions`` is inline so the blob is self-contained; ``hash`` is
            included as a stable SQL-side identity/dedup key.
        """
        return {"hash": self.versionHash(), "versions": dict(self.versions)}

    @classmethod
    def fromDict(cls, blobDict: Mapping[str, Any]) -> PackageVersions:
        """Build from the JSON blob dict stored in the ConsDB column.

        The ``hash`` key, if present, is ignored: ``versions`` is the source of
        truth and the hash is derived from it (recompute with `versionHash`).

        Parameters
        ----------
        blobDict : `Mapping` [`str`, `Any`]
            The blob, as produced by `toDict`.

        Returns
        -------
        packageVersions : `PackageVersions`
            The parsed versions.

        Raises
        ------
        ValueError
            Raised if the blob has no ``versions`` key.
        """
        if "versions" not in blobDict:
            raise ValueError(f"Package-version blob has no 'versions' key: {dict(blobDict)!r}")
        return cls(versions=dict(blobDict["versions"]))

    def toJson(self) -> str:
        """Render as a canonical JSON string.

        Returns
        -------
        jsonString : `str`
            The `toDict` blob serialised with sorted keys, so identical version
            sets always serialise identically.
        """
        return json.dumps(self.toDict(), sort_keys=True)

    @classmethod
    def fromJson(cls, jsonString: str | bytes) -> PackageVersions:
        """Build from a JSON string, as produced by `toJson`.

        Parameters
        ----------
        jsonString : `str` or `bytes`
            The JSON document to parse.

        Returns
        -------
        packageVersions : `PackageVersions`
            The parsed versions.

        Raises
        ------
        ValueError
            Raised if the document is not valid JSON, or parses to something
            other than the expected blob shape.
        """
        parsed = json.loads(jsonString)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object for a package-version blob, got {type(parsed)}")
        return cls.fromDict(parsed)


def readPackageVersionsFromConsDb(
    client: ConsDbClient,
    instrument: str,
    dayObs: int,
    seqNum: int,
    table: str = PACKAGE_VERSIONS_TABLE,
    column: str = PACKAGE_VERSIONS_COLUMN,
) -> PackageVersions | None:
    """Read the package versions for an exposure back from ConsDB.

    Reads back the blob Rapid Analysis writes. Tolerant of how the JSONB column
    is delivered over the query API: both an already-parsed JSON object and a
    JSON string are accepted.

    Parameters
    ----------
    client : `ConsDbClient`
        The client used to talk to ConsDB.
    instrument : `str`
        The instrument name, e.g. ``"LSSTCam"``.
    dayObs : `int`
        The dayObs of the exposure.
    seqNum : `int`
        The seqNum of the exposure.
    table : `str`, optional
        The table name within the instrument's schema, without the
        ``cdb_<instrument>.`` prefix. Defaults to `PACKAGE_VERSIONS_TABLE`.
    column : `str`, optional
        The JSONB column to read. Defaults to `PACKAGE_VERSIONS_COLUMN`.

    Returns
    -------
    packageVersions : `PackageVersions` or `None`
        The versions recorded for the exposure, or `None` if the exposure has
        no row in the table or the column is null.

    Raises
    ------
    RuntimeError
        Raised if more than one row matches, which should be impossible.
    TypeError
        Raised if the column value is of an unexpected type.
    """
    _checkIsSqlIdentifier(instrument, "instrument name")
    _checkIsSqlIdentifier(table, "table name")
    _checkIsSqlIdentifier(column, "column name")
    schemaName = f"cdb_{instrument.lower()}"
    # Join via the exposure table rather than assuming day_obs/seq_num are
    # denormalised onto the quicklook table.
    query = (
        f"SELECT q.{column} "
        f"FROM {schemaName}.{table} q "
        f"JOIN {schemaName}.exposure e ON e.exposure_id = q.exposure_id "
        f"WHERE e.day_obs = {int(dayObs)} AND e.seq_num = {int(seqNum)}"
    )
    result = client.query(query)
    if len(result) == 0:
        return None
    if len(result) > 1:
        raise RuntimeError(
            f"Got {len(result)} package-version rows for {instrument} ({dayObs=}, {seqNum=});"
            " expected at most one"
        )
    return _packageVersionsFromCell(result[column][0], column)


def readPackageVersionsForExposure(
    client: ConsDbClient,
    expRecord: DimensionRecord,
    table: str = PACKAGE_VERSIONS_TABLE,
    column: str = PACKAGE_VERSIONS_COLUMN,
) -> PackageVersions | None:
    """Read the package versions for an exposure record back from ConsDB.

    Convenience wrapper over `readPackageVersionsFromConsDb` that takes the
    dataId straight off an exposure ``DimensionRecord`` instead of its unpacked
    ``instrument``/``dayObs``/``seqNum``.

    Parameters
    ----------
    client : `ConsDbClient`
        The client used to talk to ConsDB.
    expRecord : `lsst.daf.butler.DimensionRecord`
        The exposure record whose versions to read.
    table : `str`, optional
        The table name within the instrument's schema, without the
        ``cdb_<instrument>.`` prefix. Defaults to `PACKAGE_VERSIONS_TABLE`.
    column : `str`, optional
        The JSONB column to read. Defaults to `PACKAGE_VERSIONS_COLUMN`.

    Returns
    -------
    packageVersions : `PackageVersions` or `None`
        The versions recorded for the exposure, or `None` if there is no row or
        the column is null.
    """
    return readPackageVersionsFromConsDb(
        client,
        expRecord.instrument,
        expRecord.day_obs,
        expRecord.seq_num,
        table=table,
        column=column,
    )


def readPackageVersionsByHash(
    client: ConsDbClient,
    instrument: str,
    versionHash: str,
    table: str = PACKAGE_VERSIONS_TABLE,
    column: str = PACKAGE_VERSIONS_COLUMN,
) -> PackageVersions | None:
    """Read the package versions for a given version hash from ConsDB.

    A hash identifies a version *set*, which can appear on many exposures, so
    this returns the versions from an arbitrary matching row - they are
    identical across every row that shares the hash - or `None` if no row
    carries the hash yet. Matches on the ``hash`` key of the JSONB blob, which
    `PackageVersions.toDict` writes alongside the inline versions.

    Parameters
    ----------
    client : `ConsDbClient`
        The client used to talk to ConsDB.
    instrument : `str`
        The instrument whose schema to search, e.g. ``"LSSTCam"``.
    versionHash : `str`
        The `PackageVersions.versionHash` to look up (a hex digest).
    table : `str`, optional
        The table name within the instrument's schema, without the
        ``cdb_<instrument>.`` prefix. Defaults to `PACKAGE_VERSIONS_TABLE`.
    column : `str`, optional
        The JSONB column to read. Defaults to `PACKAGE_VERSIONS_COLUMN`.

    Returns
    -------
    packageVersions : `PackageVersions` or `None`
        The versions with that hash, or `None` if no row carries it.

    Raises
    ------
    ValueError
        Raised if ``versionHash`` is not a bare hex digest (it is interpolated
        into the query as a string literal, so it is validated first).
    TypeError
        Raised if the column value is of an unexpected type.
    """
    _checkIsSqlIdentifier(instrument, "instrument name")
    _checkIsSqlIdentifier(table, "table name")
    _checkIsSqlIdentifier(column, "column name")
    if not re.fullmatch(r"[0-9a-fA-F]+", versionHash):
        raise ValueError(f"Invalid version hash: {versionHash!r}")
    # versionHash() always produces lowercase hex and the SQL comparison is
    # case-sensitive, so normalise: an uppercase paste of a valid hash must
    # find the same row rather than silently returning None.
    versionHash = versionHash.lower()
    schemaName = f"cdb_{instrument.lower()}"
    # ``->>'hash'`` extracts the blob's hash key (see toDict); LIMIT 1 because
    # the versions are identical across every row that shares the hash.
    query = (
        f"SELECT q.{column} "
        f"FROM {schemaName}.{table} q "
        f"WHERE q.{column}->>'hash' = '{versionHash}' "
        "LIMIT 1"
    )
    result = client.query(query)
    if len(result) == 0:
        return None
    return _packageVersionsFromCell(result[column][0], column)


def _packageVersionsFromCell(cell: Any, column: str) -> PackageVersions | None:
    """Parse a ConsDB ``package_versions`` cell into a `PackageVersions`.

    Tolerant of how the JSONB column is delivered over the query API: a null
    (or masked) cell yields `None`, an already-parsed JSON object or a JSON
    string both yield a `PackageVersions`.

    Parameters
    ----------
    cell : `Any`
        The raw cell value from the query result.
    column : `str`
        The column name, for the error message only.

    Returns
    -------
    packageVersions : `PackageVersions` or `None`
        The parsed versions, or `None` if the cell is null.

    Raises
    ------
    TypeError
        Raised if the cell value is of an unexpected type.
    """
    if cell is None or cell is np.ma.masked:
        return None
    if isinstance(cell, (str, bytes)):
        return PackageVersions.fromJson(cell)
    if isinstance(cell, Mapping):
        return PackageVersions.fromDict(cell)
    raise TypeError(f"Unexpected type for the {column} column: {type(cell)}")
