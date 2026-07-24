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

"""The science-package version data model, and its ConsDB round-trip.

The Rapid Analysis backend records, for every image it dispatches, the exact
versions of the handful of science packages that determine the AOS results,
plus a single overall "version number" - an integer that increments whenever
the combined set of package versions changes. The mapping from number to
versions is kept in a small per-instrument JSON registry file, owned and
written solely by Rapid Analysis.

This module is the public home of that data model. It lives here, rather than
in the Rapid Analysis package, because most users cannot import Rapid Analysis
code but do need to consume the data: `PackageVersions` and its
(de)serialisation define the wire format, and the functions below read and
write it in its three homes:

- the ConsDB ``package_versions`` JSONB column (pending: see
  `PACKAGE_VERSIONS_TABLE`), via `writePackageVersionsToConsDb` /
  `readPackageVersionsFromConsDb`;
- the on-disk version-number registry, via `resolveVersionNumber` (Rapid
  Analysis' write path) and `loadPackageVersionRegistry` (everyone else's read
  path);
- and, for data taken before the ConsDB column existed,
  `writeBackdatedPackageVersions` back-fills ConsDB from the registry.

The JSON blob written to ConsDB is::

    {"versions": {"ts_wep": "v17.6.1-alpha", ...}, "version_number": 42}

with ``version_number`` null when it could not be determined. The blob shape is
pinned by tests in both this package and Rapid Analysis; change it only with a
migration plan for the rows already written.
"""

from __future__ import annotations

__all__ = [
    "UNKNOWN_VERSION",
    "UNKNOWN_VERSION_NUMBER",
    "PACKAGE_VERSIONS_TABLE",
    "PACKAGE_VERSIONS_COLUMN",
    "PackageVersions",
    "getRegistryPath",
    "loadPackageVersionRegistry",
    "resolveVersionNumber",
    "writePackageVersionsToConsDb",
    "readPackageVersionsFromConsDb",
    "writeBackdatedPackageVersions",
]

import datetime
import hashlib
import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .utils import computeExposureId

if TYPE_CHECKING:
    from .consdbClient import ConsDbClient

_log = logging.getLogger(__name__)

# Sentinel recorded when a package's version genuinely couldn't be determined
# at processing time. Readers should expect to encounter it.
UNKNOWN_VERSION = "unknown"

# Sentinel overall version number used when the registry couldn't be reached.
# Negative so it can never collide with a real (>= 1) number.
UNKNOWN_VERSION_NUMBER = -1

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
    wire format. The overall version number is the single integer rapid
    analysis assigns to each distinct set of versions (see
    `resolveVersionNumber`); it is `None` when not known, e.g. for an object
    built purely from scraped versions before the registry was consulted.

    Parameters
    ----------
    versions : `dict` [`str`, `str`]
        Mapping of package name to version (a git tag/SHA for git checkouts, an
        installed distribution version for installed packages, or
        `UNKNOWN_VERSION`).
    versionNumber : `int`, optional
        The overall version number for this set of versions.
    """

    versions: dict[str, str]
    versionNumber: int | None = None

    def toDict(self) -> dict[str, Any]:
        """Render as the JSON blob dict stored in the ConsDB column.

        Returns
        -------
        blobDict : `dict` [`str`, `Any`]
            The pinned wire shape: ``{"versions": {...}, "version_number":
            <int or None>}``. ``version_number`` is always present (null when
            unknown) so the blob shape is stable for SQL-side consumers.
        """
        return {"versions": dict(self.versions), "version_number": self.versionNumber}

    @classmethod
    def fromDict(cls, blobDict: Mapping[str, Any]) -> PackageVersions:
        """Build from the JSON blob dict stored in the ConsDB column.

        Parameters
        ----------
        blobDict : `Mapping` [`str`, `Any`]
            The blob, as produced by `toDict`. A missing ``version_number`` key
            is tolerated (treated as null) so blobs written by a future
            shape-reduction still parse.

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
        versionNumber = blobDict.get("version_number")
        return cls(
            versions=dict(blobDict["versions"]),
            versionNumber=int(versionNumber) if versionNumber is not None else None,
        )

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

    def versionHash(self) -> str:
        """A stable, order-independent hash of the package versions.

        Used as the identity of a version set in the registry: two pods with
        the same package versions hash identically and therefore share an
        overall version number, regardless of the order the packages happen to
        be recorded in. ``versionNumber`` is deliberately excluded - the number
        is *derived from* this hash by `resolveVersionNumber`, so including it
        would be circular.

        Returns
        -------
        hexdigest : `str`
            The SHA-256 hex digest of the canonicalised versions.
        """
        canonical = json.dumps(self.versions, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def getRegistryPath(registryDir: str, instrument: str) -> str:
    """Build the path to an instrument's overall-version registry file.

    The registry is kept per-instrument: each rapid analysis head node owns the
    registry for its own instrument, so there is only ever a single writer per
    file.

    Parameters
    ----------
    registryDir : `str`
        The directory holding the registry files.
    instrument : `str`
        The instrument the registry is for, e.g. ``"LSSTCam"``.

    Returns
    -------
    registryPath : `str`
        The path to the registry JSON file.
    """
    return os.path.join(registryDir, f"packageVersionRegistry-{instrument}.json")


def _loadRegistryEntries(registryPath: str) -> list[dict[str, Any]]:
    """Load the list of entries from a registry file, or ``[]`` if absent.

    Parameters
    ----------
    registryPath : `str`
        The path to the registry JSON file.

    Returns
    -------
    entries : `list` [`dict` [`str`, `Any`]]
        The registry entries, or an empty list if the file does not exist.

    Raises
    ------
    ValueError
        Raised if the file's ``entries`` value is not a list.
    """
    if not os.path.isfile(registryPath):
        return []
    with open(registryPath) as f:
        data = json.load(f)
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Malformed registry at {registryPath}: 'entries' is not a list")
    return entries


def _saveRegistryEntries(registryPath: str, entries: list[dict[str, Any]]) -> None:
    """Atomically write the registry entries to ``registryPath``.

    Parameters
    ----------
    registryPath : `str`
        The path to the registry JSON file to write.
    entries : `list` [`dict` [`str`, `Any`]]
        The registry entries to write.
    """
    directory = os.path.dirname(registryPath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmpPath = f"{registryPath}.tmp"
    with open(tmpPath, "w") as f:
        json.dump({"entries": entries}, f, indent=2, sort_keys=True)
    os.replace(tmpPath, registryPath)  # atomic, so a concurrent reader never sees a half-written file


def loadPackageVersionRegistry(registryPath: str) -> dict[int, PackageVersions]:
    """Load a version-number registry file from disk.

    This is the read side of the registry the rapid analysis head node writes
    via `resolveVersionNumber`, mapping each overall version number to the
    package versions it stands for. Use it (with
    `writePackageVersionsToConsDb`) to back-fill ConsDB for data processed
    before the ConsDB column existed - or `writeBackdatedPackageVersions`,
    which composes the two.

    Unlike the never-raising write path, this raises on any problem: backfill
    tooling should fail loudly rather than silently write nothing.

    Parameters
    ----------
    registryPath : `str`
        The path to the registry JSON file, e.g. from `getRegistryPath`.

    Returns
    -------
    registry : `dict` [`int`, `PackageVersions`]
        Mapping of overall version number to its package versions, with each
        `PackageVersions.versionNumber` filled in.

    Raises
    ------
    FileNotFoundError
        Raised if there is no file at ``registryPath``.
    ValueError
        Raised if the file does not have the expected registry shape.
    """
    if not os.path.isfile(registryPath):
        raise FileNotFoundError(f"No package-version registry found at {registryPath}")
    registry: dict[int, PackageVersions] = {}
    for entry in _loadRegistryEntries(registryPath):
        try:
            number = int(entry["number"])
            versions = dict(entry["versions"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Malformed registry entry in {registryPath}: {entry!r}") from e
        registry[number] = PackageVersions(versions=versions, versionNumber=number)
    return registry


def resolveVersionNumber(
    registryPath: str,
    packageVersions: PackageVersions,
    timestamp: str | None = None,
    log: logging.Logger | None = None,
) -> int:
    """Map a set of package versions to its overall, incrementing number.

    The registry at ``registryPath`` records every distinct package-version set
    seen so far and the integer assigned to it. If ``packageVersions`` is
    already known, its existing number is returned. Otherwise it is assigned
    the next integer (one more than the highest seen), recorded in the
    registry, and that number returned.

    This is the registry's only write path, called solely by the rapid analysis
    head node (one process per instrument, hence one writer per file). Robust:
    any problem reading or writing the registry results in a warning and
    `UNKNOWN_VERSION_NUMBER`, never an exception - recording an overall number
    must not be able to disrupt processing.

    Parameters
    ----------
    registryPath : `str`
        The path to the registry JSON file.
    packageVersions : `PackageVersions`
        The versions to look up or record. Only ``versions`` is consulted (via
        `PackageVersions.versionHash`); its ``versionNumber`` is ignored.
    timestamp : `str`, optional
        The timestamp recorded against a newly-assigned number. Defaults to the
        current UTC time; injectable for deterministic tests.
    log : `logging.Logger`, optional
        The logger to use. Defaults to this module's logger.

    Returns
    -------
    number : `int`
        The overall version number (``>= 1``), or `UNKNOWN_VERSION_NUMBER` if
        the registry could not be reached.
    """
    if log is None:
        log = _log

    targetHash = packageVersions.versionHash()
    try:
        entries = _loadRegistryEntries(registryPath)
        for entry in entries:
            if entry.get("hash") == targetHash:
                return int(entry["number"])

        nextNumber = max((int(entry["number"]) for entry in entries), default=0) + 1
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entries.append(
            {
                "number": nextNumber,
                "hash": targetHash,
                "versions": dict(packageVersions.versions),
                "firstSeen": timestamp,
            }
        )
        _saveRegistryEntries(registryPath, entries)
        log.info("Assigned new overall package-version number %d to %s", nextNumber, packageVersions.versions)
        return nextNumber
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        log.warning("Could not resolve the overall package-version number at %s: %s", registryPath, e)
        return UNKNOWN_VERSION_NUMBER


def writePackageVersionsToConsDb(
    client: ConsDbClient,
    instrument: str,
    dayObs: int,
    seqNum: int,
    packageVersions: PackageVersions,
    exposureId: int | None = None,
    controller: str = "O",
    allowUpdate: bool = True,
    table: str = PACKAGE_VERSIONS_TABLE,
    column: str = PACKAGE_VERSIONS_COLUMN,
) -> None:
    """Write the package versions for an exposure to ConsDB.

    The `PackageVersions.toDict` blob is written to the JSONB ``column`` of
    ``cdb_<instrument>.<table>`` for the exposure identified by ``(dayObs,
    seqNum)``. Raises on failure (e.g. the column not existing yet, or the
    server rejecting the write) - callers doing bulk backfill should expect and
    handle that per-row.

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
    packageVersions : `PackageVersions`
        The versions to write.
    exposureId : `int`, optional
        The exposure id. ConsDB requires it in the row values when updating an
        exposure-level table; computed from ``(instrument, controller, dayObs,
        seqNum)`` if not supplied.
    controller : `str`, optional
        The controller letter used when computing ``exposureId``, e.g. ``"O"``
        (the OCS, the default) or ``"C"`` (the CCS). Ignored if ``exposureId``
        is supplied.
    allowUpdate : `bool`, optional
        Allow updating an existing row. Defaults to `True` because the target
        row usually already exists by the time versions are written.
    table : `str`, optional
        The table name within the instrument's schema, without the
        ``cdb_<instrument>.`` prefix. Defaults to `PACKAGE_VERSIONS_TABLE`.
    column : `str`, optional
        The JSONB column to write. Defaults to `PACKAGE_VERSIONS_COLUMN`.
    """
    _checkIsSqlIdentifier(table, "table name")
    _checkIsSqlIdentifier(column, "column name")
    if exposureId is None:
        exposureId = computeExposureId(instrument, controller, dayObs, seqNum)
    client.insert(
        instrument=instrument,
        table=f"cdb_{instrument.lower()}.{table}",
        obs_id=(dayObs, seqNum),  # tuple-form required for updating non ccd-type tables
        values={column: packageVersions.toDict(), "exposure_id": exposureId},
        allow_update=allowUpdate,
    )


def readPackageVersionsFromConsDb(
    client: ConsDbClient,
    instrument: str,
    dayObs: int,
    seqNum: int,
    table: str = PACKAGE_VERSIONS_TABLE,
    column: str = PACKAGE_VERSIONS_COLUMN,
) -> PackageVersions | None:
    """Read the package versions for an exposure back from ConsDB.

    The inverse of `writePackageVersionsToConsDb`. Tolerant of how the JSONB
    column comes back over the query API: both an already-parsed JSON object
    and a JSON string are accepted.

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
    cell = result[column][0]
    if cell is None or cell is np.ma.masked:
        return None
    if isinstance(cell, (str, bytes)):
        return PackageVersions.fromJson(cell)
    if isinstance(cell, Mapping):
        return PackageVersions.fromDict(cell)
    raise TypeError(f"Unexpected type for the {column} column: {type(cell)}")


def writeBackdatedPackageVersions(
    client: ConsDbClient,
    instrument: str,
    dayObs: int,
    seqNum: int,
    versionNumber: int,
    registryPath: str,
    exposureId: int | None = None,
    controller: str = "O",
    allowUpdate: bool = True,
) -> PackageVersions:
    """Back-fill ConsDB with the package versions for an already-taken image.

    For data processed before the ConsDB column existed, where the overall
    version number used for each image is known: the number is expanded to its
    full package-version set via the on-disk registry, and the resulting blob
    written to ConsDB exactly as a live write would have been.

    This re-reads the registry file on every call for simplicity; when
    back-filling many images, load it once with `loadPackageVersionRegistry`
    and call `writePackageVersionsToConsDb` per image instead.

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
    versionNumber : `int`
        The overall version number the image was processed with.
    registryPath : `str`
        The path to the registry JSON file mapping version numbers to package
        versions, e.g. from `getRegistryPath`.
    exposureId : `int`, optional
        The exposure id; computed if not supplied, see
        `writePackageVersionsToConsDb`.
    controller : `str`, optional
        The controller letter used when computing ``exposureId``.
    allowUpdate : `bool`, optional
        Allow updating an existing row. Defaults to `True`.

    Returns
    -------
    packageVersions : `PackageVersions`
        The versions that were written, as resolved from the registry.

    Raises
    ------
    ValueError
        Raised if ``versionNumber`` is not present in the registry.
    """
    registry = loadPackageVersionRegistry(registryPath)
    packageVersions = registry.get(versionNumber)
    if packageVersions is None:
        raise ValueError(
            f"Version number {versionNumber} is not in the registry at {registryPath};"
            f" known numbers: {sorted(registry)}"
        )
    writePackageVersionsToConsDb(
        client,
        instrument,
        dayObs,
        seqNum,
        packageVersions,
        exposureId=exposureId,
        controller=controller,
        allowUpdate=allowUpdate,
    )
    return packageVersions
