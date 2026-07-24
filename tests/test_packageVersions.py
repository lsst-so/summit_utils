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

"""Tests for the science-package version data model and its ConsDB round-trip.

Two wire formats are pinned here and must not drift without a migration plan:
the ConsDB JSON blob (``{"versions": {...}, "version_number": N}``) and the
on-disk version-number registry written by the Rapid Analysis head node
(``{"entries": [{"number", "hash", "versions", "firstSeen"}]}``) - deployed
registry files and any already-written ConsDB rows use these shapes, so the
verbatim-decode tests below fail if either changes.
"""

import json
import os
from pathlib import Path

import pytest
import responses

from lsst.summit.utils import ConsDbClient
from lsst.summit.utils.packageVersions import (
    PACKAGE_VERSIONS_COLUMN,
    UNKNOWN_VERSION_NUMBER,
    PackageVersions,
    getRegistryPath,
    loadPackageVersionRegistry,
    readPackageVersionsFromConsDb,
    resolveVersionNumber,
    writeBackdatedPackageVersions,
    writePackageVersionsToConsDb,
)
from lsst.summit.utils.utils import computeExposureId

VERSIONS = {"ts_wep": "v17.6.1-alpha", "donut_viz": "v4.4.0-alpha", "rubintv_production": "abc123"}


@pytest.fixture
def client() -> ConsDbClient:
    """A ConsDbClient pointed at a fake url; use @responses.activate to mock
    the connection.
    """
    return ConsDbClient("http://example.com/consdb")


def test_toDictPinsBlobShape() -> None:
    # this is the wire format written to the ConsDB JSONB column - an exact
    # match, not a subset check, so any shape drift fails here
    pv = PackageVersions(versions=dict(VERSIONS), versionNumber=42)
    assert pv.toDict() == {"versions": VERSIONS, "version_number": 42}
    # version_number is always present, null when unknown
    assert PackageVersions(versions=dict(VERSIONS)).toDict() == {"versions": VERSIONS, "version_number": None}


def test_dictRoundTrip() -> None:
    pv = PackageVersions(versions=dict(VERSIONS), versionNumber=42)
    assert PackageVersions.fromDict(pv.toDict()) == pv
    pvNoNumber = PackageVersions(versions=dict(VERSIONS))
    assert PackageVersions.fromDict(pvNoNumber.toDict()) == pvNoNumber


def test_jsonRoundTrip() -> None:
    pv = PackageVersions(versions=dict(VERSIONS), versionNumber=42)
    assert PackageVersions.fromJson(pv.toJson()) == pv
    # canonical: sorted keys, so identical sets serialise identically
    reordered = PackageVersions(versions=dict(reversed(list(VERSIONS.items()))), versionNumber=42)
    assert reordered.toJson() == pv.toJson()


def test_fromDictToleratesMissingVersionNumber() -> None:
    # a blob without the version_number key parses as unknown, not an error
    pv = PackageVersions.fromDict({"versions": VERSIONS})
    assert pv.versions == VERSIONS
    assert pv.versionNumber is None


def test_fromDictRejectsMissingVersions() -> None:
    with pytest.raises(ValueError, match="no 'versions' key"):
        PackageVersions.fromDict({"version_number": 42})


def test_fromJsonRejectsNonObject() -> None:
    with pytest.raises(ValueError, match="Expected a JSON object"):
        PackageVersions.fromJson('["not", "an", "object"]')


def test_versionHashStableAndOrderIndependent() -> None:
    a = PackageVersions(versions={"ts_wep": "v1", "donut_viz": "v2"})
    b = PackageVersions(versions={"donut_viz": "v2", "ts_wep": "v1"})  # different insertion order
    assert a.versionHash() == b.versionHash()


def test_versionHashIgnoresVersionNumber() -> None:
    # the number is derived from the hash, so it must not feed back into it
    a = PackageVersions(versions={"ts_wep": "v1"}, versionNumber=1)
    b = PackageVersions(versions={"ts_wep": "v1"}, versionNumber=99)
    assert a.versionHash() == b.versionHash()


def test_versionHashChangesWhenAVersionChanges() -> None:
    a = PackageVersions(versions={"ts_wep": "v1", "donut_viz": "v2"})
    c = PackageVersions(versions={"ts_wep": "v1", "donut_viz": "v3"})
    assert a.versionHash() != c.versionHash()


def test_getRegistryPathIsPerInstrument() -> None:
    path = getRegistryPath("/some/dir", "LSSTCam")
    assert path == "/some/dir/packageVersionRegistry-LSSTCam.json"
    assert path != getRegistryPath("/some/dir", "LSSTComCam")


def test_resolveVersionNumberFirstSetGetsOne(tmp_path: Path) -> None:
    path = getRegistryPath(str(tmp_path), "LSSTCam")
    pv = PackageVersions(versions={"ts_wep": "v1"})
    assert resolveVersionNumber(path, pv, timestamp="t0") == 1


def test_resolveVersionNumberSameSetReturnsSameNumber(tmp_path: Path) -> None:
    path = getRegistryPath(str(tmp_path), "LSSTCam")
    pv = PackageVersions(versions={"ts_wep": "v1"})
    first = resolveVersionNumber(path, pv, timestamp="t0")
    second = resolveVersionNumber(path, pv, timestamp="t1")  # later call, same versions
    assert first == second


def test_resolveVersionNumberNewSetIncrements(tmp_path: Path) -> None:
    path = getRegistryPath(str(tmp_path), "LSSTCam")
    pv1 = PackageVersions(versions={"ts_wep": "v1"})
    pv2 = PackageVersions(versions={"ts_wep": "v2"})
    assert resolveVersionNumber(path, pv1, timestamp="t0") == 1
    assert resolveVersionNumber(path, pv2, timestamp="t1") == 2
    # and the old set still maps to its original number
    assert resolveVersionNumber(path, pv1, timestamp="t2") == 1


def test_resolveVersionNumberPersistsRegistryFormat(tmp_path: Path) -> None:
    # pins the on-disk registry file format written by the head node
    path = getRegistryPath(str(tmp_path), "LSSTCam")
    pv1 = PackageVersions(versions={"ts_wep": "v1"})
    pv2 = PackageVersions(versions={"ts_wep": "v2"})
    resolveVersionNumber(path, pv1, timestamp="t0")
    resolveVersionNumber(path, pv2, timestamp="t1")
    with open(path) as f:
        entries = json.load(f)["entries"]
    assert len(entries) == 2
    byNumber = {e["number"]: e for e in entries}
    assert byNumber[1]["versions"] == {"ts_wep": "v1"}
    assert byNumber[2]["versions"] == {"ts_wep": "v2"}
    assert byNumber[1]["hash"] == pv1.versionHash()
    assert byNumber[1]["firstSeen"] == "t0"


def test_resolveVersionNumberCreatesMissingDirectory(tmp_path: Path) -> None:
    path = getRegistryPath(str(tmp_path / "newdir"), "LSSTCam")
    pv = PackageVersions(versions={"ts_wep": "v1"})
    assert resolveVersionNumber(path, pv, timestamp="t0") == 1
    assert os.path.isfile(path)


def test_resolveVersionNumberRobustToUnwritablePath(tmp_path: Path) -> None:
    # a path under a regular file (not a dir) can't be created -> sentinel
    notADir = tmp_path / "afile"
    notADir.write_text("x")
    path = os.path.join(str(notADir), "packageVersionRegistry-LSSTCam.json")
    pv = PackageVersions(versions={"ts_wep": "v1"})
    assert resolveVersionNumber(path, pv, timestamp="t0") == UNKNOWN_VERSION_NUMBER


def test_resolveVersionNumberRobustToMalformedRegistry(tmp_path: Path) -> None:
    path = getRegistryPath(str(tmp_path), "LSSTCam")
    with open(path, "w") as f:
        f.write("{ this is not valid json")
    pv = PackageVersions(versions={"ts_wep": "v1"})
    assert resolveVersionNumber(path, pv, timestamp="t0") == UNKNOWN_VERSION_NUMBER


def test_loadPackageVersionRegistryRoundTrip(tmp_path: Path) -> None:
    path = getRegistryPath(str(tmp_path), "LSSTCam")
    pv1 = PackageVersions(versions={"ts_wep": "v1"})
    pv2 = PackageVersions(versions={"ts_wep": "v2"})
    resolveVersionNumber(path, pv1, timestamp="t0")
    resolveVersionNumber(path, pv2, timestamp="t1")
    registry = loadPackageVersionRegistry(path)
    assert registry == {
        1: PackageVersions(versions={"ts_wep": "v1"}, versionNumber=1),
        2: PackageVersions(versions={"ts_wep": "v2"}, versionNumber=2),
    }


def test_loadPackageVersionRegistryDecodesDeployedFormat(tmp_path: Path) -> None:
    # a verbatim registry file as written by the deployed head node; if this
    # fails, reading back-dated data from the summit has been broken
    deployedRegistry = {
        "entries": [
            {
                "firstSeen": "2026-07-20T01:02:03.456789+00:00",
                "hash": "0f" * 32,
                "number": 1,
                "versions": {"ts_wep": "v17.6.1-alpha", "danish": "1.1.1"},
            }
        ]
    }
    path = tmp_path / "packageVersionRegistry-LSSTCam.json"
    path.write_text(json.dumps(deployedRegistry))
    registry = loadPackageVersionRegistry(str(path))
    assert registry == {
        1: PackageVersions(versions={"ts_wep": "v17.6.1-alpha", "danish": "1.1.1"}, versionNumber=1)
    }


def test_loadPackageVersionRegistryRaisesOnMissingFile(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        loadPackageVersionRegistry(str(tmp_path / "nonexistent.json"))


def test_loadPackageVersionRegistryRaisesOnMalformedEntry(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"entries": [{"number": 1}]}))  # no versions key
    with pytest.raises(ValueError, match="Malformed registry entry"):
        loadPackageVersionRegistry(str(path))


@responses.activate
def test_writePackageVersionsToConsDb(client: ConsDbClient) -> None:
    pv = PackageVersions(versions=dict(VERSIONS), versionNumber=42)
    responses.post(
        "http://example.com/consdb/insert/LSSTCam/cdb_lsstcam.exposure_quicklook/by_seq_num/20250624/123",
        json={"message": "ok"},
        match=[
            responses.matchers.query_param_matcher({"u": "1"}),
            responses.matchers.json_params_matcher(
                {
                    "table": "cdb_lsstcam.exposure_quicklook",
                    "values": {
                        "package_versions": {"versions": VERSIONS, "version_number": 42},
                        "exposure_id": 5025062400123,
                    },
                }
            ),
        ],
    )
    writePackageVersionsToConsDb(client, "LSSTCam", 20250624, 123, pv, exposureId=5025062400123)


@responses.activate
def test_writePackageVersionsToConsDbComputesExposureId(client: ConsDbClient) -> None:
    expectedExposureId = computeExposureId("LSSTCam", "O", 20250624, 123)
    pv = PackageVersions(versions=dict(VERSIONS), versionNumber=42)
    responses.post(
        "http://example.com/consdb/insert/LSSTCam/cdb_lsstcam.exposure_quicklook/by_seq_num/20250624/123",
        json={"message": "ok"},
        match=[
            responses.matchers.json_params_matcher(
                {
                    "table": "cdb_lsstcam.exposure_quicklook",
                    "values": {
                        "package_versions": {"versions": VERSIONS, "version_number": 42},
                        "exposure_id": expectedExposureId,
                    },
                }
            ),
        ],
    )
    writePackageVersionsToConsDb(client, "LSSTCam", 20250624, 123, pv)


def test_writePackageVersionsToConsDbRejectsBadIdentifiers(client: ConsDbClient) -> None:
    pv = PackageVersions(versions=dict(VERSIONS))
    with pytest.raises(ValueError, match="Invalid table name"):
        writePackageVersionsToConsDb(client, "LSSTCam", 20250624, 123, pv, table="bad table; drop")
    with pytest.raises(ValueError, match="Invalid column name"):
        writePackageVersionsToConsDb(client, "LSSTCam", 20250624, 123, pv, column="bad-column")


@responses.activate
def test_readPackageVersionsFromConsDbParsesJsonObjectCell(client: ConsDbClient) -> None:
    # the JSONB column coming back as an already-parsed JSON object; the query
    # SQL is pinned here too (the join via the exposure table)
    blob = {"versions": VERSIONS, "version_number": 42}
    expectedQuery = (
        "SELECT q.package_versions "
        "FROM cdb_lsstcam.exposure_quicklook q "
        "JOIN cdb_lsstcam.exposure e ON e.exposure_id = q.exposure_id "
        "WHERE e.day_obs = 20250624 AND e.seq_num = 123"
    )
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": [[blob]]},
        match=[responses.matchers.json_params_matcher({"query": expectedQuery})],
    )
    pv = readPackageVersionsFromConsDb(client, "LSSTCam", 20250624, 123)
    assert pv == PackageVersions(versions=dict(VERSIONS), versionNumber=42)


@responses.activate
def test_readPackageVersionsFromConsDbParsesJsonStringCell(client: ConsDbClient) -> None:
    # ...and coming back as a JSON string, in case the server serialises
    # JSONB that way instead
    blob = json.dumps({"versions": VERSIONS, "version_number": 42})
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": [[blob]]},
    )
    pv = readPackageVersionsFromConsDb(client, "LSSTCam", 20250624, 123)
    assert pv == PackageVersions(versions=dict(VERSIONS), versionNumber=42)


@responses.activate
def test_readPackageVersionsFromConsDbNoRowReturnsNone(client: ConsDbClient) -> None:
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": []},
    )
    assert readPackageVersionsFromConsDb(client, "LSSTCam", 20250624, 123) is None


@responses.activate
def test_readPackageVersionsFromConsDbNullCellReturnsNone(client: ConsDbClient) -> None:
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": [[None]]},
    )
    assert readPackageVersionsFromConsDb(client, "LSSTCam", 20250624, 123) is None


@responses.activate
def test_consDbRoundTrip(client: ConsDbClient) -> None:
    # write -> read round-trip through the mocked server: whatever blob the
    # write sends, handing it back through the query path reproduces the
    # original object
    pv = PackageVersions(versions=dict(VERSIONS), versionNumber=42)
    responses.post(
        "http://example.com/consdb/insert/LSSTCam/cdb_lsstcam.exposure_quicklook/by_seq_num/20250624/123",
        json={"message": "ok"},
    )
    writePackageVersionsToConsDb(client, "LSSTCam", 20250624, 123, pv, exposureId=5025062400123)
    requestBody = responses.calls[0].request.body
    assert requestBody is not None
    writtenBlob = json.loads(requestBody)["values"][PACKAGE_VERSIONS_COLUMN]
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": [[writtenBlob]]},
    )
    assert readPackageVersionsFromConsDb(client, "LSSTCam", 20250624, 123) == pv


@responses.activate
def test_writeBackdatedPackageVersions(client: ConsDbClient, tmp_path: Path) -> None:
    path = getRegistryPath(str(tmp_path), "LSSTCam")
    pv = PackageVersions(versions=dict(VERSIONS))
    assert resolveVersionNumber(path, pv, timestamp="t0") == 1
    responses.post(
        "http://example.com/consdb/insert/LSSTCam/cdb_lsstcam.exposure_quicklook/by_seq_num/20250624/123",
        json={"message": "ok"},
        match=[
            responses.matchers.json_params_matcher(
                {
                    "table": "cdb_lsstcam.exposure_quicklook",
                    "values": {
                        "package_versions": {"versions": VERSIONS, "version_number": 1},
                        "exposure_id": 5025062400123,
                    },
                }
            ),
        ],
    )
    written = writeBackdatedPackageVersions(
        client, "LSSTCam", 20250624, 123, versionNumber=1, registryPath=path, exposureId=5025062400123
    )
    assert written == PackageVersions(versions=dict(VERSIONS), versionNumber=1)


def test_writeBackdatedPackageVersionsRejectsUnknownNumber(client: ConsDbClient, tmp_path: Path) -> None:
    path = getRegistryPath(str(tmp_path), "LSSTCam")
    resolveVersionNumber(path, PackageVersions(versions={"ts_wep": "v1"}), timestamp="t0")
    with pytest.raises(ValueError, match="known numbers: \\[1\\]"):
        writeBackdatedPackageVersions(client, "LSSTCam", 20250624, 123, versionNumber=7, registryPath=path)
