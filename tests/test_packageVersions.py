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

"""Tests for the science-package version data model and reading it from ConsDB.

The ConsDB JSON blob (``{"hash": ..., "versions": {...}}``) is a wire format
that must not drift without a migration plan: any already-written ConsDB rows
use this shape, so the verbatim tests below fail if it changes. The read path
also proves the design's key property - the versions come back with only a
ConsDB client. Writing that blob lives in Rapid Analysis, which reuses the
shape (`toDict`) and the table/column constants pinned here.
"""

import json
from types import SimpleNamespace
from typing import cast

import pytest
import responses

from lsst.daf.butler import DimensionRecord
from lsst.summit.utils import ConsDbClient
from lsst.summit.utils.packageVersions import (
    PACKAGE_VERSIONS_COLUMN,
    PackageVersions,
    readPackageVersionsByHash,
    readPackageVersionsForExposure,
    readPackageVersionsFromConsDb,
)

VERSIONS = {
    "ts_wep": "v17.6.1-alpha",
    "donut_viz": "v4.4.0-alpha",
    "rubintv_production": "abc123",
}
# The hash is a stable SHA-256 of the canonical (sorted-key) versions dict.
EXPECTED_HASH = PackageVersions(versions=VERSIONS).versionHash()


@pytest.fixture
def client() -> ConsDbClient:
    """A ConsDbClient pointed at a fake url; use @responses.activate to mock
    the connection.
    """
    return ConsDbClient("http://example.com/consdb")


def test_versionHashStableAndOrderIndependent() -> None:
    a = PackageVersions(versions={"ts_wep": "v1", "donut_viz": "v2"})
    b = PackageVersions(versions={"donut_viz": "v2", "ts_wep": "v1"})  # different insertion order
    assert a.versionHash() == b.versionHash()
    assert len(a.versionHash()) == 64  # SHA-256 hex digest


def test_versionHashChangesWhenAVersionChanges() -> None:
    a = PackageVersions(versions={"ts_wep": "v1", "donut_viz": "v2"})
    c = PackageVersions(versions={"ts_wep": "v1", "donut_viz": "v3"})
    assert a.versionHash() != c.versionHash()


def test_toDictPinsBlobShape() -> None:
    # this is the wire format stored in the ConsDB JSONB column - an exact
    # match, not a subset check, so any shape drift fails here. Rapid Analysis
    # writes exactly this, so the same test is duplicated there.
    pv = PackageVersions(versions=dict(VERSIONS))
    assert pv.toDict() == {"hash": EXPECTED_HASH, "versions": VERSIONS}


def test_dictRoundTrip() -> None:
    pv = PackageVersions(versions=dict(VERSIONS))
    assert PackageVersions.fromDict(pv.toDict()) == pv


def test_jsonRoundTrip() -> None:
    pv = PackageVersions(versions=dict(VERSIONS))
    assert PackageVersions.fromJson(pv.toJson()) == pv
    # canonical: sorted keys, so identical sets serialise identically
    reordered = PackageVersions(versions=dict(reversed(list(VERSIONS.items()))))
    assert reordered.toJson() == pv.toJson()


def test_fromDictIgnoresHashAndTrustsVersions() -> None:
    # hash is derived from versions, so a stale/absent hash in the blob is not
    # authoritative: fromDict rebuilds from versions alone
    fromNoHash = PackageVersions.fromDict({"versions": VERSIONS})
    fromStaleHash = PackageVersions.fromDict({"hash": "deadbeef", "versions": VERSIONS})
    assert fromNoHash == fromStaleHash == PackageVersions(versions=VERSIONS)
    assert fromStaleHash.versionHash() == EXPECTED_HASH  # recomputed, not the stale value


def test_fromDictRejectsMissingVersions() -> None:
    with pytest.raises(ValueError, match="no 'versions' key"):
        PackageVersions.fromDict({"hash": "deadbeef"})


def test_fromJsonRejectsNonObject() -> None:
    with pytest.raises(ValueError, match="Expected a JSON object"):
        PackageVersions.fromJson('["not", "an", "object"]')


@responses.activate
def test_readPackageVersionsFromConsDbParsesJsonObjectCell(
    client: ConsDbClient,
) -> None:
    # the JSONB column coming back as an already-parsed JSON object; the query
    # SQL is pinned here too (the join via the exposure table)
    blob = {"hash": EXPECTED_HASH, "versions": VERSIONS}
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
    assert pv == PackageVersions(versions=dict(VERSIONS))


@responses.activate
def test_readPackageVersionsFromConsDbParsesJsonStringCell(
    client: ConsDbClient,
) -> None:
    # ...and coming back as a JSON string, in case the server serialises
    # JSONB that way instead
    blob = json.dumps({"hash": EXPECTED_HASH, "versions": VERSIONS})
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": [[blob]]},
    )
    pv = readPackageVersionsFromConsDb(client, "LSSTCam", 20250624, 123)
    assert pv == PackageVersions(versions=dict(VERSIONS))


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
def test_readPackageVersionsForExposure(client: ConsDbClient) -> None:
    # the DimensionRecord wrapper must unpack the dataId and delegate: it
    # produces the same query keyed by this record's day_obs/seq_num
    blob = {"hash": EXPECTED_HASH, "versions": VERSIONS}
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
    record = cast(DimensionRecord, SimpleNamespace(instrument="LSSTCam", day_obs=20250624, seq_num=123))
    pv = readPackageVersionsForExposure(client, record)
    assert pv == PackageVersions(versions=dict(VERSIONS))


@responses.activate
def test_readPackageVersionsByHash(client: ConsDbClient) -> None:
    # look up a version set by its hash; the SQL (the ->>'hash' filter and the
    # LIMIT 1) is pinned so it cannot drift from the blob shape toDict writes
    blob = {"hash": EXPECTED_HASH, "versions": VERSIONS}
    expectedQuery = (
        "SELECT q.package_versions "
        "FROM cdb_lsstcam.exposure_quicklook q "
        f"WHERE q.package_versions->>'hash' = '{EXPECTED_HASH}' "
        "LIMIT 1"
    )
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": [[blob]]},
        match=[responses.matchers.json_params_matcher({"query": expectedQuery})],
    )
    pv = readPackageVersionsByHash(client, "LSSTCam", EXPECTED_HASH)
    assert pv == PackageVersions(versions=dict(VERSIONS))
    assert pv is not None and pv.versionHash() == EXPECTED_HASH


@responses.activate
def test_readPackageVersionsByHashNormalisesCase(client: ConsDbClient) -> None:
    # versionHash() always produces lowercase hex and the SQL string
    # comparison is case-sensitive, so an uppercase paste of a valid hash must
    # be lowercased before interpolation - it used to silently return None.
    # The matcher pins the emitted SQL, so this fails if the hash reaches the
    # query un-normalised.
    blob = {"hash": EXPECTED_HASH, "versions": VERSIONS}
    expectedQuery = (
        "SELECT q.package_versions "
        "FROM cdb_lsstcam.exposure_quicklook q "
        f"WHERE q.package_versions->>'hash' = '{EXPECTED_HASH}' "
        "LIMIT 1"
    )
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": [[blob]]},
        match=[responses.matchers.json_params_matcher({"query": expectedQuery})],
    )
    pv = readPackageVersionsByHash(client, "LSSTCam", EXPECTED_HASH.upper())
    assert pv == PackageVersions(versions=dict(VERSIONS))


@responses.activate
def test_readPackageVersionsByHashNoRowReturnsNone(client: ConsDbClient) -> None:
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": []},
    )
    assert readPackageVersionsByHash(client, "LSSTCam", EXPECTED_HASH) is None


def test_readPackageVersionsByHashRejectsNonHexHash(client: ConsDbClient) -> None:
    # the hash is interpolated into the query as a string literal, so a non-hex
    # value (e.g. an injection attempt) must be rejected before any request
    with pytest.raises(ValueError, match="Invalid version hash"):
        readPackageVersionsByHash(client, "LSSTCam", "abc'; DROP TABLE x; --")
