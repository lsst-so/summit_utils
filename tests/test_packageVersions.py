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

The JSON stored in ConsDB (``{"hash": ..., "versions": {...}}``) is a wire
format: rows already written use it, so `test_toDictPinsWireFormat` fails if it
changes.
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
    """A ConsDbClient pointed at a fake url.

    The tests below use the ``responses`` library to mock the HTTP layer:
    ``@responses.activate`` intercepts all requests, and ``responses.post(url,
    json=...)`` registers the canned reply for the POST the client makes to
    ``/query``.
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


def test_toDictPinsWireFormat() -> None:
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
    # the JSONB column coming back as an already-parsed JSON object. This
    # registers the mock reply for the client's POST to /query.
    blob = {"hash": EXPECTED_HASH, "versions": VERSIONS}
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": [[blob]]},
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
    # the DimensionRecord wrapper must unpack the record's dataId into the
    # query, so check the record's day_obs/seq_num reached the request
    blob = {"hash": EXPECTED_HASH, "versions": VERSIONS}
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": [[blob]]},
    )
    record = cast(DimensionRecord, SimpleNamespace(instrument="LSSTCam", day_obs=20250624, seq_num=123))
    pv = readPackageVersionsForExposure(client, record)
    assert pv == PackageVersions(versions=dict(VERSIONS))
    body = responses.calls[0].request.body
    assert isinstance(body, (str, bytes))
    sentQuery = json.loads(body)["query"]
    assert "cdb_lsstcam" in sentQuery
    assert "20250624" in sentQuery and "123" in sentQuery


@responses.activate
def test_readPackageVersionsByHash(client: ConsDbClient) -> None:
    # look up a version set by its hash
    blob = {"hash": EXPECTED_HASH, "versions": VERSIONS}
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": [[blob]]},
    )
    pv = readPackageVersionsByHash(client, "LSSTCam", EXPECTED_HASH)
    assert pv == PackageVersions(versions=dict(VERSIONS))
    assert pv is not None and pv.versionHash() == EXPECTED_HASH


@responses.activate
def test_readPackageVersionsByHashNormalisesCase(client: ConsDbClient) -> None:
    # versionHash() always produces lowercase hex and the SQL string
    # comparison is case-sensitive, so an uppercase paste of a valid hash must
    # be lowercased before it reaches the query - it used to silently return
    # None.
    blob = {"hash": EXPECTED_HASH, "versions": VERSIONS}
    responses.post(
        "http://example.com/consdb/query",
        json={"columns": [PACKAGE_VERSIONS_COLUMN], "data": [[blob]]},
    )
    pv = readPackageVersionsByHash(client, "LSSTCam", EXPECTED_HASH.upper())
    assert pv == PackageVersions(versions=dict(VERSIONS))
    body = responses.calls[0].request.body
    assert isinstance(body, (str, bytes))
    sentQuery = json.loads(body)["query"]
    assert EXPECTED_HASH in sentQuery
    assert EXPECTED_HASH.upper() not in sentQuery


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
