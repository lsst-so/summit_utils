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

import json
from typing import Any

import pytest
import responses
from astropy.table import Table
from requests import HTTPError, Response, Timeout

from lsst.summit.utils import ConsDbClient, FlexibleMetadataInfo
from lsst.summit.utils.consdbClient import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_READ_TIMEOUT,
    getCcdVisitTableForDay,
)


@pytest.fixture
def client() -> ConsDbClient:
    """Initialize client with a fake url
    Requires mocking connection with @responses.activate decorator
    """
    return ConsDbClient("http://example.com/consdb")


def test_table_name() -> None:
    instrument = "latiss"
    obs_type = "exposure"
    assert (
        ConsDbClient.compute_flexible_metadata_table_name(instrument, obs_type)
        == "cdb_latiss.exposure_flexdata"
    )


@responses.activate
def test_add_flexible_metadata_key(client: ConsDbClient) -> None:
    instrument = "latiss"
    obs_type = "exposure"
    responses.post(
        "http://example.com/consdb/flex/latiss/exposure/addkey",
        json={
            "message": "Key added to flexible metadata",
            "key": "foo",
            "instrument": "latiss",
            "obs_type": "exposure",
        },
        match=[
            responses.matchers.json_params_matcher({"key": "foo", "dtype": "bool", "doc": "bool key"}),
        ],
    )
    responses.post(
        "http://example.com/consdb/flex/latiss/exposure/addkey",
        json={
            "message": "Key added to flexible metadata",
            "key": "bar",
            "instrument": "latiss",
            "obs_type": "exposure",
        },
        match=[
            responses.matchers.json_params_matcher({"key": "bar", "dtype": "int", "doc": "int key"}),
        ],
    )
    responses.post(
        "http://example.com/consdb/flex/bad_instrument/exposure/addkey",
        status=404,
        json={"message": "Unknown instrument", "value": "bad_instrument", "valid": ["latiss"]},
    )
    responses.post(
        "http://example.com/consdb/flex/latiss/bad_obs_type/addkey",
        status=404,
        json={"message": "Unknown observation type", "value": "bad_obs_type", "valid": ["exposure"]},
    )

    assert (
        client.add_flexible_metadata_key(instrument, obs_type, "foo", "bool", "bool key").json()["key"]
        == "foo"
    )
    assert (
        client.add_flexible_metadata_key(instrument, obs_type, "bar", "int", "int key").json()["instrument"]
        == "latiss"
    )
    with pytest.raises(HTTPError, match="404") as e:
        client.add_flexible_metadata_key("bad_instrument", obs_type, "error", "int", "instrument error")
    assert "Unknown instrument" in str(e.value.__notes__)
    assert e.value.response is not None
    json_data = e.value.response.json()
    assert json_data["message"] == "Unknown instrument"
    assert json_data["value"] == "bad_instrument"
    assert json_data["valid"] == ["latiss"]
    with pytest.raises(HTTPError, match="404"):
        client.add_flexible_metadata_key(instrument, "bad_obs_type", "error", "int", "obs_type error")


@responses.activate
def test_get_flexible_metadata_keys(client: ConsDbClient) -> None:
    description = {"foo": ["bool", "a", None, None], "bar": ["float", "b", "deg", "pos.eq.ra"]}
    responses.get(
        "http://example.com/consdb/flex/latiss/exposure/schema",
        json=description,
    )
    instrument = "latiss"
    obs_type = "exposure"
    assert client.get_flexible_metadata_keys(instrument, obs_type) == {
        "foo": FlexibleMetadataInfo("bool", "a"),
        "bar": FlexibleMetadataInfo("float", "b", "deg", "pos.eq.ra"),
    }


@responses.activate
def test_get_flexible_metadata(client: ConsDbClient) -> None:
    results = {"bool_key": True, "int_key": 42, "float_key": 3.14159, "str_key": "foo"}
    responses.get(
        "http://example.com/consdb/flex/latiss/exposure/obs/271828",
        json=results,
    )
    responses.get(
        "http://example.com/consdb/flex/latiss/exposure/obs/271828?k=float_key", json={"float_key": 3.14159}
    )
    responses.get(
        "http://example.com/consdb/flex/latiss/exposure/obs/271828?k=int_key&k=float_key",
        json={"float_key": 3.14159, "int_key": 42},
    )
    instrument = "latiss"
    obs_type = "exposure"
    obs_id = 271828
    assert client.get_flexible_metadata(instrument, obs_type, obs_id) == results
    assert client.get_flexible_metadata(instrument, obs_type, obs_id, ["float_key"]) == {
        "float_key": results["float_key"]
    }
    assert client.get_flexible_metadata(instrument, obs_type, obs_id, ["int_key", "float_key"]) == {
        "int_key": results["int_key"],
        "float_key": results["float_key"],
    }


@responses.activate
def test_insert_flexible_metadata(client: ConsDbClient) -> None:
    instrument = "latiss"
    obs_type = "exposure"
    with pytest.raises(ValueError):
        client.insert_flexible_metadata(instrument, obs_type, 271828)
    # TODO: more POST tests


@responses.activate
def test_schema(client: ConsDbClient) -> None:
    description = {"foo": ("bool", "a"), "bar": ("int", "b")}
    responses.get(
        "http://example.com/consdb/schema/latiss/misc_table",
        json=description,
    )
    instrument = "latiss"
    table = "misc_table"
    assert client.schema(instrument, table) == description


@responses.activate
@pytest.mark.parametrize(
    "secret, redacted",
    [
        ("usdf:v987wefVMPz", "us***:v9***"),
        ("u:v", "u***:v***"),
        ("ulysses", "ul***"),
        (":alberta94", "***:al***"),
    ],
)
def test_clean_token_url_response(secret: str, redacted: str) -> None:
    """Test tokens URL is cleaned when an error is thrown from requests
    Use with pytest raises assert an error'
    assert that url does not contain tokens
    """
    domain = "@usdf-fake.slackers.stanford.edu/consdb"
    complex_client = ConsDbClient(f"https://{secret}{domain}")

    obs_type = "exposure"
    responses.post(
        f"https://{secret}{domain}/flex/bad_instrument/exposure/addkey",
        status=404,
    )
    with pytest.raises(HTTPError, match="404") as error:
        complex_client.add_flexible_metadata_key(
            "bad_instrument", obs_type, "error", "int", "instrument error"
        )

    url = error.value.args[0].split()[-1]
    sanitized = f"https://{redacted}{domain}/flex/bad_instrument/exposure/addkey"
    assert url == sanitized


def test_client(client: ConsDbClient) -> None:
    """Test ConsDbClient is initialized properly"""
    assert "clean_url" in str(client.session.hooks["response"])
    assert client.connect_timeout == DEFAULT_CONNECT_TIMEOUT
    assert client.read_timeout == DEFAULT_READ_TIMEOUT
    assert client.timeout == (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)


def test_timeout_override() -> None:
    """The timeouts are configurable, including disabling them entirely."""
    tuned = ConsDbClient("http://example.com/consdb", connect_timeout=5, read_timeout=30)
    assert tuned.timeout == (5, 30)
    unbounded = ConsDbClient("http://example.com/consdb", connect_timeout=None, read_timeout=None)
    assert unbounded.timeout == (None, None)


def test_get_passes_timeout(client: ConsDbClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET requests must carry a timeout so a stalled server cannot hang."""
    captured: dict[str, Any] = {}

    def fake_get(url: str, **kwargs: Any) -> Response:
        captured.update(kwargs)
        response = Response()
        response.status_code = 200
        response._content = b"{}"
        return response

    monkeypatch.setattr(client.session, "get", fake_get)
    client.schema()
    assert captured["timeout"] == (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)


def test_post_passes_timeout(client: ConsDbClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST requests must carry a timeout so a stalled server cannot hang."""
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> Response:
        captured.update(kwargs)
        response = Response()
        response.status_code = 200
        response._content = b'{"message": "Data inserted"}'
        return response

    monkeypatch.setattr(client.session, "post", fake_post)
    client.insert("latiss", "exposure", 271828, {"foo": 1})
    assert captured["timeout"] == (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)


def test_timeout_propagates(client: ConsDbClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A timed-out request surfaces as requests.Timeout to the caller."""

    def fake_get(url: str, **kwargs: Any) -> Response:
        raise Timeout("timed out")

    monkeypatch.setattr(client.session, "get", fake_get)
    with pytest.raises(Timeout):
        client.schema()


@responses.activate
def test_insert_obs_id(client: ConsDbClient) -> None:
    """An integer obs_id targets the ``.../obs/{obs_id}`` endpoint."""
    responses.post(
        "http://example.com/consdb/insert/latiss/exposure/obs/271828",
        json={"message": "Data inserted"},
        match=[
            responses.matchers.json_params_matcher(
                {"table": "exposure", "obs_id": 271828, "values": {"foo": 1}}
            ),
        ],
    )
    assert client.insert("latiss", "exposure", 271828, {"foo": 1}).json()["message"] == "Data inserted"


@responses.activate
def test_insert_by_seq_num(client: ConsDbClient) -> None:
    """A 2-tuple targets the ``.../{day_obs}/{seq_num}`` endpoint."""
    responses.post(
        "http://example.com/consdb/insert/latiss/exposure/by_seq_num/20240603/123",
        json={"message": "Data inserted"},
        match=[
            responses.matchers.json_params_matcher({"table": "exposure", "values": {"foo": 1}}),
        ],
    )
    assert (
        client.insert("latiss", "exposure", (20240603, 123), {"foo": 1}).json()["message"] == "Data inserted"
    )


@responses.activate
def test_insert_by_seq_num_detector(client: ConsDbClient) -> None:
    """A 3-tuple targets the ``.../{day_obs}/{seq_num}/{detector}``
    endpoint for per-detector (ccdexposure-level) tables."""
    responses.post(
        "http://example.com/consdb/insert/lsstcam/ccdexposure/by_seq_num/20240603/123/94",
        json={"message": "Data inserted"},
        match=[
            responses.matchers.json_params_matcher({"table": "ccdexposure", "values": {"foo": 1}}),
        ],
    )
    assert (
        client.insert("lsstcam", "ccdexposure", (20240603, 123, 94), {"foo": 1}).json()["message"]
        == "Data inserted"
    )


@responses.activate
def test_insert_allow_update(client: ConsDbClient) -> None:
    """allow_update appends ``?u=1`` to upsert against the addressed key."""
    responses.post(
        "http://example.com/consdb/insert/lsstcam/ccdexposure/by_seq_num/20240603/123/94?u=1",
        json={"message": "Data inserted"},
        match=[responses.matchers.query_param_matcher({"u": "1"})],
    )
    assert (
        client.insert("lsstcam", "ccdexposure", (20240603, 123, 94), {"foo": 1}, allow_update=True).json()[
            "message"
        ]
        == "Data inserted"
    )


def test_insert_bad_obs_id_tuple(client: ConsDbClient) -> None:
    """A tuple that is not length 2 or 3 is rejected before any request."""
    with pytest.raises(AssertionError, match="obs_id tuple"):
        client.insert("latiss", "exposure", (20240603,), {"foo": 1})  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match="obs_id tuple"):
        client.insert("latiss", "exposure", (20240603, 123, 94, 0), {"foo": 1})  # type: ignore[arg-type]


def test_insert_no_values(client: ConsDbClient) -> None:
    """Inserting with no values raises before any request."""
    with pytest.raises(ValueError, match="No values to insert"):
        client.insert("latiss", "exposure", 271828, {})


@responses.activate
def test_getCcdVisitTableForDay_dedupes_overlapping_columns(client: ConsDbClient) -> None:
    """Columns already present in ccdvisit1_quicklook must not be re-selected.

    ``cvq.*`` pulls in every column of ccdvisit1_quicklook. As ConsDB
    denormalises identity columns (visit_id, detector, seq_num, ...) onto the
    quicklook table, naming those again explicitly from the joined tables makes
    the server return duplicate column names, which astropy refuses to build a
    Table from (DM-55152). They must be dropped from the SELECT instead.
    """
    url = "http://example.com/consdb/query"

    # First query is the LIMIT 0 schema probe: pretend the quicklook table has
    # denormalised visit_id, detector and seq_num onto itself.
    responses.post(
        url,
        json={"columns": ["ccdvisit_id", "visit_id", "detector", "seq_num", "psf_sigma"], "data": []},
    )
    # Second query is the real data query; its response only has to build
    # cleanly, the behaviour under test is the SELECT clause that was sent.
    responses.post(
        url,
        json={
            "columns": ["ccdvisit_id", "visit_id", "detector", "seq_num", "psf_sigma", "band"],
            "data": [[1, 100, 7, 5, 1.2, "r"]],
        },
    )

    table = getCcdVisitTableForDay(client, 20240101)
    assert isinstance(table, Table)

    # Only inspect the SELECT clause; the WHERE clause legitimately references
    # cv.visit_id etc. in its join conditions.
    requestBody = responses.calls[1].request.body
    assert isinstance(requestBody, (str, bytes))
    sentQuery = json.loads(requestBody)["query"]
    selectClause = sentQuery.split(" FROM ")[0]
    # Columns already provided by cvq.* must not be re-selected explicitly...
    assert "cvq.*" in selectClause
    assert "cv.visit_id" not in selectClause
    assert "cv.detector" not in selectClause
    assert "v.seq_num" not in selectClause
    # ...but columns the quicklook table lacks must still be pulled from visit1
    for col in ("v.band", "v.exp_time", "v.day_obs", "v.img_type"):
        assert col in selectClause


@responses.activate
def test_getCcdVisitTableForDay_keeps_columns_when_no_overlap(client: ConsDbClient) -> None:
    """When the quicklook table shares no names, every extra is selected."""
    url = "http://example.com/consdb/query"
    responses.post(url, json={"columns": ["ccdvisit_id", "psf_sigma"], "data": []})
    responses.post(url, json={"columns": ["ccdvisit_id", "psf_sigma"], "data": [[1, 1.2]]})

    getCcdVisitTableForDay(client, 20240101)

    requestBody = responses.calls[1].request.body
    assert isinstance(requestBody, (str, bytes))
    sentQuery = json.loads(requestBody)["query"]
    selectClause = sentQuery.split(" FROM ")[0]
    for col in ("cv.detector", "cv.visit_id", "v.band", "v.exp_time", "v.seq_num", "v.day_obs", "v.img_type"):
        assert col in selectClause


# TODO: more POST tests
#    client.insert_multiple(instrument, table, obs_dict, allow_update)
#    client.query(query)
