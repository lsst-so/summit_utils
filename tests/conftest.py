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

"""pytest-recording configuration for the test suite.

Test classes which talk to the EFD are decorated with ``@pytest.mark.vcr``,
which makes pytest-recording replay (or record) each test's HTTP traffic from
a cassette in ``tests/data/cassettes``. See ``utils.getVcr`` for details of the
cassette layout and how to re-record.
"""

from typing import Any

import pytest
from utils import CASSETTE_DIR, VCR_CONFIG


def pytest_configure(config: pytest.Config) -> None:
    if not config.pluginmanager.hasplugin("recording"):
        raise RuntimeError("The pytest-recording plugin is required to run the test suite")
    # Keep the setUpClass recordings (see utils.getVcr) in step with the
    # record mode given on the command line, e.g. ``--record-mode=all``.
    recordMode = config.getoption("--record-mode", default=None)
    if recordMode is not None:
        VCR_CONFIG["record_mode"] = recordMode


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    # All modules share a single, flat cassette directory
    return CASSETTE_DIR


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    return dict(VCR_CONFIG)


@pytest.fixture
def default_cassette_name(request: pytest.FixtureRequest) -> str:
    # Name cassettes after the bare test function name (the pytest-recording
    # default prefixes the class name), matching the existing recordings.
    return request.node.name
