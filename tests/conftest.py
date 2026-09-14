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

"""VCR plugin configuration for the test suite.

Test classes which talk to the EFD are decorated with ``@pytest.mark.vcr``,
which makes the VCR pytest plugin replay (or record) each test's HTTP traffic
from a cassette in ``tests/data/cassettes``. See ``utils.getVcr`` for details
of the cassette layout and how to re-record.

Two mutually exclusive plugins provide the ``vcr`` marker: ``pytest-vcr``
(shipped with older rubin-env releases) and ``pytest-recording`` (its
maintained replacement). This file supports whichever one is installed, so
that the transition between them in the stack environment does not have to be
coordinated with this package. Once the environment has moved to
``pytest-recording`` the ``pytest-vcr`` support here can be dropped.

The differences which matter here are:

* the record mode option is ``--vcr-record`` for ``pytest-vcr`` and
  ``--record-mode`` for ``pytest-recording``;
* the cassette name is set via the ``vcr_cassette_name`` fixture for
  ``pytest-vcr`` and ``default_cassette_name`` for ``pytest-recording``;
* ``pytest-vcr`` consumes ``vcr_config`` from a module-scoped fixture, so the
  override below must be module-scoped too (``pytest-recording`` accepts any
  scope).
"""

from typing import Any

import pytest
from utils import CASSETTE_DIR, VCR_CONFIG

# Map the loaded plugin's module name to its record mode command line option.
_VCR_PLUGINS = {
    "pytest_recording.plugin": "--record-mode",
    "pytest_vcr": "--vcr-record",
}


def _getVcrPlugin(config: pytest.Config) -> str | None:
    """Return the module name of the VCR plugin which is loaded, if any.

    Plugins are identified by their module rather than their registered name,
    so that this also works when a plugin is loaded explicitly with ``-p``
    rather than via its entry point.
    """
    for _, plugin in config.pluginmanager.list_name_plugin():
        moduleName = getattr(plugin, "__name__", None)
        if moduleName in _VCR_PLUGINS:
            return moduleName
    return None


def pytest_configure(config: pytest.Config) -> None:
    plugin = _getVcrPlugin(config)
    if plugin is None:
        raise RuntimeError(
            "A VCR pytest plugin (pytest-recording or pytest-vcr) is required to run the test suite"
        )
    # Keep the setUpClass recordings (see utils.getVcr) in step with the
    # record mode given on the command line, e.g. ``--record-mode=all``.
    recordMode = config.getoption(_VCR_PLUGINS[plugin], default=None)
    if recordMode is not None:
        VCR_CONFIG["record_mode"] = recordMode


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    # All modules share a single, flat cassette directory
    return CASSETTE_DIR


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    return dict(VCR_CONFIG)


def _getCassetteName(request: pytest.FixtureRequest) -> str:
    # Name cassettes after the bare test function name (both plugins default
    # to prefixing the class name), matching the existing recordings.
    return request.node.name


@pytest.fixture
def default_cassette_name(request: pytest.FixtureRequest) -> str:
    # pytest-recording
    return _getCassetteName(request)


@pytest.fixture
def vcr_cassette_name(request: pytest.FixtureRequest) -> str:
    # pytest-vcr
    return _getCassetteName(request)
