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

import os

import vcr

__all__ = ("CASSETTE_DIR", "VCR_CONFIG", "getVcr")

# The single source of truth for how cassettes are recorded and matched. It is
# consumed by pytest-recording via the fixtures in ``conftest.py`` for all test
# methods, and by ``getVcr()`` for the ``setUpClass`` methods.
CASSETTE_DIR = os.path.join(os.path.dirname(__file__), "data", "cassettes")
VCR_CONFIG = {
    "record_mode": "none",
    # matching ignores host/port and so is independent of whether requests go
    # through the USDF proxy or not.
    "match_on": ["method", "path", "query", "body"],
}


def getVcr() -> vcr.VCR:
    """Get a ``vcr.VCR`` for recording ``setUpClass`` methods.

    Test methods (and their ``setUp``/``tearDown``) are handled by the
    ``pytest-recording`` plugin, via ``@pytest.mark.vcr`` on the test class.
    That plugin installs cassettes through a per-test fixture, so it cannot
    cover ``setUpClass``, which runs before any per-test fixture. For that
    case, use the returned object as ``@classVcr.use_cassette()``, which uses
    the same configuration and cassette directory as the plugin. This can also
    be nested inside the plugin's cassette to record a ``setUp`` method
    separately from its tests.

    Cassettes live in ``tests/data/cassettes`` and are named after the bare
    function name (``setUpClass.yaml``, ``test_getEfdData.yaml``, ...). They
    are shared between test modules, so ``setUpClass.yaml`` holds the
    recordings for every module's ``setUpClass``.

    To update the cassettes or generate new ones, make sure you have a working
    connection to the EFD and run with ``pytest --record-mode=all``, both
    via pytest directly and via scons, as these generate slightly different
    HTTP requests for some reason. Also make sure to do this at the summit
    (USDF coverage is provided by the same recording, since matching ignores
    host/port and so is independent of whether requests go through a proxy).
    The TTS is explicitly skipped and does not need to follow this procedure.
    """
    return vcr.VCR(
        cassette_library_dir=CASSETTE_DIR,
        path_transformer=vcr.VCR.ensure_suffix(".yaml"),
        **VCR_CONFIG,
    )
