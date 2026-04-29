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

__all__ = ("getVcr",)


def getVcr():
    """Get a VCR object for use in tests.

    Use record_mode="none" to run tests for normal operation. To update files
    or generate new ones, make sure you have a working connection to the EFD
    and temporarily run with mode="all" via *both* python/pytest *and* with
    scons, as these generate slightly different HTTP requests for some reason.
    Also make sure to do this at the summit (USDF coverage is provided by the
    same recording, since matching ignores host/port and so is independent of
    whether requests go through a proxy). The TTS is explicitly skipped and
    does not need to follow this procedure.
    """
    dirname = os.path.dirname(__file__)
    cassette_library_dir = os.path.join(dirname, "data", "cassettes")
    safe_vcr = vcr.VCR(
        record_mode="none",
        cassette_library_dir=cassette_library_dir,
        path_transformer=vcr.VCR.ensure_suffix(".yaml"),
        match_on=["method", "path", "query", "body"],
    )
    return safe_vcr
