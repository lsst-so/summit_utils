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

import asyncio
import logging
import os
import tempfile
import unittest
from typing import Any

import matplotlib.pyplot as plt
from utils import getVcr

import lsst.utils.tests
from lsst.summit.utils.efdUtils import makeEfdClient
from lsst.summit.utils.m1m3.inertia_compensation_system import evaluate_m1m3_ics_single_slew
from lsst.summit.utils.m1m3.plots.plot_ics import FIGURE_HEIGHT, FIGURE_WIDTH, plot_hp_measured_data
from lsst.summit.utils.tmaUtils import TMAEvent, TMAEventMaker

vcr = getVcr()


@vcr.use_cassette()
class M1M3ICSTestCase(lsst.utils.tests.TestCase):
    # class attributes populated in setUp
    client: Any
    dayObs: int
    seqNumToPlot: int
    tmaEventMaker: TMAEventMaker
    events: list[TMAEvent]
    sampleData: Any
    outputDir: str
    log: logging.Logger

    @classmethod
    @vcr.use_cassette()
    def setUp(cls) -> None:
        try:
            cls.client = makeEfdClient(testing=True)
        except RuntimeError:
            raise unittest.SkipTest("Could not instantiate an EFD client")

        cls.dayObs = 20230728  # need a day with M1M3 data
        cls.seqNumToPlot = 38

        cls.tmaEventMaker = TMAEventMaker(cls.client)
        cls.events = cls.tmaEventMaker.getEvents(cls.dayObs)  # does the fetch
        cls.sampleData = cls.tmaEventMaker._data[cls.dayObs]  # pull the data from the object and test length
        cls.outputDir = tempfile.mkdtemp()
        cls.log = logging.getLogger(__name__)

    @vcr.use_cassette()
    def tearDown(self) -> None:
        loop = asyncio.get_event_loop()
        if self.client.influx_client is not None:
            loop.run_until_complete(self.client.influx_client.close())

    @vcr.use_cassette()
    def test_analysis(self) -> None:
        self.log.info(f"Writing temp output files to {self.outputDir}")
        plotFilename = os.path.join(self.outputDir, "testPlotting_exp.jpg")
        statFilename = os.path.join(self.outputDir, "m1m3_ics_stats.csv")
        dataFilename = os.path.join(self.outputDir, "m1m3_ics_df.csv")

        event = self.events[self.seqNumToPlot]

        results = evaluate_m1m3_ics_single_slew(event, self.client)
        results.stats.to_csv(statFilename)
        results.df.to_csv(dataFilename)

        self.assertTrue(os.path.isfile(dataFilename))
        # data is big, about 2.5MB at time of writing
        self.assertTrue(os.path.getsize(dataFilename) > 400_000)
        # stats are small, about 1.8kB at time of writing
        self.assertTrue(os.path.isfile(statFilename))
        self.assertTrue(os.path.getsize(statFilename) > 1000)

        dpi = 300
        fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=dpi)
        fig = plot_hp_measured_data(results, fig)
        fig.savefig(plotFilename)
        self.assertTrue(os.path.isfile(plotFilename))
        self.assertTrue(os.path.getsize(plotFilename) > 200_000)  # plot is about 400kB at 300 dpi


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: object) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
