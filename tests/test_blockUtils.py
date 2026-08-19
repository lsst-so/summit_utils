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

"""Test cases for utils."""

import asyncio
import json
import os
import unittest
from typing import Any

import pandas as pd
from utils import getVcr

import lsst.utils.tests
from lsst.summit.utils.blockUtils import BlockParser
from lsst.summit.utils.efdUtils import makeEfdClient

__all__ = ("writeNewBlockInfoTestTruthValues",)

HAS_EFD_CLIENT = True
try:
    import lsst_efd_client  # noqa: F401 just need to check this is available
except ImportError:
    HAS_EFD_CLIENT = False

vcr = getVcr()

DELIMITER = "||"  # don't use a comma, as str(list) will naturally contain commas
TESTDIR = os.path.abspath(os.path.dirname(__file__))


def getBlockInfoTestTruthValues(dayObs: int) -> dict[tuple[str, int], str]:
    """Get the current truth values for the block information.

    Parameters
    ----------
    dayObs : `int`, optional
        The dayObs to get the truth values for.

    Returns
    -------
    data : `dict` [`tuple` [`int`, `int`], `str`]
        The block info truth data.
    """
    dataFilename = os.path.join(TESTDIR, "data", f"blockInfoData_{dayObs}.json")

    with open(dataFilename, "r") as f:
        loaded = json.loads(f.read())

    data = {}
    for dayObsSeqNumStr, line in loaded.items():
        blockNum = str(dayObsSeqNumStr.split(f"{DELIMITER}")[0])
        blockSeqNum = int(dayObsSeqNumStr.split(f"{DELIMITER}")[1])
        data[blockNum, blockSeqNum] = line
    return data


def writeNewBlockInfoTestTruthValues(dayObs: int) -> None:
    """This function is used to write out the truth values for the test cases.

    If bugs are found in the parsing, it's possible these values could change,
    and would need to be updated. If that happens, run this function, and check
    the new values into git.
    """
    blockParser = BlockParser(dayObs)

    data = {}
    for block in (blockParser).getBlockNums():
        seqNums = blockParser.getSeqNums(block)
        for seqNum in seqNums:
            blockInfo = blockParser.getBlockInfo(block, seqNum)
            assert blockInfo is not None
            line = (
                f"{blockInfo.blockId}{DELIMITER}"
                f"{blockInfo.begin}{DELIMITER}"
                f"{blockInfo.end}{DELIMITER}"
                f"{blockInfo.salIndices}{DELIMITER}"
                f"{blockInfo.tickets}{DELIMITER}"
                f"{len(blockInfo.states)}"
            )
            # must store as string not tuple for json serialization
            data[f"{block}{DELIMITER}{seqNum}"] = line

    dataFilename = os.path.join(TESTDIR, "data", f"blockInfoData_{dayObs}.json")
    with open(dataFilename, "w") as f:
        json.dump(data, f)


@unittest.skipIf(not HAS_EFD_CLIENT, "No EFD client available")
@vcr.use_cassette()
class BlockParserTestCase(lsst.utils.tests.TestCase):
    # class attributes populated in setUpClass
    client: Any
    dayObsNoTestCases: int
    dayObsWithCases: int
    dayObsNoBlocks: int
    blockParser: BlockParser
    blockNums: list[str]
    blockDict: dict[str, list[int]]

    @classmethod
    @vcr.use_cassette()
    def setUpClass(cls) -> None:
        try:
            cls.client = makeEfdClient(testing=True)
        except RuntimeError:
            raise unittest.SkipTest("Could not instantiate an EFD client")

        cls.dayObsNoTestCases = 20230615
        cls.dayObsWithCases = 20250420  # blocks = ['365', 'T282', 'T3', 'T379', 'T380', 'T4', 'T454']
        cls.dayObsNoBlocks = 20230531  # contains data but no blocks
        cls.blockParser = BlockParser(dayObs=cls.dayObsNoTestCases, client=cls.client)
        cls.blockNums = cls.blockParser.getBlockNums()
        cls.blockDict = {}
        for block in cls.blockNums:
            cls.blockDict[block] = cls.blockParser.getSeqNums(block)

    @vcr.use_cassette()
    def tearDown(self) -> None:
        loop = asyncio.get_event_loop()
        if self.client.influx_client is not None:
            loop.run_until_complete(self.client.influx_client.close())

    @vcr.use_cassette()
    def test_parsing(self) -> None:
        blockNums = self.blockParser.getBlockNums()
        self.assertTrue(all(isinstance(n, str)) for n in blockNums)
        self.assertEqual(blockNums, list(self.blockDict.keys()))

        for block, seqNums in self.blockDict.items():
            self.assertTrue(isinstance(block, str))
            self.assertIsInstance(seqNums, list)
            self.assertTrue(all(isinstance(s, int)) for s in seqNums)

            found = self.blockParser.getSeqNums(block)
            self.assertTrue(all(isinstance(s, int) for s in found))
            self.assertEqual(found, seqNums)
            self.blockParser.printBlockEvolution(block)

            for seqNum in seqNums:
                data = self.blockParser.getRows(block, seqNum)
                self.assertIsInstance(data, pd.DataFrame)
                self.assertGreater(len(data), 0)
                self.blockParser.getBlockInfo(block=block, seqNum=seqNum)
                self.blockParser.printBlockEvolution(block, seqNum=seqNum)

    @vcr.use_cassette()
    def test_notFoundBehavior(self) -> None:
        # no block data on this day so check init doesn't raise
        blockParser = BlockParser(dayObs=self.dayObsNoBlocks, client=self.client)
        self.assertIsInstance(blockParser, BlockParser)

        # check the queries which return nothing give nothing back gracefully
        blocks = blockParser.getBlockNums()
        self.assertIsInstance(blocks, list)
        self.assertEqual(len(blocks), 0)

        seqNums = blockParser.getSeqNums(block=123)
        self.assertIsInstance(seqNums, list)
        self.assertEqual(len(seqNums), 0)

        # just check this doesn't raise
        blockParser.getBlockInfo(block=1, seqNum=1)

        # now switch back to one with data, and make sure the same is true
        # when there is data present
        blockParser = self.blockParser
        seqNums = blockParser.getSeqNums(block=9999999)
        self.assertIsInstance(seqNums, list)
        self.assertEqual(len(seqNums), 0)

        # just check this doesn't raise
        blockParser.getBlockInfo(block=9999999, seqNum=9999999)

    @vcr.use_cassette()
    def test_actualValues(self) -> None:
        for dayObs in [self.dayObsNoTestCases, self.dayObsWithCases]:
            data = getBlockInfoTestTruthValues(dayObs)
            blockParser = BlockParser(dayObs, client=self.client)

            for block in blockParser.getBlockNums():
                seqNums = blockParser.getSeqNums(block)
                for seqNum in seqNums:
                    blockInfo = blockParser.getBlockInfo(block, seqNum)
                    line = data[blockInfo.blockNumber, blockInfo.seqNum]
                    items = line.split(f"{DELIMITER}")
                    self.assertEqual(items[0], blockInfo.blockId)
                    self.assertEqual(items[1], str(blockInfo.begin.value))
                    self.assertEqual(items[2], str(blockInfo.end.value))
                    self.assertEqual(items[3], str(blockInfo.salIndices))
                    self.assertEqual(items[4], str(blockInfo.tickets))
                    self.assertEqual(items[5], str(len(blockInfo.states)))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: object) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
