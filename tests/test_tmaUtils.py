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
import os
import unittest
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import TimeDelta
from utils import getVcr

import lsst.utils.tests
from lsst.summit.utils.dateTime import calcNextDay, getDayObsStartTime
from lsst.summit.utils.efdUtils import makeEfdClient
from lsst.summit.utils.enums import PowerState
from lsst.summit.utils.tmaUtils import (
    AxisMotionState,
    TMAEvent,
    TMAEventMaker,
    TMAState,
    TMAStateMachine,
    _initializeTma,
    filterBadValues,
    getAxisAndType,
    getAzimuthElevationDataForEvent,
    getCommandsDuringEvent,
    getSlewsFromEventList,
    getTracksFromEventList,
    plotEvent,
)

__all__ = [
    "writeNewTmaEventTestTruthValues",
]

TESTDIR = os.path.abspath(os.path.dirname(__file__))
vcr = getVcr()


def getTmaEventTestTruthValues() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the current truth values for the TMA event test cases.

    Returns
    -------
    seqNums : `np.array` of `int`
        The sequence numbers of the events.
    startRows : `np.array` of `int`
        The _startRow numbers of the events.
    endRows : `np.array` of `int`
        The _endRow numbers of the events.
    types : `np.array` of `str`
        The event types, as a string, i.e. the ``TMAEvent.name`` of the event's
        ``event.type``.
    endReasons : `np.array` of `str`
        The event end reasons, as a string, i.e. the ``TMAEvent.name`` of the
        event's ``event.endReason``.
    """
    dataFilename = os.path.join(TESTDIR, "data", "tmaEventData.txt")

    seqNums, startRows, endRows, types, endReasons = np.genfromtxt(
        dataFilename, delimiter=",", dtype=None, names=True, encoding="utf-8", unpack=True
    )
    return seqNums, startRows, endRows, types, endReasons


def writeNewTmaEventTestTruthValues() -> None:
    """This function is used to write out the truth values for the test cases.

    If the internal event creation logic changes, these values can change, and
    will need to be updated. Run this function, and check the new values into
    git.

    Note: if you have cause to update values with this function, make sure to
    update the version number on the TMAEvent class.
    """
    dayObs = 20241210  # obviously must match the day in the test class

    eventMaker = TMAEventMaker()
    events = eventMaker.getEvents(dayObs)

    dataFilename = os.path.join(TESTDIR, "data", "tmaEventData.txt")

    columnHeader = "seqNum,startRow,endRow,type,endReason"
    with open(dataFilename, "w") as f:
        f.write(columnHeader + "\n")
        for event in events:
            line = (
                f"{event.seqNum},{event._startRow},{event._endRow},{event.type.name},"
                f"{event.endReason.name}"
            )
            f.write(line + "\n")


def makeValid(tma: TMAStateMachine) -> None:
    """Helper function to turn a TMA into a valid state."""
    for name, value in tma._parts.items():
        if value == tma._UNINITIALIZED_VALUE:
            tma._parts[name] = 1


def _turnOn(tma: TMAStateMachine) -> None:
    """Helper function to turn TMA axes on for testing.

    Do not call directly in normal usage or code, as this just arbitrarily
    sets values to turn the axes on.

    Parameters
    ----------
    tma : `lsst.summit.utils.tmaUtils.TMAStateMachine`
        The TMA state machine model to initialize.
    """
    tma._parts["azimuthSystemState"] = PowerState.ON
    tma._parts["elevationSystemState"] = PowerState.ON


class TmaUtilsTestCase(lsst.utils.tests.TestCase):
    def test_tmaInit(self) -> None:
        tma = TMAStateMachine()
        self.assertFalse(tma._isValid)

        # setting one axis should not make things valid
        tma._parts["azimuthMotionState"] = 1
        self.assertFalse(tma._isValid)

        # setting all the other components should make things valid
        tma._parts["azimuthInPosition"] = 1
        tma._parts["azimuthSystemState"] = 1
        tma._parts["elevationInPosition"] = 1
        tma._parts["elevationMotionState"] = 1
        tma._parts["elevationSystemState"] = 1
        self.assertTrue(tma._isValid)

    def test_tmaReferences(self) -> None:
        """Check the linkage between the component lists and the _parts
        dict.
        """
        tma = TMAStateMachine()

        # setting one axis should not make things valid
        self.assertEqual(tma._parts["azimuthMotionState"], tma._UNINITIALIZED_VALUE)
        self.assertEqual(tma._parts["elevationMotionState"], tma._UNINITIALIZED_VALUE)
        tma.motion[0] = AxisMotionState.TRACKING  # set azimuth to 0
        tma.motion[1] = AxisMotionState.TRACKING  # set azimuth to 0
        self.assertEqual(tma._parts["azimuthMotionState"], AxisMotionState.TRACKING)
        self.assertEqual(tma._parts["elevationMotionState"], AxisMotionState.TRACKING)

    def test_getAxisAndType(self) -> None:
        # check both the long and short form names work
        for s in ["azimuthMotionState", "lsst.sal.MTMount.logevent_azimuthMotionState"]:
            self.assertEqual(getAxisAndType(s), ("azimuth", "MotionState"))

        # check in position, and use elevation instead of azimuth to test that
        for s in ["elevationInPosition", "lsst.sal.MTMount.logevent_elevationInPosition"]:
            self.assertEqual(getAxisAndType(s), ("elevation", "InPosition"))

        for s in ["azimuthSystemState", "lsst.sal.MTMount.logevent_azimuthSystemState"]:
            self.assertEqual(getAxisAndType(s), ("azimuth", "SystemState"))

    def test_initStateLogic(self) -> None:
        tma = TMAStateMachine()
        self.assertFalse(tma._isValid)
        self.assertFalse(tma.isMoving)
        self.assertFalse(tma.canMove)
        self.assertFalse(tma.isTracking)
        self.assertFalse(tma.isSlewing)
        self.assertEqual(tma.state, TMAState.UNINITIALIZED)

        _initializeTma(tma)  # we're valid, but still aren't moving and can't
        self.assertTrue(tma._isValid)
        self.assertNotEqual(tma.state, TMAState.UNINITIALIZED)
        self.assertTrue(tma.canMove)
        self.assertTrue(tma.isNotMoving)
        self.assertFalse(tma.isMoving)
        self.assertFalse(tma.isTracking)
        self.assertFalse(tma.isSlewing)

        _turnOn(tma)  # can now move, still valid, but not in motion
        self.assertTrue(tma._isValid)
        self.assertTrue(tma.canMove)
        self.assertTrue(tma.isNotMoving)
        self.assertFalse(tma.isMoving)
        self.assertFalse(tma.isTracking)
        self.assertFalse(tma.isSlewing)

        # consider manipulating the axes by hand here and testing these?
        # it's likely not worth it, given how much this exercised elsewhere,
        # but these are the only functions not yet being directly tested
        # tma._axesInFault()
        # tma._axesOff()
        # tma._axesOn()
        # tma._axesInMotion()
        # tma._axesTRACKING()
        # tma._axesInPosition()


@vcr.use_cassette()
class TMAEventMakerTestCase(lsst.utils.tests.TestCase):
    # class attributes populated in setUpClass
    client: Any
    dayObs: int
    dayObsWithBlockInfo: int
    tmaEventMaker: TMAEventMaker
    events: list[TMAEvent]
    sampleData: Any

    @classmethod
    @vcr.use_cassette()
    def setUpClass(cls) -> None:
        try:
            cls.client = makeEfdClient(testing=True)
        except RuntimeError:
            raise unittest.SkipTest("Could not instantiate an EFD client")

        cls.dayObs = 20241210
        cls.dayObsWithBlockInfo = 20230615
        # get a sample expRecord here to test expRecordToTimespan
        cls.tmaEventMaker = TMAEventMaker(cls.client)
        cls.events = cls.tmaEventMaker.getEvents(cls.dayObs)  # does the fetch
        cls.sampleData = cls.tmaEventMaker._data[cls.dayObs]  # pull the data from the object and test length

    @vcr.use_cassette()
    def tearDown(self) -> None:
        loop = asyncio.get_event_loop()
        if self.client.influx_client is not None:
            loop.run_until_complete(self.client.influx_client.close())

    @vcr.use_cassette()
    def test_events(self) -> None:
        data = self.sampleData
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 800)

    @vcr.use_cassette()
    def test_rowDataForValues(self) -> None:
        rowsFor = set(self.sampleData["rowFor"])
        self.assertEqual(len(rowsFor), 6)

        # hard coding these ensures that you can't extend the axes/model
        # without being explicit about it here.
        correct = {
            "azimuthInPosition",
            "azimuthMotionState",
            "azimuthSystemState",
            "elevationInPosition",
            "elevationMotionState",
            "elevationSystemState",
        }
        self.assertSetEqual(rowsFor, correct)

    @vcr.use_cassette()
    def test_monotonicTimeInDataframe(self) -> None:
        # ensure that each row is later than the previous
        times = self.sampleData["private_efdStamp"]
        self.assertTrue(np.all(np.diff(times) > 0))

    @vcr.use_cassette()
    def test_monotonicTimeApplicationOfRows(self) -> None:
        # ensure you can apply rows in the correct order
        tma = TMAStateMachine()
        row1 = self.sampleData.iloc[0]
        row2 = self.sampleData.iloc[1]

        # just running this check it is OK
        tma.apply(row1)
        tma.apply(row2)

        # and that if you apply them in reverse order then things will raise
        tma = TMAStateMachine()
        with self.assertRaises(ValueError):
            tma.apply(row2)
            tma.apply(row1)

    @vcr.use_cassette()
    def test_fullDaySequence(self) -> None:
        # make sure we can apply all the data from the day without falling
        # through the logic sieve
        for engineering in (True, False):
            tma = TMAStateMachine(engineeringMode=engineering)

            _initializeTma(tma)

            for rowNum, row in self.sampleData.iterrows():
                tma.apply(row)

    @vcr.use_cassette()
    def test_endToEnd(self) -> None:
        eventMaker = self.tmaEventMaker
        events = eventMaker.getEvents(self.dayObs)
        self.assertIsInstance(events, list)
        self.assertEqual(len(events), 320)
        self.assertIsInstance(events[0], TMAEvent)

        slews = [e for e in events if e.type == TMAState.SLEWING]
        tracks = [e for e in events if e.type == TMAState.TRACKING]
        self.assertEqual(len(slews), 172)
        self.assertEqual(len(tracks), 148)

        seqNums, startRows, endRows, types, endReasons = getTmaEventTestTruthValues()
        for eventNum, event in enumerate(events):
            self.assertEqual(event.seqNum, seqNums[eventNum])
            self.assertEqual(event._startRow, startRows[eventNum])
            self.assertEqual(event._endRow, endRows[eventNum])
            self.assertEqual(event.type.name, types[eventNum])
            self.assertEqual(event.endReason.name, endReasons[eventNum])

        eventSet = set(slews)  # check we can hash
        eventSet.update(slews)  # check it ignores duplicates
        self.assertEqual(len(eventSet), len(slews))

    @vcr.use_cassette()
    def test_noDataBehaviour(self) -> None:
        eventMaker = self.tmaEventMaker
        noDataDayObs = 19600101  # do not use 19700101 - there is data for that day!
        with self.assertLogs(level="WARNING") as cm:
            correctMsg = f"No EFD data found for dayObs={noDataDayObs}"
            events = eventMaker.getEvents(noDataDayObs)
            self.assertIsInstance(events, list)
            self.assertEqual(len(events), 0)
            msg = cm.output[0]
            self.assertIn(correctMsg, msg)

    @vcr.use_cassette()
    def test_helperFunctions(self) -> None:
        eventMaker = self.tmaEventMaker
        events = eventMaker.getEvents(self.dayObs)

        slews = [e for e in events if e.type == TMAState.SLEWING]
        tracks = [e for e in events if e.type == TMAState.TRACKING]
        foundSlews = getSlewsFromEventList(events)
        foundTracks = getTracksFromEventList(events)
        self.assertEqual(slews, foundSlews)
        self.assertEqual(tracks, foundTracks)

    def test_filterBadValues(self) -> None:
        # NB: if you add enough spurious values that the median is no longer
        # the value around which your "good" values are oscillating the first
        # two points will get replaced and this can be very confusing!

        # test no bad values
        # mean = median = 1.0
        values = np.array([1.0, 0.96, 1.0, 1.04, 0.95, 1.0, 1.05, 1.0, 1.05, 1.0, 0.95])
        mean = np.mean(values)
        nReplaced = filterBadValues(values)
        self.assertEqual(nReplaced, 0)
        self.assertEqual(np.mean(values), mean)

        # test with one bad values
        values = np.array([1.0, 0.96, 1.0, 1.04, 2.95, 1.0, 1.05, 1.0, 1.05, 1.0, 0.95])
        nReplaced = filterBadValues(values)
        self.assertEqual(nReplaced, 1)

        # test with two consecutive bad values
        values = np.array([1.0, 0.96, 1.0, 1.04, 2.95, 3.0, 1.05, 1.0, 1.05, 1.0, 0.95])
        nReplaced = filterBadValues(values)
        self.assertEqual(nReplaced, 2)

        # test with three consecutive bad values
        values = np.array([1.0, 0.96, 1.0, 1.04, 2.95, 3.0, 4.05, 1.0, 1.05, 1.0, 0.95])
        nReplaced = filterBadValues(values)
        self.assertEqual(nReplaced, 3)

        # test with three consecutive bad values and another at the end
        values = np.array([1.0, 0.96, 1.0, 1.04, 2.95, 3.0, 4.05, 1.0, 1.05, 1.0, 3.95])
        nReplaced = filterBadValues(values)
        self.assertEqual(nReplaced, 4)

        # test with more than three consecutive bad values
        values = np.array([1.0, 0.96, 1.0, 1.04, 2.95, 3.0, 4.05, 5.0, 1.05, 1.0, 0.95])
        nReplaced = filterBadValues(values)
        self.assertEqual(nReplaced, 3)
        self.assertIn(5.0, values)  # check the last bad value is still there specifically

        # test with more than three consecutive bad values and another bad
        # value at the end
        values = np.array([1.0, 0.96, 1.0, 1.04, 2.95, 3.0, 4.05, 5.0, 1.05, 1.0, 2.95])
        nReplaced = filterBadValues(values)
        self.assertEqual(nReplaced, 4)

        # test with bad values in first two positions
        values = np.array([2.0, 1.96, 1.0, 1.04, 0.95, 1.0, 1.05, 1.0, 1.05, 1.0, 0.95])  # median = 1.0
        nReplaced = filterBadValues(values)
        self.assertEqual(nReplaced, 2)

        # test with bad values in first two positions and one in the middle
        values = np.array([2.0, 1.96, 1.0, 1.04, 0.95, 5.0, 1.04, 1.0, 1.05, 1.0, 0.95])
        nReplaced = filterBadValues(values)
        self.assertEqual(nReplaced, 3)

        # check that the last two good values are always used for correction,
        # including when there are more than three consecutive bad values.
        values = np.array([1.0, 0.96, 1.0, 1.02, 2.95, 3.0, 4.05, 5.0, 1.05, 1.0, 2.95])
        expected = np.array([1.0, 0.96, 1.0, 1.02, 1.01, 1.01, 1.01, 5.0, 1.05, 1.0, 1.025])
        nReplaced = filterBadValues(values)
        residuals = np.abs(values - expected)
        self.assertEqual(nReplaced, 4)
        self.assertTrue(np.all(residuals < 1e-6))

        # check with one good point after an overflowing run of bad to make
        # sure the correction is always applied with good values, not the naive
        # average of the last two even if they might be bad
        values = np.array([1.0, 0.96, 1.0, 1.02, 2.95, 3.0, 4.05, 5.0, 1.05, 2.95, 1.0])
        expected = np.array([1.0, 0.96, 1.0, 1.02, 1.01, 1.01, 1.01, 5.0, 1.05, 1.035, 1.0])
        nReplaced = filterBadValues(values)
        residuals = np.abs(values - expected)
        self.assertEqual(nReplaced, 4)
        self.assertTrue(np.all(residuals < 1e-6))

        # check with non-default maxDelta
        values = np.array([1.0, 0.96, 1.0, 1.02, 2.95, 3.0, 4.05, 5.0, 1.05, 1.0, 2.95])
        nReplaced = filterBadValues(values, maxDelta=10)
        self.assertEqual(nReplaced, 0)

        values = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.1,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )
        nReplaced = filterBadValues(values, maxDelta=0.01)
        self.assertEqual(nReplaced, 1)

    @vcr.use_cassette()
    def test_getEvent(self) -> None:
        # test the singular event getter, and what happens if the event doesn't
        # exist for the day
        eventMaker = self.tmaEventMaker
        events = eventMaker.getEvents(self.dayObs)
        nEvents = len(events)

        event = eventMaker.getEvent(self.dayObs, 0)
        self.assertIsInstance(event, TMAEvent)
        self.assertEqual(event, events[0])
        event = eventMaker.getEvent(self.dayObs, 100)
        self.assertIsInstance(event, TMAEvent)
        self.assertEqual(event, events[100])

        with self.assertLogs(level="WARNING") as cm:
            correctMsg = f"Event {nEvents + 1} not found for {self.dayObs}"
            event = eventMaker.getEvent(self.dayObs, nEvents + 1)
            msg = cm.output[0]
            self.assertIn(correctMsg, msg)

    @vcr.use_cassette()
    def test_printing(self) -> None:
        eventMaker = self.tmaEventMaker
        events = eventMaker.getEvents(self.dayObs)

        # test str(), repr(), and _ipython_display_() for an event
        print(str(events[0]))
        print(repr(events[0]))
        print(events[0]._ipython_display_())

        # spot-check both a slow and a track to print
        slews = [e for e in events if e.type == TMAState.SLEWING]
        tracks = [e for e in events if e.type == TMAState.TRACKING]
        eventMaker.printEventDetails(slews[0])
        eventMaker.printEventDetails(tracks[0])
        eventMaker.printEventDetails(events[-1])

        # check the full day trick works
        eventMaker.printFullDayStateEvolution(self.dayObs)

        tma = TMAStateMachine()
        _initializeTma(tma)  # the uninitialized state contains wrong types for printing
        eventMaker.printTmaDetailedState(tma)

    @vcr.use_cassette()
    def test_getAxisData(self) -> None:
        eventMaker = self.tmaEventMaker
        events = eventMaker.getEvents(self.dayObs)

        azData, elData = getAzimuthElevationDataForEvent(self.client, events[0])
        self.assertIsInstance(azData, pd.DataFrame)
        self.assertIsInstance(elData, pd.DataFrame)

        paddedAzData, paddedElData = getAzimuthElevationDataForEvent(
            self.client, events[0], prePadding=2, postPadding=1
        )
        self.assertGreater(len(paddedAzData), len(azData))
        self.assertGreater(len(paddedElData), len(elData))

        # just check this doesn't raise when called, and check we can pass the
        # data in
        plotEvent(self.client, events[0], azimuthData=azData, elevationData=elData)

    @vcr.use_cassette()
    def test_plottingAndCommands(self) -> None:
        eventMaker = self.tmaEventMaker
        events = eventMaker.getEvents(self.dayObs)
        event = events[10]  # this one has commands, and we'll check that later

        # check we _can_ plot without a figure, and then stop doing that
        plotEvent(self.client, event)

        fig = plt.figure(figsize=(10, 8))
        # just check this doesn't raise when called
        plotEvent(self.client, event, fig=fig)
        plt.close(fig)

        commandsToPlot = ["raDecTarget", "moveToTarget", "startTracking", "stopTracking"]
        commands = getCommandsDuringEvent(self.client, event, commandsToPlot, doLog=False)
        self.assertTrue(not all([time is None for time in commands.values()]))  # at least one command

        plotEvent(self.client, event, fig=fig, commands=commands)

        del fig

    @vcr.use_cassette()
    def test_findEvent(self) -> None:
        eventMaker = self.tmaEventMaker
        # addBlockInfo=True because it shouldn't affect the comparison, and
        # this also then ensures that the code is exercised too
        events = eventMaker.getEvents(self.dayObs, addBlockInfo=True)
        event = events[28]  # this one has a contiguous event before it

        time = event.begin
        found = eventMaker.findEvent(time)
        self.assertEqual(found, event)

        dt = TimeDelta(0.01, format="sec")
        # must be just inside to get the same event back, because if a moment
        # is shared it gives the one which starts with the moment (whilst
        # logging info messages about it)
        time = event.end - dt
        found = eventMaker.findEvent(time)
        self.assertEqual(found, event)

        # now check that if we're a hair after, we don't get the same event
        time = event.end + dt
        found = eventMaker.findEvent(time)
        self.assertNotEqual(found, event)

        # Now check the cases which don't find an event at all. It would be
        # nice to check the log messages here, but it seems too fragile to be
        # worth it
        dt = TimeDelta(1, format="sec")
        tooEarlyOnDay = getDayObsStartTime(self.dayObs) + dt  # 1 second after start of day
        found = eventMaker.findEvent(tooEarlyOnDay)
        self.assertIsNone(found)

        # 1 second before end of day and this day does not end with an open
        # event
        tooLateOnDay = getDayObsStartTime(calcNextDay(self.dayObs)) - dt
        found = eventMaker.findEvent(tooLateOnDay)
        self.assertIsNone(found)

        # going just inside the last event of the day should be fine
        lastEvent = events[-1]
        found = eventMaker.findEvent(lastEvent.end - dt)
        self.assertEqual(found, lastEvent)

        # going at the very end of the last event of the day should actually
        # find nothing, because the last moment of an event isn't actually in
        # the event itself, because of how contiguous events are defined to
        # behave (being half-open intervals)
        found = eventMaker.findEvent(lastEvent.end)
        self.assertIsNone(found, lastEvent)

    @vcr.use_cassette()
    def test_eventAssociatedWith(self) -> None:
        eventMaker = self.tmaEventMaker
        events = eventMaker.getEvents(self.dayObsWithBlockInfo)
        eventsWithBlockInfo = [e for e in events if e.blockInfos]
        eventsWithoutBlockInfo = [e for e in events if not e.blockInfos]
        self.assertEqual(len(events), 69)
        self.assertEqual(len(eventsWithBlockInfo), 65)
        self.assertEqual(len(eventsWithoutBlockInfo), 4)

        self.assertIsNotNone(eventsWithoutBlockInfo[0].blockInfos)
        self.assertIsInstance(eventsWithoutBlockInfo[0].blockInfos, list)
        self.assertEqual(len(eventsWithoutBlockInfo[0].blockInfos), 0)

        event = eventsWithBlockInfo[0]
        self.assertIsInstance(event, TMAEvent)
        self.assertTrue(event.associatedWith(ticket="SITCOM-906"))
        self.assertFalse(event.associatedWith(ticket="SITCOM-905"))

        self.assertTrue(event.associatedWith(salIndex=100017))
        self.assertFalse(event.associatedWith(salIndex=100018))

        self.assertTrue(event.associatedWith(block=6))
        self.assertFalse(event.associatedWith(block=5))

        # check it works with any and all of the arguments
        self.assertTrue(event.associatedWith(block=6, salIndex=100017))
        self.assertTrue(event.associatedWith(block=6, salIndex=100017, ticket="SITCOM-906"))

        # check it's false if any are false
        self.assertFalse(event.associatedWith(block=7, salIndex=100017, ticket="SITCOM-906"))  # 1 wrong
        self.assertFalse(event.associatedWith(block=6, salIndex=100018, ticket="SITCOM-906"))  # 1 wrong
        self.assertFalse(event.associatedWith(block=6, salIndex=100017, ticket="SITCOM-907"))  # 1 wrong
        self.assertFalse(event.associatedWith(block=1, salIndex=1, ticket="SITCOM-1"))  # all wrong

        # check with the blockSeqNum, with and without the other items
        self.assertTrue(event.associatedWith(block=6, blockSeqNum=1))
        self.assertFalse(event.associatedWith(block=6, blockSeqNum=2))
        self.assertTrue(event.associatedWith(block=6, blockSeqNum=1, salIndex=100017))
        self.assertFalse(event.associatedWith(block=6, blockSeqNum=2, salIndex=100017))
        self.assertTrue(event.associatedWith(block=6, blockSeqNum=1, salIndex=100017, ticket="SITCOM-906"))
        self.assertFalse(event.associatedWith(block=6, blockSeqNum=2, salIndex=100017, ticket="SITCOM-906"))

        with self.assertRaises(ValueError):
            event.associatedWith()
            event.associatedWith(blockSeqNum=1)  # nonsense to ask for a seqNum without a block number


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: object) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
