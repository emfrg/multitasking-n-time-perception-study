#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019)
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195.
        https://doi.org/10.3758/s13428-018-01193-y

"""

from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins

plugins.activatePlugins()
prefs.hardware["audioLib"] = "ptb"
prefs.hardware["audioLatencyMode"] = "3"
from psychopy import (
    sound,
    gui,
    visual,
    core,
    data,
    event,
    logging,
    clock,
    colors,
    layout,
    hardware,
)
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED,
    STARTED,
    PLAYING,
    PAUSED,
    STOPPED,
    FINISHED,
    PRESSED,
    RELEASED,
    FOREVER,
    priority,
)

import numpy as np
from numpy import (
    sin,
    cos,
    tan,
    log,
    log10,
    pi,
    average,
    sqrt,
    std,
    deg2rad,
    rad2deg,
    linspace,
    asarray,
)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os
import sys

from psychopy.hardware import keyboard


_thisDir = os.path.dirname(os.path.abspath(__file__))

psychopyVersion = "2024.2.4"
expName = "Multitasking_Experiment_UU"

expInfo = {
    "participant": f"{randint(111111, 999999):06.0f}",
    "date|hid": data.getDateStr(),
    "expName|hid": expName,
    "psychopyVersion|hid": psychopyVersion,
}

PILOTING = False
_fullScr = True
_winSize = [1800, 1169]


try:
    parsePipeSyntax = data.utils.parsePipeSyntax
except (AttributeError, ImportError):
    try:
        from psychopy.gui.util import parsePipeSyntax
    except ImportError:

        def parsePipeSyntax(key, stripKey=True):
            key = key.split("|", 1)[0]
            return (key.strip() if stripKey else key, [])


# Compatibility patches for different PsychoPy versions
from psychopy import data
from psychopy.constants import NOT_STARTED, STARTED, FINISHED

try:
    _ = data.Routine
except AttributeError:
    try:
        from psychopy.data.routines import Routine as _Routine
    except ImportError:
        try:
            from psychopy.data.routine import Routine as _Routine
        except ImportError:

            class _Routine:
                """Minimal stand-in for Builder's data.Routine."""

                def __init__(self, *, name="unnamed", components=None):
                    self.name = name
                    self.components = list(components or [])

                    self.status = NOT_STARTED
                    self.forceEnded = False
                    self.maxDuration = None
                    self.maxDurationReached = False
                    self.tStart = None
                    self.tStop = None
                    self.tStartRefresh = None
                    self.tStopRefresh = None
                    self.frameNStart = None
                    self.frameNStop = None

                def __iter__(self):
                    return iter(self.components)

                def __bool__(self):
                    return self.status == STARTED

            _Routine.__module__ = "psychopy.data"

    data.Routine = _Routine


# Patch to ensure TrialHandler2 rows support .thisN access
from psychopy import data

if not hasattr(data, "TrialHandler2"):
    data.TrialHandler2 = data.TrialHandler


class _Row(dict):
    """Dict with attribute access and .thisN property."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    __setattr__ = dict.__setitem__


def _wrap_rows(self):
    """Convert all trialList entries to _Row objects with .thisN attribute."""
    for i, row in enumerate(self.trialList):
        if isinstance(row, _Row):
            continue

        if row is None:
            row = {}

        if isinstance(row, dict):
            wrapped = _Row(row)
            wrapped.thisN = i
            self.trialList[i] = wrapped
        else:
            self.trialList[i] = _Row(thisN=i)


if not getattr(data.TrialHandler2, "_acePatched", False):
    _orig_init = data.TrialHandler2.__init__

    def _init_and_wrap(self, *a, **k):
        _orig_init(self, *a, **k)
        _wrap_rows(self)

    data.TrialHandler2.__init__ = _init_and_wrap
    data.TrialHandler2._acePatched = True


# Patch to handle unrecognized kwargs in older PsychoPy versions
from psychopy import visual, data
import inspect, functools


def _strip_unknown_kwargs(cls, extra_allowed=()):
    """
    Wrap cls.__init__ to remove unrecognised keyword arguments.

    Parameters:
        cls: Class to patch
        extra_allowed: Tuple of parameter names to keep regardless
    """
    real_init = cls.__init__
    sig = inspect.signature(real_init)
    known = set(sig.parameters)

    @functools.wraps(real_init)
    def wrapper(self, *args, **kwargs):
        unknown = [k for k in kwargs if k not in known and k not in extra_allowed]
        for k in unknown:
            kwargs.pop(k, None)
        return real_init(self, *args, **kwargs)

    cls.__init__ = wrapper


for stim in (visual.TextStim, visual.Rect, visual.Circle, visual.Line):
    _strip_unknown_kwargs(stim)

_strip_unknown_kwargs(data.ExperimentHandler)


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.

    Parameters:
        expInfo : dict
            Information about this experiment.

    Returns:
        dict: Information about this experiment.
    """
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    """
    for key in list(expInfo):
        newKey, _ = parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)

    if dataDir is None:
        dataDir = _thisDir
    filename = f"data/{expInfo['participant']}_{expName}_{expInfo['date']}"

    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)

    thisExp = data.ExperimentHandler(
        name=expName,
        version="",
        extraInfo=expInfo,
        runtimeInfo=None,
        originPath=os.path.abspath(__file__),
        savePickle=True,
        saveWideText=True,
        dataFileName=os.path.join(dataDir, filename),
        sortColumns="time",
    )
    thisExp.setPriority("thisRow.t", priority.CRITICAL)
    thisExp.setPriority("expName", priority.LOW)

    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.

    Parameters:
        filename : str or pathlib.Path
            Filename to save log file and data files as, doesn't need an extension.

    Returns:
        psychopy.logging.LogFile: Text stream to receive inputs from the logging system.
    """
    import logging as _stdlog

    if PILOTING:
        lvl = prefs.piloting["pilotConsoleLoggingLevel"]
    else:
        lvl = "warning"
    if isinstance(lvl, str):
        lvl = getattr(_stdlog, lvl.upper(), _stdlog.INFO)
    logging.console.setLevel(lvl)

    logFile = logging.LogFile(f"{filename}.log")
    if PILOTING:
        lvlFile = prefs.piloting["pilotLoggingLevel"]
    else:
        lvlFile = "info"
    if isinstance(lvlFile, str):
        lvlFile = getattr(_stdlog, lvlFile.upper(), _stdlog.INFO)
    logFile.setLevel(lvlFile)

    return logFile


# PATCH: Problem was that cursor was turning into I-beam
def _force_arrow_cursor(win):
    """Ensure OS cursor is the default arrow for pyglet-backed windows."""
    wh = getattr(win, "winHandle", None)
    if not wh or not hasattr(wh, "get_system_mouse_cursor"):
        return
    try:
        arrow = wh.get_system_mouse_cursor(wh.CURSOR_DEFAULT)
    except Exception:
        return

    @wh.event
    def on_mouse_enter(x, y):
        wh.set_mouse_cursor(arrow)

    @wh.event
    def on_mouse_motion(x, y, dx, dy):
        wh.set_mouse_cursor(arrow)

    @wh.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        wh.set_mouse_cursor(arrow)


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window

    Parameters:
        expInfo : dict
            Information about this experiment, created by the `showExpInfoDlg` function.
        win : psychopy.visual.Window or None
            If given, configures this window; otherwise creates a new one.

    Returns:
        psychopy.visual.Window: Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug("Fullscreen settings ignored as running in pilot mode.")

    if win is None:
        win = visual.Window(
            size=_winSize,
            fullscr=_fullScr,
            screen=0,
            winType="pyglet",
            allowGUI=False,
            allowStencil=False,
            monitor="testMonitor",
            color="white",
            colorSpace="rgb",
            backgroundImage="",
            backgroundFit="none",
            blendMode="avg",
            useFBO=True,
            units="height",
            checkTiming=False,
        )
    else:
        win.color = [0, 0, 0]
        win.colorSpace = "rgb"
        win.backgroundImage = ""
        win.backgroundFit = "none"
        win.units = "height"

    if expInfo is not None:
        if getattr(win, "_monitorFrameRate", None) is None:
            try:
                win._monitorFrameRate = win.getActualFrameRate(
                    infoMsg="Running internal processes, please wait..."
                )
            except TypeError:
                logging.debug("Measuring frame rate without infoMsg...")
                win._monitorFrameRate = win.getActualFrameRate()
        expInfo["frameRate"] = win._monitorFrameRate

    win.hideMessage()

    # PATCH: Enforce classic arrow cursor for this window
    _force_arrow_cursor(win)

    return win


def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.

    Parameters:
        thisExp : psychopy.data.ExperimentHandler
            Handler object for this experiment, contains the data to save and information about
            where to save it to.
        win : psychopy.visual.Window
            Window for this experiment.
        timers : list, tuple
            List of timers to reset once pausing is finished.
        playbackComponents : list, tuple
            List of any components with a `pause` method which need to be paused.
    """
    if thisExp.status != PAUSED:
        return

    pauseTimer = core.Clock()
    for comp in playbackComponents:
        comp.pause()

    from psychopy.hardware import keyboard

    defaultKeyboard = keyboard.Keyboard(backend="psychtoolbox")

    while thisExp.status == PAUSED:
        if defaultKeyboard.getKeys(keyList=["escape"]):
            endExperiment(thisExp, win=win)
        clock.time.sleep(0.001)

    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)

    for comp in playbackComponents:
        comp.play()

    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.

    Parameters:
        expInfo : dict
            Information about this experiment, created by the `setupExpInfo` function.
        thisExp : psychopy.data.ExperimentHandler
            Handler object for this experiment, contains the data to save and information about
            where to save it to.
        win : psychopy.visual.Window
            Window in which to run this experiment.
        globalClock : psychopy.core.clock.Clock or None
            Clock to get global time from - supply None to make a new one.
        thisSession : psychopy.session.Session or None
            Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.status = STARTED
    win.winHandle.activate()
    exec = environmenttools.setExecEnvironment(globals())

    from psychopy.hardware import keyboard

    defaultKeyboard = keyboard.Keyboard(backend="psychtoolbox")

    os.chdir(_thisDir)
    filename = thisExp.dataFileName
    frameTolerance = 0.001
    endExpNow = False

    if "frameRate" in expInfo and expInfo["frameRate"] is not None:
        frameDur = 1.0 / round(expInfo["frameRate"])
    else:
        frameDur = 1.0 / 60.0

    from random import sample, uniform

    # Initialize components for Routine "global_definitions"
    dummy = visual.TextStim(
        win=win,
        name="dummy",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # Run 'Begin Experiment' code from define_classes

    # ------------------------------------------------------------------
    # Generic base: every drawable component inherits from this
    # ------------------------------------------------------------------
    class ExperimentComponent:
        """Base class that gives every visual object .show() / .hide()."""

        def __init__(self):
            # Holds anything that has an .autoDraw attribute
            self._elements = []

        # ----- helpers -------------------------------------------------
        def _register(self, *els):
            """Add one or many PsychoPy stimuli or lists/tuples thereof."""
            for el in els:
                if el is None:
                    continue
                if isinstance(el, (list, tuple)):
                    self._elements.extend(el)
                else:
                    self._elements.append(el)

        # ----- public API ----------------------------------------------
        def show(self):
            for el in self._elements:
                el.autoDraw = True

        def hide(self):
            for el in self._elements:
                el.autoDraw = False

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    class Keyboard(ExperimentComponent):
        """
        Letter keyboard that generates ("letter", value) or ("delete", None) events.
        """

        KEYBOARD_POSITION = (0.0, -0.2)

        def __init__(
            self,
            win,
            keyboard_position=KEYBOARD_POSITION,
            button_width=0.065,
            button_height=0.065,
            button_text_height=0.039,
            input_display_height=0.065,
            input_display_pos=(0, 0.20),
            button_fill_color="lightgrey",
            button_line_color="black",
            button_text_color="black",
            input_text_color="black",
            button_font="Arial",
            show_screen=False,
            show_delete=False,
            enable_click_feedback=True,
            enable_hover_feedback=False,
            hover_color="#e0e0e0",
            click_color="#c0c0c0",
            feedback_duration=0.15,
        ):
            super().__init__()
            self.win = win
            self.keyboard_position = keyboard_position
            self.button_width = button_width
            self.button_height = button_height
            self.button_text_height = button_text_height
            self.input_display_height = input_display_height
            self.input_display_pos = input_display_pos
            self.show_screen = show_screen
            self.show_delete = show_delete

            # Feedback parameters
            self.enable_click_feedback = enable_click_feedback
            self.enable_hover_feedback = enable_hover_feedback
            self.hover_color = hover_color
            self.click_color = click_color
            self.feedback_duration = feedback_duration
            self.normal_color = button_fill_color

            # State for hover and click effects
            self.hovered_button = None
            self.clicked_button = None
            self.click_timer = core.Clock()

            # Letter grid layout
            self.letter_grid = [
                ["A", "B", "C", "D", "E", "F", "G"],
                ["H", "I", "J", "K", "L", "M", "N"],
                ["O", "P", "Q", "R", "S", "T", "U"],
                ["V", "W", "X", "Y", "Z"],
            ]

            self.button_fill_color = button_fill_color
            self.button_line_color = button_line_color
            self.button_text_color = button_text_color
            self.input_text_color = input_text_color
            self.button_font = button_font

            self.keyboard_buttons = []
            self.button_elements = []
            self.input_string = ""
            self.events = []
            self.mouse_clicked = False

            self._create_buttons()
            if self.show_screen:
                self._create_title()
                self._create_input_display()
            self._create_borders()
            self._create_delete_button()
            self.hide()

        def _create_buttons(self):
            button_spacing = 0.013

            # Calculate horizontal spacing for each row to center it
            row_widths = [
                len(row) * self.button_width + (len(row) - 1) * button_spacing
                for row in self.letter_grid
            ]
            max_width = max(row_widths)

            # Create buttons for each letter in the grid
            for i, row in enumerate(self.letter_grid):
                row_width = row_widths[i]
                row_left_x = -row_width / 2.0

                for j, letter in enumerate(row):
                    x = (
                        row_left_x
                        + j * (self.button_width + button_spacing)
                        + self.button_width / 2.0
                    )
                    y = 0.104 - i * (self.button_height + button_spacing)

                    self.keyboard_buttons.append(
                        {
                            "label": letter,
                            "pos": (
                                x + self.keyboard_position[0],
                                y + self.keyboard_position[1],
                            ),
                        }
                    )

            # Create the visual elements for each button
            for btn in self.keyboard_buttons:
                rect = visual.Rect(
                    win=self.win,
                    pos=btn["pos"],
                    width=self.button_width,
                    height=self.button_height,
                    fillColor=self.button_fill_color,
                    lineColor=self.button_line_color,
                )
                txt = visual.TextStim(
                    win=self.win,
                    pos=btn["pos"],
                    text=btn["label"],
                    height=self.button_text_height,
                    color=self.button_text_color,
                    font=self.button_font,
                )
                self.button_elements.append(
                    {"label": btn["label"], "rect": rect, "text": txt}
                )
                self._register(rect, txt)

        def _create_delete_button(self):
            if self.show_delete:
                # Position delete button with scaled dimensions
                delete_x = 0.3 + self.keyboard_position[0]
                delete_y = -0.26 + self.keyboard_position[1]

                delete_rect = visual.Rect(
                    win=self.win,
                    pos=(delete_x, delete_y),
                    width=0.13,
                    height=self.button_height,
                    fillColor=self.button_fill_color,
                    lineColor=self.button_line_color,
                )
                delete_text = visual.TextStim(
                    win=self.win,
                    pos=(delete_x, delete_y),
                    text="DELETE",
                    height=self.button_text_height * 0.8,
                    color=self.button_text_color,
                    font=self.button_font,
                )

                self.delete_button = {
                    "label": "DELETE",
                    "rect": delete_rect,
                    "text": delete_text,
                }
                self._register(delete_rect, delete_text)
            else:
                self.delete_button = None

        def _create_title(self):
            self.title_stim = visual.TextStim(
                win=self.win,
                text="Keyboard",
                pos=(self.keyboard_position[0], 0.28 + self.keyboard_position[1]),
                height=0.035,
                color=self.input_text_color,
                bold=True,
            )
            self._register(self.title_stim)

        def _create_input_display(self):
            self.inputDisplay = visual.TextStim(
                win=self.win,
                text=" ",
                pos=(
                    self.input_display_pos[0] + self.keyboard_position[0],
                    self.input_display_pos[1] + self.keyboard_position[1],
                ),
                height=self.input_display_height,
                color=self.input_text_color,
            )
            self._register(self.inputDisplay)

        def _create_borders(self):
            # Determine the bounds of the keyboard
            max_row_length = max(len(row) for row in self.letter_grid)
            button_spacing = 0.013

            # Calculate keyboard dimensions
            kb_width = max_row_length * (self.button_width + button_spacing) + 0.052
            kb_height = (
                len(self.letter_grid) * (self.button_height + button_spacing) + 0.052
            )

            # Create border around keyboard
            self.keyboard_border = visual.Rect(
                win=self.win,
                pos=self.keyboard_position,
                width=kb_width,
                height=kb_height,
                lineColor="black",
                fillColor=None,
            )
            self._register(self.keyboard_border)

            # Create border around input display if showing screen
            if self.show_screen:
                self.output_border = visual.Rect(
                    win=self.win,
                    pos=(self.inputDisplay.pos[0], self.inputDisplay.pos[1]),
                    width=0.455,
                    height=0.13,
                    lineColor="black",
                    fillColor=None,
                )
                self._register(self.output_border)

        def draw(self):
            # Draw keyboard components
            self.keyboard_border.draw()

            # Draw buttons with hover/click effects applied
            for el in self.button_elements:
                el["rect"].draw()
                el["text"].draw()

            if self.show_delete and self.delete_button:
                self.delete_button["rect"].draw()
                self.delete_button["text"].draw()

            if self.show_screen:
                self.title_stim.draw()
                self.inputDisplay.text = self.input_string
                self.inputDisplay.draw()
                self.output_border.draw()

            # Reset click highlight after duration
            if (
                self.enable_click_feedback
                and self.clicked_button
                and self.click_timer.getTime() >= self.feedback_duration
            ):
                self.clicked_button["rect"].fillColor = self.normal_color
                self.clicked_button = None

        def check_mouse(self, mouse):
            mx, my = mouse.getPos()
            mouse_pressed = mouse.getPressed()[0]

            # Reset hover effect (if enabled)
            if self.enable_hover_feedback and self.hovered_button:
                self.hovered_button["rect"].fillColor = self.normal_color
                self.hovered_button = None

            # Handle hover effects
            if self.enable_hover_feedback:
                # Check for hover on letter buttons
                for el in self.button_elements:
                    rx, ry = el["rect"].pos
                    half_w = el["rect"].width / 2.0
                    half_h = el["rect"].height / 2.0
                    if (
                        rx - half_w <= mx <= rx + half_w
                        and ry - half_h <= my <= ry + half_h
                    ):
                        # Apply hover effect if not currently clicked
                        if not self.clicked_button or self.clicked_button != el:
                            el["rect"].fillColor = self.hover_color
                            self.hovered_button = el
                        break

                # Check for hover on delete button
                if self.show_delete and self.delete_button:
                    rx, ry = self.delete_button["rect"].pos
                    half_w = self.delete_button["rect"].width / 2.0
                    half_h = self.delete_button["rect"].height / 2.0
                    if (
                        rx - half_w <= mx <= rx + half_w
                        and ry - half_h <= my <= ry + half_h
                    ):
                        # Apply hover effect if not currently clicked
                        if (
                            not self.clicked_button
                            or self.clicked_button != self.delete_button
                        ):
                            self.delete_button["rect"].fillColor = self.hover_color
                            self.hovered_button = self.delete_button

            # Handle mouse clicks
            if mouse_pressed:
                if not self.mouse_clicked:
                    # Check letter buttons
                    for el in self.button_elements:
                        rx, ry = el["rect"].pos
                        half_w = el["rect"].width / 2.0
                        half_h = el["rect"].height / 2.0
                        if (
                            rx - half_w <= mx <= rx + half_w
                            and ry - half_h <= my <= ry + half_h
                        ):
                            # Apply click effect if feedback enabled
                            if self.enable_click_feedback:
                                el["rect"].fillColor = self.click_color
                                self.clicked_button = el
                                self.click_timer.reset()

                            # Generate letter event
                            letter = el["label"]
                            self.events.append(("letter", letter))
                            if self.show_screen:
                                self.input_string += letter
                            break

                    # Check delete button
                    if self.show_delete and self.delete_button:
                        rx, ry = self.delete_button["rect"].pos
                        half_w = self.delete_button["rect"].width / 2.0
                        half_h = self.delete_button["rect"].height / 2.0
                        if (
                            rx - half_w <= mx <= rx + half_w
                            and ry - half_h <= my <= ry + half_h
                        ):
                            # Apply click effect if feedback enabled
                            if self.enable_click_feedback:
                                self.delete_button["rect"].fillColor = self.click_color
                                self.clicked_button = self.delete_button
                                self.click_timer.reset()

                            # Generate delete event
                            self.events.append(("delete", None))
                            if self.show_screen and self.input_string:
                                self.input_string = self.input_string[:-1]

                    self.mouse_clicked = True
            else:
                self.mouse_clicked = False

        def get_events(self):
            ev, self.events = self.events[:], []
            return ev

        def set_delete_visibility(self, show):
            """Toggle visibility of the delete button."""
            self.show_delete = show
            if self.delete_button:
                self.delete_button["rect"].opacity = 1.0 if show else 0.0
                self.delete_button["text"].opacity = 1.0 if show else 0.0

        def set_feedback_enabled(self, click_enabled=None, hover_enabled=None):
            """Enable or disable feedback effects separately."""
            if click_enabled is not None:
                self.enable_click_feedback = click_enabled
            if hover_enabled is not None:
                self.enable_hover_feedback = hover_enabled

            # Reset any active effects if disabled
            if not self.enable_hover_feedback and self.hovered_button:
                self.hovered_button["rect"].fillColor = self.normal_color
                self.hovered_button = None

            if not self.enable_click_feedback and self.clicked_button:
                self.clicked_button["rect"].fillColor = self.normal_color
                self.clicked_button = None

    # ------------------------------------------------------------------
    # TextEntryScreen
    # ------------------------------------------------------------------
    class TextEntryScreen(ExperimentComponent):
        """Text entry screen with word display and cursor."""

        def __init__(
            self,
            win,
            max_chars=9,
            pos=(0, 0.13),
            letter_height=0.065,
            letter_color="black",
            cursor_color="darkred",
            font="Arial",
            mode="easy",
            show_reminder=True,
        ):
            super().__init__()
            self.win = win
            self.pos = pos
            self.letter_height = letter_height
            self.letter_color = letter_color
            self.cursor_color = cursor_color
            self.font = font
            self.mode = mode
            self.show_reminder = show_reminder

            # Text entry state
            self.target_word = ""
            self.entered_text = ""
            self.current_position = 0
            self.first_key_pressed = False

            # Visual elements
            self.letter_spacing = 0.0715
            self.max_chars = max_chars

            # Frame width based on max characters
            self.frame_width = self.max_chars * self.letter_spacing + 0.065

            # Calculate left edge position
            self.left_x = self.pos[0] - self.frame_width / 2.0 + self.letter_spacing
            self.text_y = self.pos[1]

            # Initialize text elements for target word display
            self.target_word_stim = visual.TextStim(
                win=self.win,
                text="",
                pos=(self.pos[0], self.pos[1] + 0.195),
                height=self.letter_height * 1.2,
                color=self.letter_color,
                font=self.font,
            )

            # Create placeholders for letter positions
            self.letter_stims = []
            for i in range(self.max_chars):
                x = self.left_x + i * self.letter_spacing
                stim = visual.TextStim(
                    win=self.win,
                    text="",
                    pos=(x, self.text_y),
                    height=self.letter_height,
                    color=self.letter_color,
                    font=self.font,
                )
                self.letter_stims.append(stim)

            # Create the entry frame
            self.entry_frame = visual.Rect(
                win=self.win,
                pos=self.pos,
                width=self.frame_width,
                height=0.13,
                lineColor="black",
                fillColor=None,
            )

            # Create reminder text if enabled
            if self.show_reminder:
                self.reminder_text = visual.TextStim(
                    win=self.win,
                    text="",
                    pos=(0, -0.45),
                    height=self.letter_height / 3,
                    color="black",
                    font=self.font,
                )

            # Register all visual elements
            self._register(self.target_word_stim, self.entry_frame)
            self._register(self.letter_stims)
            if self.show_reminder:
                self._register(self.reminder_text)

            self.hide()

        def set_target_word(self, word):
            """Set the target word and initialize display."""
            self.target_word = word
            self.target_word_stim.text = f"Word: {word}"
            self.entered_text = ""
            self.current_position = 0
            self.first_key_pressed = False

            # Update reminder text with word length
            if self.show_reminder:
                word_length = len(word)
                self.reminder_text.text = TYPING_REMINDER_MESSAGE

            # Clear all letter displays
            for stim in self.letter_stims:
                stim.text = ""

        def update_with_letter(self, letter):
            """Add a letter at the current position."""
            if self.current_position < self.max_chars:
                # Check if this is the first keystroke in hard mode
                if self.mode == "hard" and not self.first_key_pressed:
                    self.first_key_pressed = True
                    self.target_word_stim.text = ""

                # Update internal text representation
                if len(self.entered_text) <= self.current_position:
                    self.entered_text += letter
                else:
                    self.entered_text = (
                        self.entered_text[: self.current_position]
                        + letter
                        + self.entered_text[self.current_position + 1 :]
                    )

                # Update visual display (only in easy mode)
                if self.mode == "easy" or not self.first_key_pressed:
                    self.letter_stims[self.current_position].text = letter

                # Move to next position
                self.current_position += 1

                # Optional typing delay
                if TYPING_DELAY_ENABLED and TYPING_DELAY > 0:
                    core.wait(TYPING_DELAY)

        def delete_letter(self):
            """Delete the letter at the previous position."""
            if self.current_position > 0:
                self.current_position -= 1

                # Update internal text
                if self.current_position < len(self.entered_text):
                    self.entered_text = (
                        self.entered_text[: self.current_position]
                        + self.entered_text[self.current_position + 1 :]
                    )

                # Update visual display (only in easy mode)
                if self.mode == "easy":
                    self.letter_stims[self.current_position].text = ""

        def draw(self):
            """Draw all components of the text entry screen."""
            # Draw the frame and target word
            self.entry_frame.draw()
            self.target_word_stim.draw()

            # Draw letter stimuli
            for stim in self.letter_stims:
                stim.draw()

            # Draw reminder text if enabled
            if self.show_reminder:
                self.reminder_text.draw()

    # ------------------------------------------------------------------
    # PrimaryTask  (composite of Keyboard + TextEntryScreen)
    # ------------------------------------------------------------------
    class PrimaryTask(ExperimentComponent):
        """
        Composite task: text entry screen + keyboard + optional OK button.
        """

        def __init__(
            self, win, keyboard, mouse, mode, ok_rect, ok_text, show_reminder=True
        ):
            super().__init__()
            self.win = win
            self.keyboard = keyboard
            self.mouse = mouse
            self.ok_rect = ok_rect
            self.ok_text = ok_text

            self.text_entry_screen = TextEntryScreen(
                win=win, mode=mode, show_reminder=show_reminder
            )
            self.target_word = ""
            self.submitted = False

            # Auto-submission system with delay
            self.submission_state = "none"
            self.submission_timer = core.Clock()
            self.feedback_screen = SubmissionFeedbackScreen(win)

            # Register OK button only if SHOW_OK_BUTTON is True (for debugging)
            if SHOW_OK_BUTTON:
                self._register(ok_rect, ok_text)
            self.hide()

        def show(self):
            self.text_entry_screen.show()
            self.keyboard.show()
            if SHOW_OK_BUTTON:
                self.ok_rect.autoDraw = True
                self.ok_text.autoDraw = True

        def hide(self):
            self.text_entry_screen.hide()
            self.keyboard.hide()
            self.feedback_screen.hide()
            self.submission_state = "none"
            if SHOW_OK_BUTTON:
                self.ok_rect.autoDraw = False
                self.ok_text.autoDraw = False

        def reset(self):
            self.keyboard.input_string = ""
            self.keyboard.events.clear()
            self.text_entry_screen.entered_text = ""
            self.text_entry_screen.current_position = 0
            self.submitted = False
            self.submission_state = "none"
            self.feedback_screen.hide()

        def set_target_word(self, word):
            """Set the target word to be entered."""
            self.target_word = word
            self.text_entry_screen.set_target_word(word)

        def draw(self):
            if self.submission_state == "showing_feedback":
                self.feedback_screen.draw()
            else:
                self.text_entry_screen.draw()
                self.keyboard.draw()
                if SHOW_OK_BUTTON:
                    self.ok_rect.fillColor = "lightgrey"
                    self.ok_text.color = "black"
                    self.ok_rect.draw()
                    self.ok_text.draw()

        def check_input(self):
            # Handle submission state machine
            if self.submission_state == "delaying":
                if self.submission_timer.getTime() >= INTERRUPT_DELAY:
                    self.text_entry_screen.hide()
                    self.keyboard.hide()
                    if SHOW_OK_BUTTON:
                        self.ok_rect.autoDraw = False
                        self.ok_text.autoDraw = False
                    self.submission_state = "showing_feedback"
                    self.submission_timer.reset()
                    self.feedback_screen.show()
                return "continue"

            elif self.submission_state == "showing_feedback":
                if self.submission_timer.getTime() >= 1.0:
                    self.submission_state = "none"
                    self.feedback_screen.hide()
                    self.submitted = True
                    return "submitted"
                return "continue"

            # Normal input processing
            self.keyboard.check_mouse(self.mouse)

            # Handle keyboard events
            for etype, val in self.keyboard.get_events():
                if etype == "letter":
                    self.text_entry_screen.update_with_letter(val)
                    # Check for auto-submission after adding letter
                    if len(self.text_entry_screen.entered_text) == len(
                        self.target_word
                    ):
                        self.submission_state = "delaying"
                        self.submission_timer.reset()
                        return "continue"
                elif etype == "delete":
                    self.text_entry_screen.delete_letter()

            # OK button check (only if SHOW_OK_BUTTON is True - for debugging)
            if SHOW_OK_BUTTON and self.mouse.getPressed()[0]:
                mx, my = self.mouse.getPos()
                if (
                    okButtonPos[0] - okButtonWidth / 2
                    <= mx
                    <= okButtonPos[0] + okButtonWidth / 2
                    and okButtonPos[1] - okButtonHeight / 2
                    <= my
                    <= okButtonPos[1] + okButtonHeight / 2
                ):
                    self.submitted = True
                    return "submitted"

            return "continue"

        def record_data(self):
            experimentData.append(
                {
                    "target_word": self.target_word,
                    "entered_text": self.text_entry_screen.entered_text,
                    "is_correct": self.text_entry_screen.entered_text
                    == self.target_word,
                }
            )

        def show_submission(self):
            msg = f"Submitted text: {self.text_entry_screen.entered_text}"
            submission_stim = visual.TextStim(
                win=self.win, text=msg, pos=(0, 0), height=0.08, color="black"
            )
            submission_stim.draw()

    # ------------------------------------------------------------------
    # TrialManager
    # ------------------------------------------------------------------
    class TrialManager:
        """
        Controls the timing of task interruptions and stores n-back parameters.
        """

        def __init__(self, interruption_positions, nback_data):
            self.reset(interruption_positions, nback_data)

        def reset(self, interruption_positions, nback_data):
            self.interruption_positions = list(interruption_positions)
            self.nback_data = nback_data
            self._next_ptr = 0
            self._flag = False
            self._nback_idx = 0

        def update(self, primary_task):
            """
            Call each frame: when the current position in text entry
            hits the next interruption position, just set the flag.
            """
            current_position = primary_task.text_entry_screen.current_position
            if (
                self._next_ptr < len(self.interruption_positions)
                and current_position == self.interruption_positions[self._next_ptr]
            ):
                self._flag = True
                self._next_ptr += 1

        def consume(self) -> bool:
            if self._flag:
                self._flag = False
                return True
            return False

        def get_current_nback_data(self):
            """Get n-back parameters for the current interruption"""
            if self._nback_idx < len(self.nback_data):
                data = self.nback_data[self._nback_idx]
                self._nback_idx += 1
                return data
            return None

        @property
        def should_interrupt(self) -> bool:
            return self._flag

    # ------------------------------------------------------------------
    # NBack Task
    # ------------------------------------------------------------------
    class NbackTask(ExperimentComponent):
        """
        N-back task for interruption periods.
        Presents stimuli (digits) in RSVP fashion and requires space press for n-back matches.
        No built-in instructions - handle separately.
        """

        def __init__(
            self,
            win,
            n_back_level=1,
            num_stims=10,
            stim_list=None,
            stim_durations=None,
            matches=None,
            isi_times=None,
            stim_duration=0.5,
            response_window_ratio=0.5,
            feedback_enabled=True,
            mask_enabled=True,
            text_height=0.15,
            text_color="black",
            feedback_duration=0.5,
            font="Arial",
            hide_buffer=0.15,
            minimum_feedback_duration=0.8,
            mask_is_hashtag=False,
            show_reminder=True,
            show_intro_screen=False,
            intro_duration=1.5,
        ):
            super().__init__()
            self.win = win
            self.n_back_level = n_back_level
            self.num_stims = num_stims
            self.stim_duration = stim_duration
            self.response_window_ratio = response_window_ratio
            self.feedback_enabled = feedback_enabled
            self.mask_enabled = mask_enabled
            self.feedback_duration = feedback_duration
            self.hide_buffer = hide_buffer
            self.minimum_feedback_duration = minimum_feedback_duration
            self.show_reminder = show_reminder

            self.show_intro_screen = show_intro_screen
            self.intro_duration = intro_duration

            # Update flags
            self.needs_screen_update = False

            # Store stimulus durations from CSV
            self.stim_durations = stim_durations or [stim_duration] * num_stims

            # Create stimulus border rectangle
            self.stim_border = visual.Rect(
                win=win,
                pos=(0, 0),
                width=0.6,
                height=0.25,
                lineColor="black",
                fillColor=None,
                lineWidth=2,
            )

            # Create n-back level label
            self.level_label = visual.TextStim(
                win=win,
                text=f"{n_back_level}-back",
                pos=(0, 0.3),
                height=text_height / 2,
                color=text_color,
                font=font,
                bold=True,
            )

            # Create stimuli
            self.stim_text = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0),
                height=text_height,
                color=text_color,
                font=font,
            )

            self.feedback_text = visual.TextStim(
                win=win,
                text="",
                pos=(0, -0.3),
                height=text_height / 2,
                color=text_color,
                font=font,
            )

            if self.show_intro_screen:
                self.intro_text = visual.TextStim(
                    win=win,
                    text=f"{n_back_level}-back task.",
                    pos=(0, 0),
                    height=text_height / 2,
                    color=text_color,
                    font=font,
                    bold=True,
                )

            # Create reminder text if enabled
            if self.show_reminder:
                position_word = "position" if n_back_level == 1 else "positions"
                self.reminder_text = visual.TextStim(
                    win=win,
                    text=f"REMINDER: Press SPACE when there is a match with {n_back_level} {position_word} back",
                    pos=(0, -0.45),
                    height=text_height / 6,
                    color="black",
                    font=font,
                )

            if mask_is_hashtag:
                # Create mask stimulus (# symbol)
                self.mask_text = visual.TextStim(
                    win=win,
                    text="#",
                    pos=(0, 0),
                    height=text_height,
                    color=text_color,
                    font=font,
                )
            else:
                self.mask_rect = visual.Rect(
                    win=win,
                    pos=(0, 0),
                    width=text_height * 0.9,
                    height=text_height * 0.9,
                    fillColor=text_color,
                    lineColor=None,
                )

            # Register stimuli with parent class
            components = [
                self.stim_border,
                self.level_label,
                self.stim_text,
                self.feedback_text,
            ]

            if self.show_intro_screen:
                components.insert(0, self.intro_text)

            # Add mask
            if mask_is_hashtag:
                components.append(self.mask_text)
            else:
                components.append(self.mask_rect)

            # Add reminder if enabled
            if self.show_reminder:
                components.append(self.reminder_text)

            # Register all
            self._register(*components)

            # Store mask type for later use
            self.mask_is_hashtag = mask_is_hashtag

            # State variables - all data must come from CSV
            self.letters = stim_list
            self.isi_times = isi_times
            self.matches = (
                [pos in matches for pos in range(num_stims)] if matches else []
            )
            self.responses = [False] * num_stims
            self.is_match = False
            self.responded = False
            self.response_correct = False
            self.current_index = -1
            self.current_state = "begin"
            self.timer = core.Clock()
            self.state_time = 0

            # Feedback timing
            self.feedback_timer = core.Clock()
            self.feedback_showing = False

            # Calculate response windows from provided ISI times
            self.response_windows = [
                self.stim_durations[i] + (isi * self.response_window_ratio)
                for i, isi in enumerate(self.isi_times)
            ]

        def start(self):
            """Begin the n-back task sequence"""
            if self.show_intro_screen:
                self.current_state = "intro"
                self.timer.reset()
            else:
                # Skip the ready state and start immediately
                self.current_state = "begin"
                self.current_index = 0
                self.timer.reset()
                # Show the constant elements
                self.stim_border.autoDraw = True
                self.level_label.autoDraw = True

        def update(self):
            """Update task state based on timing and input"""
            current_time = self.timer.getTime()
            keys = event.getKeys(keyList=["space"])

            # Handle feedback timing independently of states
            if (
                self.feedback_showing
                and self.feedback_timer.getTime() >= self.minimum_feedback_duration
            ):
                self.feedback_text.autoDraw = False
                self.feedback_showing = False
                self.needs_screen_update = True

            # Handle state transitions
            if self.current_state == "intro":
                if current_time >= self.intro_duration:
                    # Hide intro text
                    self.intro_text.autoDraw = False
                    # Show the n-back elements for the first time
                    self.stim_border.autoDraw = True
                    self.level_label.autoDraw = True
                    if self.show_reminder:
                        self.reminder_text.autoDraw = True
                    # Move to begin state
                    self.current_state = "begin"
                    self.current_index = 0
                    self.needs_screen_update = True

            elif self.current_state == "begin":
                # Start showing first stimulus
                self.stim_text.text = self.letters[self.current_index]
                self.stim_text.autoDraw = True
                # Ensure mask is hidden based on type
                if self.mask_is_hashtag:
                    self.mask_text.autoDraw = False
                else:
                    self.mask_rect.autoDraw = False
                self.current_state = "stimulus"
                self.timer.reset()
                self.responded = False
                self.is_match = self.matches[self.current_index]
                self.needs_screen_update = True

            elif self.current_state == "stimulus":
                # Handle responses during stimulus presentation
                if "space" in keys and not self.responded:
                    self.responded = True
                    self.responses[self.current_index] = True
                    self.response_correct = self.is_match
                    if self.feedback_enabled:
                        self._show_feedback(
                            "Correct!" if self.response_correct else "Incorrect!",
                            "green" if self.response_correct else "darkred",
                        )

                # Get duration for current stimulus from CSV
                current_stim_duration = self.stim_durations[self.current_index]

                # Hide stimulus early to prevent overlap
                if current_time >= (current_stim_duration - self.hide_buffer):
                    self.stim_text.autoDraw = False
                    self.needs_screen_update = True

                # Transition to ISI after full stimulus duration
                if current_time >= current_stim_duration:
                    # Show mask during ISI if enabled (after buffer period)
                    if self.mask_enabled:
                        if self.mask_is_hashtag:
                            self.mask_text.autoDraw = True
                        else:
                            self.mask_rect.autoDraw = True
                        self.needs_screen_update = True
                    self.current_state = "isi"
                    self.timer.reset()

            elif self.current_state == "isi":
                # Handle responses during ISI (up to response window)
                if (
                    "space" in keys
                    and not self.responded
                    and current_time
                    <= self.response_windows[self.current_index]
                    - self.stim_durations[self.current_index]
                ):
                    self.responded = True
                    self.responses[self.current_index] = True
                    self.response_correct = self.is_match
                    if self.feedback_enabled:
                        self._show_feedback(
                            "Correct!" if self.response_correct else "Incorrect!",
                            "green" if self.response_correct else "darkred",
                        )

                # Check if we need to show missed feedback
                if (
                    not self.responded
                    and self.is_match
                    and current_time
                    > self.response_windows[self.current_index]
                    - self.stim_durations[self.current_index]
                    and self.feedback_enabled
                ):
                    self._show_feedback("Missed!", "darkred")

                # Hide mask early to prevent overlap with next stimulus
                if self.mask_enabled and current_time >= (
                    self.isi_times[self.current_index] - self.hide_buffer
                ):
                    if self.mask_is_hashtag:
                        self.mask_text.autoDraw = False
                    else:
                        self.mask_rect.autoDraw = False
                    self.needs_screen_update = True

                # Transition to next letter after ISI time
                if current_time >= self.isi_times[self.current_index]:
                    # Move to next stimulus or complete
                    self.current_index += 1
                    if self.current_index < self.num_stims:
                        self.current_state = "begin"
                    else:
                        # Task is complete, hide all elements
                        self.stim_text.autoDraw = False
                        if self.mask_is_hashtag:
                            self.mask_text.autoDraw = False
                        else:
                            self.mask_rect.autoDraw = False
                        self.stim_border.autoDraw = False
                        self.level_label.autoDraw = False
                        # Hide feedback if still showing
                        if self.feedback_showing:
                            self.feedback_text.autoDraw = False
                            self.feedback_showing = False
                        self.needs_screen_update = True
                        return "end"

            return "continue"

        def _show_feedback(self, text, color):
            """Helper method to show feedback and start timing"""
            self.feedback_text.text = text
            self.feedback_text.color = color
            self.feedback_text.autoDraw = True
            self.feedback_timer.reset()
            self.feedback_showing = True
            self.needs_screen_update = True

        def should_flip(self):
            """Check if screen needs updating and reset flag"""
            if self.needs_screen_update:
                self.needs_screen_update = False
                return True
            return False

        def draw(self):
            """Ensure all elements are drawn"""

            if self.current_state == "intro":
                if hasattr(self, "intro_text") and self.intro_text.autoDraw:
                    self.intro_text.draw()
                return

            # Border and level label are always drawn
            self.stim_border.draw()
            self.level_label.draw()

            # Draw reminder text if enabled
            if self.show_reminder:
                self.reminder_text.draw()

            # Draw state-dependent elements
            if self.current_state == "stimulus":
                self.stim_text.draw()
                if self.feedback_text.autoDraw:
                    self.feedback_text.draw()
            elif self.current_state == "isi":
                if self.mask_enabled:
                    if self.mask_is_hashtag:
                        self.mask_text.draw()
                    else:
                        self.mask_rect.draw()
                if self.feedback_text.autoDraw:
                    self.feedback_text.draw()

        def get_performance(self):
            """Return task performance metrics"""
            hits = 0
            false_alarms = 0
            misses = 0
            correct_rejections = 0

            for i in range(self.num_stims):
                if self.matches[i] and self.responses[i]:
                    hits += 1
                elif not self.matches[i] and self.responses[i]:
                    false_alarms += 1
                elif self.matches[i] and not self.responses[i]:
                    misses += 1
                else:
                    correct_rejections += 1

            return {
                "hits": hits,
                "false_alarms": false_alarms,
                "misses": misses,
                "correct_rejections": correct_rejections,
                "accuracy": (
                    (hits + correct_rejections) / self.num_stims
                    if self.num_stims > 0
                    else 0
                ),
                "n_back_level": self.n_back_level,
                "num_stims": self.num_stims,
            }

        def check_key(self):
            """
            Match the interface of the original InterruptionTask.
            This bridges between the new state machine and the original interface.
            """
            status = self.update()
            return status

        def hide(self):
            """Override hide to ensure all elements are hidden properly"""
            super().hide()
            # Ensure these are specifically turned off
            self.stim_text.autoDraw = False
            if self.mask_is_hashtag:
                self.mask_text.autoDraw = False
            else:
                self.mask_rect.autoDraw = False
            self.feedback_text.autoDraw = False
            self.stim_border.autoDraw = False
            self.level_label.autoDraw = False
            if self.show_reminder:
                self.reminder_text.autoDraw = False
            if self.show_intro_screen:
                self.intro_text.autoDraw = False

        def show(self):
            """Override parent show() to handle intro screen correctly"""
            if self.show_intro_screen:
                # Only show the intro text, not all elements
                self.intro_text.autoDraw = True
                # Keep all other elements hidden
                self.stim_border.autoDraw = False
                self.level_label.autoDraw = False
                self.stim_text.autoDraw = False
                self.feedback_text.autoDraw = False
                if self.mask_is_hashtag:
                    self.mask_text.autoDraw = False
                else:
                    self.mask_rect.autoDraw = False
                if self.show_reminder:
                    self.reminder_text.autoDraw = False
            else:
                # Normal behavior - show all elements
                super().show()

    # ------------------------------------------------------------------
    # Time Estimation Display
    # ------------------------------------------------------------------
    class TimeEstimationDisplay(ExperimentComponent):
        """Time estimation display with minutes and seconds boxes."""

        def __init__(
            self,
            win,
            pos=(0, 0.2),
            digit_height=0.07,
            digit_color="black",
            line_color="black",
            font="Arial",
        ):
            super().__init__()
            self.win = win
            self.pos = pos
            self.digit_height = digit_height
            self.digit_color = digit_color
            self.line_color = line_color
            self.font = font

            # Internal state
            self.digits = "000"
            self.digit_count = 0

            # Error state management
            self.error_flash_duration = 0.3
            self.error_flash_timer = core.Clock()
            self.is_flashing_error = False
            self.default_line_color = line_color
            self.is_over_max = False
            self.using_numpad = False

            # Position calculations
            box_width = digit_height * 1.5
            box_height = digit_height * 1.5
            spacing = digit_height * 0.5

            # Calculate positions
            self.min_box_pos = (pos[0] - spacing - box_width / 2, pos[1])
            self.sec_box_pos = (pos[0] + spacing + box_width, pos[1])
            self.colon_pos = (pos[0], pos[1])

            # Create visual elements
            self._create_title()
            self._create_boxes()
            self._create_labels()
            self._create_digit_displays()
            self._create_colon()
            self._create_error_message()

            self.hide()

        def _create_title(self):
            """Create the 'How long did it last?' title."""
            title_y_offset = self.digit_height * 3

            self.title_text = visual.TextStim(
                win=self.win,
                text="How long did it last?",
                pos=(self.pos[0], self.pos[1] + title_y_offset),
                height=self.digit_height * 0.7,
                color=self.digit_color,
                font=self.font,
                bold=True,
            )

            self._register(self.title_text)

        def _create_boxes(self):
            """Create the boxes for minutes and seconds."""
            box_width = self.digit_height * 1.5
            box_height = self.digit_height * 1.5

            self.min_box = visual.Rect(
                win=self.win,
                pos=self.min_box_pos,
                width=box_width,
                height=box_height,
                lineColor=self.line_color,
                fillColor=None,
            )

            # Seconds box is twice as wide to accommodate 2 digits
            self.sec_box = visual.Rect(
                win=self.win,
                pos=self.sec_box_pos,
                width=box_width * 2,
                height=box_height,
                lineColor=self.line_color,
                fillColor=None,
            )

            self._register(self.min_box, self.sec_box)

        def _create_labels(self):
            """Create the 'min' and 'sec' labels above boxes."""
            label_height = self.digit_height * 0.5
            label_y_offset = self.digit_height * 1.0

            self.min_label = visual.TextStim(
                win=self.win,
                text="min",
                pos=(self.min_box_pos[0], self.min_box_pos[1] + label_y_offset),
                height=label_height,
                color=self.digit_color,
                font=self.font,
            )

            self.sec_label = visual.TextStim(
                win=self.win,
                text="sec",
                pos=(self.sec_box_pos[0], self.sec_box_pos[1] + label_y_offset),
                height=label_height,
                color=self.digit_color,
                font=self.font,
            )

            self._register(self.min_label, self.sec_label)

        def _create_digit_displays(self):
            """Create the digit display elements."""
            # Minutes digit
            self.min_digit = visual.TextStim(
                win=self.win,
                text="0",
                pos=self.min_box_pos,
                height=self.digit_height,
                color=self.digit_color,
                font=self.font,
            )

            # Seconds digits (tens and ones)
            sec_tens_x = self.sec_box_pos[0] - self.digit_height * 0.5
            sec_ones_x = self.sec_box_pos[0] + self.digit_height * 0.5

            self.sec_tens_digit = visual.TextStim(
                win=self.win,
                text="0",
                pos=(sec_tens_x, self.sec_box_pos[1]),
                height=self.digit_height,
                color=self.digit_color,
                font=self.font,
            )

            self.sec_ones_digit = visual.TextStim(
                win=self.win,
                text="0",
                pos=(sec_ones_x, self.sec_box_pos[1]),
                height=self.digit_height,
                color=self.digit_color,
                font=self.font,
            )

            self._register(self.min_digit, self.sec_tens_digit, self.sec_ones_digit)

        def _create_colon(self):
            """Create the colon separator."""
            self.colon = visual.TextStim(
                win=self.win,
                text=":",
                pos=self.colon_pos,
                height=self.digit_height,
                color=self.digit_color,
                font=self.font,
            )
            self._register(self.colon)

        def update_with_digit(self, digit):
            """Update the display with a new digit (shifts existing digits left)."""
            # If display is 0:00 and user enters 0, accept but don't count it
            if self.is_all_zeros() and digit == "0":
                return True

            # Check if we've reached the 3-digit limit
            if self.digit_count >= 3:
                self.flash_error()
                return False

            # Add new digit to the right
            old_digits = self.digits
            self.digits = self.digits[1:] + digit

            # Only increment count if we actually changed from all zeros
            if old_digits == "000" and self.digits != "000":
                self.digit_count = 1
            elif old_digits != "000":
                self.digit_count += 1

            # Update visual display
            self._update_display()

            # Optional typing delay
            if TYPING_DELAY_ENABLED and TYPING_DELAY > 0:
                core.wait(TYPING_DELAY)

            return True

        def delete_digit(self):
            """Delete the rightmost digit (shifts remaining digits right)."""
            if self.digit_count > 0:
                # Shift digits right and add 0 to the left
                self.digits = "0" + self.digits[:-1]
                self.digit_count -= 1
                self._update_display()

        def _update_display(self):
            """Update the visual display based on current digits."""
            # Extract individual digits
            min_digit = self.digits[0]
            sec_tens = self.digits[1]
            sec_ones = self.digits[2]

            # Update text stimuli
            self.min_digit.text = min_digit
            self.sec_tens_digit.text = sec_tens
            self.sec_ones_digit.text = sec_ones

        def flash_error(self):
            """Start the error flash animation."""
            self.is_flashing_error = True
            self.error_flash_timer.reset()
            self.min_box.lineColor = "darkred"
            self.sec_box.lineColor = "darkred"

        def _create_error_message(self):
            """Create the error message for maximum value violation."""
            message_y_offset = -self.digit_height * 1.5

            self.error_message = visual.TextStim(
                win=self.win,
                text=f"Maximum possible value is {MAX_CLOCK_VALUE}",
                pos=(self.pos[0], self.pos[1] + message_y_offset),
                height=self.digit_height * 0.4,
                color="darkred",
                font=self.font,
            )

        def update_error_state(self):
            """Update the error flash animation and maximum value violation."""
            # Handle flash error
            if self.is_flashing_error:
                if self.error_flash_timer.getTime() >= self.error_flash_duration:
                    self.is_flashing_error = False
                    if not self.is_over_max:
                        self.min_box.lineColor = self.default_line_color
                        self.sec_box.lineColor = self.default_line_color

            # Check for maximum value violation
            if self.exceeds_maximum():
                if not self.is_over_max:
                    self.is_over_max = True
                    self.min_box.lineColor = "darkred"
                    self.sec_box.lineColor = "darkred"
            else:
                if self.is_over_max:
                    self.is_over_max = False
                    self.min_box.lineColor = self.default_line_color
                    self.sec_box.lineColor = self.default_line_color

        def draw(self):
            """Draw all elements."""
            self.update_error_state()
            self.title_text.draw()
            self.min_box.draw()
            self.sec_box.draw()
            self.min_label.draw()
            self.sec_label.draw()
            self.min_digit.draw()
            self.sec_tens_digit.draw()
            self.sec_ones_digit.draw()
            self.colon.draw()

            # Draw error message only if over maximum AND using numpad
            if self.is_over_max and self.using_numpad:
                self.error_message.draw()

        def exceeds_maximum(self):
            """Check if the current time exceeds the maximum allowed value."""
            max_minutes, max_seconds = map(int, MAX_CLOCK_VALUE.split(":"))
            minutes, seconds = self.get_time()

            if minutes > max_minutes:
                return True
            elif minutes == max_minutes and seconds > max_seconds:
                return True
            return False

        def is_all_zeros(self):
            """Check if all digits are zeros."""
            return self.digits == "000"

        def is_valid_time(self):
            """Check if the current time is valid."""
            seconds = int(self.digits[1:])
            return seconds < 60

        def get_time(self):
            """Return the current time as minutes and seconds."""
            minutes = int(self.digits[0])
            seconds = int(self.digits[1:])
            return minutes, seconds

        def get_time_string(self):
            """Return the current time as a formatted string."""
            minutes = int(self.digits[0])
            seconds = int(self.digits[1:])
            return f"{minutes}:{seconds:02d}"

    # ------------------------------------------------------------------
    # Submission Feedback Screen
    # ------------------------------------------------------------------
    class SubmissionFeedbackScreen(ExperimentComponent):
        """Feedback screen showing 'Word submitted.' for 1 second."""

        def __init__(self, win):
            super().__init__()
            self.win = win

            # Title
            self.title_stim = visual.TextStim(
                win=win,
                text="Word submitted.",
                pos=(0, 0),
                height=0.06,
                color="black",
                font="Arial",
                bold=True,
            )

            self._register(self.title_stim)
            self.hide()

        def draw(self):
            """Draw the feedback message."""
            self.title_stim.draw()

    # ------------------------------------------------------------------
    # Time Estimation Slider
    # ------------------------------------------------------------------
    class TimeSlider(ExperimentComponent):
        """Horizontal slider for time input from 0:00 to 9:59."""

        def __init__(
            self,
            win,
            pos=(0, -0.2),
            width=0.8,
            height=0.02,
            handle_radius=0.02,
            track_color="lightgrey",
            handle_color="darkgrey",
            active_handle_color="blue",
            endpoint_color="black",
            label_height=0.03,
            font="Arial",
        ):
            super().__init__()
            self.win = win
            self.pos = pos
            self.width = width
            self.height = height
            self.handle_radius = handle_radius

            # Colors
            self.track_color = track_color
            self.handle_color = handle_color
            self.active_handle_color = active_handle_color
            self.endpoint_color = endpoint_color

            # Text properties
            self.label_height = label_height
            self.font = font

            # Calculate maximum value from global
            max_minutes, max_seconds = map(int, MAX_CLOCK_VALUE.split(":"))
            self.max_value = max_minutes * 60 + max_seconds

            # State
            self.value = 0
            self.is_dragging = False
            self.handle_position = (pos[0] - width / 2, pos[1])

            # Visual elements
            self._create_track()
            self._create_handle()
            self._create_endpoints()
            self._create_labels()

            self.hide()

        def _create_track(self):
            """Create the slider track."""
            self.track = visual.Rect(
                win=self.win,
                pos=self.pos,
                width=self.width,
                height=self.height,
                fillColor=self.track_color,
                lineColor=None,
            )
            self._register(self.track)

        def _create_handle(self):
            """Create the draggable handle."""
            self.handle = visual.Circle(
                win=self.win,
                pos=self.handle_position,
                radius=self.handle_radius,
                fillColor=self.handle_color,
                lineColor="black",
            )
            self._register(self.handle)

        def _create_endpoints(self):
            """Create visual endpoints."""
            left_pos = (self.pos[0] - self.width / 2, self.pos[1])
            right_pos = (self.pos[0] + self.width / 2, self.pos[1])

            self.left_endpoint = visual.Line(
                win=self.win,
                start=(left_pos[0], left_pos[1] - self.height),
                end=(left_pos[0], left_pos[1] + self.height),
                lineColor=self.endpoint_color,
                lineWidth=2,
            )

            self.right_endpoint = visual.Line(
                win=self.win,
                start=(right_pos[0], right_pos[1] - self.height),
                end=(right_pos[0], right_pos[1] + self.height),
                lineColor=self.endpoint_color,
                lineWidth=2,
            )

            self._register(self.left_endpoint, self.right_endpoint)

        def _create_labels(self):
            """Create endpoint labels."""
            label_offset = 0.05

            self.left_label = visual.TextStim(
                win=self.win,
                text="0:00",
                pos=(self.pos[0] - self.width / 2, self.pos[1] - label_offset),
                height=self.label_height,
                color="black",
                font=self.font,
            )

            self.right_label = visual.TextStim(
                win=self.win,
                text=MAX_CLOCK_VALUE,
                pos=(self.pos[0] + self.width / 2, self.pos[1] - label_offset),
                height=self.label_height,
                color="black",
                font=self.font,
            )

            self._register(self.left_label, self.right_label)

        def check_mouse(self, mouse):
            """Check mouse interaction with slider."""
            mx, my = mouse.getPos()
            mouse_pressed = mouse.getPressed()[0]

            # Check if mouse is near handle
            handle_x, handle_y = self.handle.pos
            distance = ((mx - handle_x) ** 2 + (my - handle_y) ** 2) ** 0.5
            is_near_handle = distance < self.handle_radius * 2

            # Start dragging
            if mouse_pressed and is_near_handle and not self.is_dragging:
                self.is_dragging = True
                self.handle.fillColor = self.active_handle_color

            # Continue dragging
            if self.is_dragging:
                if mouse_pressed:
                    # Constrain x position to track bounds
                    track_left = self.pos[0] - self.width / 2
                    track_right = self.pos[0] + self.width / 2
                    new_x = max(track_left, min(track_right, mx))

                    self.handle.pos = (new_x, self.pos[1])

                    # Calculate value based on position
                    fraction = (new_x - track_left) / self.width
                    self.value = int(fraction * self.max_value)
                else:
                    # Stop dragging
                    self.is_dragging = False
                    self.handle.fillColor = self.handle_color

        def get_time(self):
            """Get current time as minutes and seconds."""
            minutes = self.value // 60
            seconds = self.value % 60
            return minutes, seconds

        def get_time_string(self):
            """Get current time as formatted string."""
            minutes, seconds = self.get_time()
            return f"{minutes}:{seconds:02d}"

        def get_digits(self):
            """Get current time as three digits for display compatibility."""
            minutes, seconds = self.get_time()
            return f"{minutes}{seconds:02d}"

        def reset(self):
            """Reset slider to starting position."""
            self.value = 0
            self.handle.pos = (self.pos[0] - self.width / 2, self.pos[1])
            self.is_dragging = False
            self.handle.fillColor = self.handle_color

        def draw(self):
            """Draw all slider elements."""
            self.track.draw()
            self.left_endpoint.draw()
            self.right_endpoint.draw()
            self.handle.draw()
            self.left_label.draw()
            self.right_label.draw()

    # ------------------------------------------------------------------
    # Time Estimation Task
    # ------------------------------------------------------------------
    class TimeEstimationTask(ExperimentComponent):
        """Time estimation task with input from numpad or slider."""

        def __init__(self, win, input_device, mouse, ok_rect, ok_text):
            super().__init__()
            self.win = win
            self.input_device = input_device
            self.mouse = mouse
            self.ok_rect = ok_rect
            self.ok_text = ok_text

            # Create the time display
            self.time_display = TimeEstimationDisplay(win=win, pos=(0, 0.1))

            # Set whether using numpad based on input device type
            self.time_display.using_numpad = isinstance(input_device, NumPad)

            self.submitted = False
            self.default_ok_color = ok_rect.fillColor
            self.default_ok_text_color = ok_text.color

            self._register(ok_rect, ok_text)
            self.hide()

        def show(self):
            """Show all components."""
            self.time_display.show()
            self.input_device.show()
            self.ok_rect.autoDraw = True
            self.ok_text.autoDraw = True

        def hide(self):
            """Hide all components."""
            self.time_display.hide()
            self.input_device.hide()
            self.ok_rect.autoDraw = False
            self.ok_text.autoDraw = False

        def reset(self):
            """Reset the task to initial state."""
            self.time_display.digits = "000"
            self.time_display.digit_count = 0
            self.time_display._update_display()
            if hasattr(self.input_device, "reset"):
                self.input_device.reset()
            self.submitted = False

        def draw(self):
            """Draw all components."""
            self.time_display.draw()
            self.input_device.draw()

            # Update OK button appearance based on time validity
            if self.time_display.is_all_zeros() or self.time_display.exceeds_maximum():
                # Disable OK button if all zeros or exceeds maximum
                self.ok_rect.fillColor = "darkgrey"
                self.ok_text.color = "grey"
            else:
                # Enable OK button otherwise
                self.ok_rect.fillColor = self.default_ok_color
                self.ok_text.color = self.default_ok_text_color

            self.ok_rect.draw()
            self.ok_text.draw()

        def check_input(self):
            """Check for input from the device and update display."""
            # Handle numpad input
            if isinstance(self.input_device, NumPad):
                self.input_device.check_mouse(self.mouse)
                for etype, val in self.input_device.get_events():
                    if etype == "digit":
                        self.time_display.update_with_digit(val)
                    elif etype == "delete":
                        self.time_display.delete_digit()

            # Handle slider input
            elif isinstance(self.input_device, TimeSlider):
                self.input_device.check_mouse(self.mouse)

                # Update display directly from slider value
                digits = self.input_device.get_digits()
                self.time_display.digits = digits
                self.time_display._update_display()

                # Update digit count for proper validation
                if digits == "000":
                    self.time_display.digit_count = 0
                else:
                    self.time_display.digit_count = len(digits.lstrip("0"))

            # Check for OK button press
            if self.mouse.getPressed()[0]:
                mx, my = self.mouse.getPos()
                if (
                    self.ok_rect.pos[0] - self.ok_rect.width / 2
                    <= mx
                    <= self.ok_rect.pos[0] + self.ok_rect.width / 2
                    and self.ok_rect.pos[1] - self.ok_rect.height / 2
                    <= my
                    <= self.ok_rect.pos[1] + self.ok_rect.height / 2
                ):

                    # Check if OK button should respond
                    if (
                        not self.time_display.is_all_zeros()
                        and not self.time_display.exceeds_maximum()
                    ):
                        if self.time_display.is_valid_time():
                            self.submitted = True
                            return "submitted"
                        else:
                            # Flash error for invalid time (seconds >= 60)
                            self.time_display.flash_error()

            return "continue"

        def get_response(self):
            """Get the current time estimation response."""
            return self.time_display.get_time_string()

    # ------------------------------------------------------------------
    # NumPad
    # ------------------------------------------------------------------
    class NumPad(ExperimentComponent):
        """
        Numeric keypad that generates ("digit", value) or ("delete", None) events.
        """

        NUMPAD_POSITION = (0.5, -0.1)

        def __init__(
            self,
            win,
            numpad_position=NUMPAD_POSITION,
            button_width=0.07,
            button_height=0.07,
            button_text_height=0.04,
            input_display_height=0.05,
            input_display_pos=(0, 0.20),
            x_positions=None,
            y_positions=None,
            extra_keys=None,
            button_fill_color="lightgrey",
            button_line_color="black",
            button_text_color="black",
            input_text_color="black",
            button_font="Arial",
            show_screen=False,
        ):
            super().__init__()
            self.win = win
            self.numpad_position = numpad_position
            self.button_width = button_width
            self.button_height = button_height
            self.button_text_height = button_text_height
            self.input_display_height = input_display_height
            self.input_display_pos = input_display_pos
            self.show_screen = show_screen

            self.x_positions = x_positions or [-0.09, 0.0, 0.09]
            self.y_positions = y_positions or [0.09, 0.0, -0.09]

            self.number_grid = [["7", "8", "9"], ["4", "5", "6"], ["1", "2", "3"]]
            self.extra_keys = extra_keys or [
                {"label": "0", "pos": (-0.09, -0.18)},
                {"label": "←DEL", "pos": (0.09, -0.18)},
            ]

            self.button_fill_color = button_fill_color
            self.button_line_color = button_line_color
            self.button_text_color = button_text_color
            self.input_text_color = input_text_color
            self.button_font = button_font

            self.numpad_buttons = []
            self.button_elements = []
            self.input_string = ""
            self.events = []
            self.mouse_clicked = False

            self._create_buttons()
            if self.show_screen:
                self._create_title()
                self._create_input_display()
            self._create_borders()
            self.hide()

        def _create_buttons(self):
            for i, row in enumerate(self.number_grid):
                y = self.y_positions[i]
                for j, label in enumerate(row):
                    x = self.x_positions[j]
                    self.numpad_buttons.append({"label": label, "pos": (x, y)})
            self.numpad_buttons.extend(self.extra_keys)

            for btn in self.numpad_buttons:
                final_x = btn["pos"][0] + self.numpad_position[0]
                final_y = btn["pos"][1] + self.numpad_position[1]
                width = self.button_width
                if btn["label"] == "←DEL":
                    width = 0.14
                    right_edge = (
                        0.09 + self.numpad_position[0] + (self.button_width / 2.0)
                    )
                    final_x = right_edge - width / 2.0

                rect = visual.Rect(
                    win=self.win,
                    pos=(final_x, final_y),
                    width=width,
                    height=self.button_height,
                    fillColor=self.button_fill_color,
                    lineColor=self.button_line_color,
                )
                txt = visual.TextStim(
                    win=self.win,
                    pos=(final_x, final_y),
                    text=btn["label"],
                    height=self.button_text_height,
                    color=self.button_text_color,
                    font=self.button_font,
                )
                self.button_elements.append(
                    {"label": btn["label"], "rect": rect, "text": txt}
                )
                self._register(rect, txt)

        def _create_title(self):
            self.title_stim = visual.TextStim(
                win=self.win,
                text="Numpad",
                pos=(self.numpad_position[0], 0.28 + self.numpad_position[1]),
                height=0.035,
                color=self.input_text_color,
                bold=True,
            )
            self._register(self.title_stim)

        def _create_input_display(self):
            self.inputDisplay = visual.TextStim(
                win=self.win,
                text=" ",
                pos=(
                    self.input_display_pos[0] + self.numpad_position[0],
                    self.input_display_pos[1] + self.numpad_position[1],
                ),
                height=self.input_display_height,
                color=self.input_text_color,
            )
            self._register(self.inputDisplay)

        def _create_borders(self):
            offset_x, offset_y = self.numpad_position
            left_margin, right_margin = -0.15, 0.15
            bottom_margin = -0.24
            top_margin = 0.33 if self.show_screen else 0.18
            border_center = (
                (left_margin + right_margin) / 2 + offset_x,
                (top_margin + bottom_margin) / 2 + offset_y,
            )

            self.numpad_border = visual.Rect(
                win=self.win,
                pos=border_center,
                width=(right_margin - left_margin),
                height=(top_margin - bottom_margin),
                lineColor="black",
                fillColor=None,
            )
            self._register(self.numpad_border)

            if self.show_screen:
                self.output_border = visual.Rect(
                    win=self.win,
                    pos=(self.inputDisplay.pos[0], self.inputDisplay.pos[1]),
                    width=0.22,
                    height=0.10,
                    lineColor="black",
                    fillColor=None,
                )
                self._register(self.output_border)

        def draw(self):
            if self.show_screen:
                self.title_stim.draw()
                self.inputDisplay.text = self.input_string
                self.inputDisplay.draw()
                self.output_border.draw()
            self.numpad_border.draw()
            for el in self.button_elements:
                el["rect"].draw()
                el["text"].draw()

        def check_mouse(self, mouse):
            if mouse.getPressed()[0]:
                if not self.mouse_clicked:
                    mx, my = mouse.getPos()
                    for el in self.button_elements:
                        rx, ry = el["rect"].pos
                        half_w = el["rect"].width / 2.0
                        half_h = el["rect"].height / 2.0
                        if (
                            rx - half_w <= mx <= rx + half_w
                            and ry - half_h <= my <= ry + half_h
                        ):
                            lab = el["label"]
                            if lab == "←DEL":
                                self.events.append(("delete", None))
                                if self.show_screen and self.input_string:
                                    self.input_string = self.input_string[:-1]
                            else:
                                self.events.append(("digit", lab))
                                if self.show_screen:
                                    self.input_string += lab
                            break
                    self.mouse_clicked = True
            else:
                self.mouse_clicked = False

        def get_events(self):
            ev, self.events = self.events[:], []
            return ev

    # ------------------------------------------------------------------
    # Demographics - Gender Screen
    # ------------------------------------------------------------------
    class RadioButtonScreen(ExperimentComponent):
        """Screen with radio button selection for gender."""

        def __init__(
            self, win, title="", question="", options=[], button_text="Continue"
        ):
            super().__init__()
            self.win = win
            self.title = title
            self.question = question
            self.options = options
            self.selected_index = None

            # Button activation delay
            self.screen_start_time = None
            self.button_delay = 2.0
            self.button_active = False

            # Visual elements
            self.button_radius = 0.015
            self.option_spacing = 0.08
            self.option_height = 0.04

            # Title
            self.title_stim = visual.TextStim(
                win=win,
                text=title,
                pos=(0, 0.35),
                height=0.06,
                color="black",
                font="Arial",
                bold=True,
            )

            # Question
            self.question_stim = visual.TextStim(
                win=win,
                text=question,
                pos=(0, 0.2),
                height=0.045,
                color="black",
                font="Arial",
                wrapWidth=1.4,
            )

            # Create radio buttons and labels
            self.radio_buttons = []
            self.option_labels = []
            start_y = 0.05

            for i, option in enumerate(options):
                y_pos = start_y - (i * self.option_spacing)

                # Radio button circle (outer)
                outer_circle = visual.Circle(
                    win=win,
                    pos=(-0.3, y_pos),
                    radius=self.button_radius,
                    lineColor="black",
                    fillColor="white",
                    lineWidth=2,
                )

                # Radio button circle (inner - for selection)
                inner_circle = visual.Circle(
                    win=win,
                    pos=(-0.3, y_pos),
                    radius=self.button_radius * 0.6,
                    lineColor=None,
                    fillColor="black",
                    opacity=0,
                )

                # Option label
                label = visual.TextStim(
                    win=win,
                    text=option,
                    pos=(-0.25, y_pos),
                    height=self.option_height,
                    color="black",
                    font="Arial",
                    anchorHoriz="left",
                )

                self.radio_buttons.append(
                    {
                        "outer": outer_circle,
                        "inner": inner_circle,
                        "pos": (-0.3, y_pos),
                        "index": i,
                    }
                )
                self.option_labels.append(label)

            # Navigation button
            self.nav_button = visual.Rect(
                win=win,
                pos=(0, -0.4),
                width=0.2,
                height=0.08,
                fillColor="lightgrey",
                lineColor="black",
            )

            self.nav_text = visual.TextStim(
                win=win,
                text=button_text,
                pos=(0, -0.4),
                height=0.035,
                color="black",
                font="Arial",
            )

            # Register all components
            elements_to_register = [
                self.title_stim,
                self.question_stim,
                self.nav_button,
                self.nav_text,
            ]
            for button in self.radio_buttons:
                elements_to_register.extend([button["outer"], button["inner"]])
            elements_to_register.extend(self.option_labels)

            self._register(*elements_to_register)
            self.hide()

        def show(self):
            """Override show to start the timer."""
            import time

            super().show()
            self.screen_start_time = time.time()
            self.button_active = False

        def check_selection(self, mouse):
            """Check for radio button clicks and navigation."""
            import time

            # Check if enough time has passed for navigation button
            if self.screen_start_time is not None:
                current_time = time.time()
                time_elapsed = current_time - self.screen_start_time
                if time_elapsed >= self.button_delay:
                    self.button_active = True

            if mouse.getPressed()[0]:
                mx, my = mouse.getPos()

                # Check radio button clicks
                for button in self.radio_buttons:
                    button_x, button_y = button["pos"]
                    distance = ((mx - button_x) ** 2 + (my - button_y) ** 2) ** 0.5

                    if distance <= self.button_radius * 1.5:
                        # Update selection
                        self.selected_index = button["index"]
                        self._update_visual_selection()
                        return "selection_changed"

                # Check navigation button (only if selection made and button active)
                if (
                    self.selected_index is not None
                    and self.button_active
                    and self.nav_button.pos[0] - self.nav_button.width / 2
                    <= mx
                    <= self.nav_button.pos[0] + self.nav_button.width / 2
                    and self.nav_button.pos[1] - self.nav_button.height / 2
                    <= my
                    <= self.nav_button.pos[1] + self.nav_button.height / 2
                ):
                    return "continue"

            return "waiting"

        def _update_visual_selection(self):
            """Update visual state of radio buttons."""
            for i, button in enumerate(self.radio_buttons):
                if i == self.selected_index:
                    button["inner"].opacity = 1.0
                else:
                    button["inner"].opacity = 0.0

        def get_selection(self):
            """Get the selected option text."""
            if self.selected_index is not None:
                return self.options[self.selected_index]
            return None

        def draw(self):
            """Draw all elements."""
            self.title_stim.draw()
            self.question_stim.draw()

            # Draw radio buttons and labels
            for button in self.radio_buttons:
                button["outer"].draw()
                button["inner"].draw()

            for label in self.option_labels:
                label.draw()

            # Draw navigation button with appropriate state
            if self.selected_index is not None and self.button_active:
                self.nav_button.fillColor = "lightgrey"
                self.nav_text.color = "black"
            else:
                self.nav_button.fillColor = "darkgrey"
                self.nav_text.color = "grey"

            self.nav_button.draw()
            self.nav_text.draw()

    # ------------------------------------------------------------------
    # Demographics - Age Screen
    # ------------------------------------------------------------------
    class AgeInputScreen(ExperimentComponent):
        """Screen for age input using NumPad."""

        def __init__(self, win, mouse):
            super().__init__()
            self.win = win
            self.mouse = mouse
            self.age_value = ""

            # Button activation delay
            self.screen_start_time = None
            self.button_delay = 2.0
            self.button_active = False

            # Title
            self.title_stim = visual.TextStim(
                win=win,
                text="Demographics: Age",
                pos=(0, 0.35),
                height=0.06,
                color="black",
                font="Arial",
                bold=True,
            )

            # Question
            self.question_stim = visual.TextStim(
                win=win,
                text="Please enter your age:",
                pos=(0, 0.25),
                height=0.045,
                color="black",
                font="Arial",
            )

            # Age display box
            self.age_display_box = visual.Rect(
                win=win,
                pos=(0, 0.15),
                width=0.2,
                height=0.08,
                lineColor="black",
                fillColor="white",
            )

            # Age display text
            self.age_display_text = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0.15),
                height=0.05,
                color="black",
                font="Arial",
            )

            # Create NumPad
            self.numpad = NumPad(win=win, numpad_position=(0, -0.10), show_screen=False)

            # Navigation button
            self.nav_button = visual.Rect(
                win=win,
                pos=(0, -0.4),
                width=0.2,
                height=0.08,
                fillColor="lightgrey",
                lineColor="black",
            )

            self.nav_text = visual.TextStim(
                win=win,
                text="Continue",
                pos=(0, -0.4),
                height=0.035,
                color="black",
                font="Arial",
            )

            self._register(
                self.title_stim,
                self.question_stim,
                self.age_display_box,
                self.age_display_text,
                self.nav_button,
                self.nav_text,
            )
            self.hide()

        def show(self):
            """Override show to start the timer and show numpad."""
            import time

            super().show()
            self.numpad.show()
            self.screen_start_time = time.time()
            self.button_active = False

        def hide(self):
            """Override hide to hide numpad too."""
            super().hide()
            self.numpad.hide()

        def check_input(self):
            """Check for numpad input and navigation."""
            import time

            # Check if enough time has passed for navigation button
            if self.screen_start_time is not None:
                current_time = time.time()
                time_elapsed = current_time - self.screen_start_time
                if time_elapsed >= self.button_delay:
                    self.button_active = True

            # Handle numpad input
            self.numpad.check_mouse(self.mouse)
            for event_type, value in self.numpad.get_events():
                if event_type == "digit":
                    if len(self.age_value) < 2:
                        self.age_value += value
                        self._update_display()
                elif event_type == "delete":
                    if self.age_value:
                        self.age_value = self.age_value[:-1]
                        self._update_display()

            # Check navigation button
            if self.mouse.getPressed()[0]:
                mx, my = self.mouse.getPos()
                if (
                    self.nav_button.pos[0] - self.nav_button.width / 2
                    <= mx
                    <= self.nav_button.pos[0] + self.nav_button.width / 2
                    and self.nav_button.pos[1] - self.nav_button.height / 2
                    <= my
                    <= self.nav_button.pos[1] + self.nav_button.height / 2
                ):

                    if self._is_valid_age() and self.button_active:
                        return "continue"

            return "waiting"

        def _update_display(self):
            """Update the age display and border color."""
            self.age_display_text.text = self.age_value if self.age_value else ""

            # Update border color based on age validity
            if len(self.age_value) == 2:
                age = int(self.age_value)
                if age < 18:
                    self.age_display_box.lineColor = "darkred"
                else:
                    self.age_display_box.lineColor = "black"
            else:
                self.age_display_box.lineColor = "black"

        def _is_valid_age(self):
            """Check if entered age is valid (exactly 2 digits and >= 18)."""
            if len(self.age_value) != 2:
                return False
            try:
                age = int(self.age_value)
                return age >= 18
            except ValueError:
                return False

        def get_age(self):
            """Get the entered age as integer."""
            if self._is_valid_age():
                return int(self.age_value)
            return None

        def draw(self):
            """Draw all elements."""
            self.title_stim.draw()
            self.question_stim.draw()
            self.age_display_box.draw()
            self.age_display_text.draw()
            self.numpad.draw()

            # Draw navigation button with appropriate state
            if self._is_valid_age() and self.button_active:
                self.nav_button.fillColor = "lightgrey"
                self.nav_text.color = "black"
            else:
                self.nav_button.fillColor = "darkgrey"
                self.nav_text.color = "grey"

            self.nav_button.draw()
            self.nav_text.draw()

    # ------------------------------------------------------------------
    # Demographics - Confirmation Screen
    # ------------------------------------------------------------------
    class DemographicsConfirmationScreen(ExperimentComponent):
        """Confirmation screen for demographics data."""

        def __init__(self, win):
            super().__init__()
            self.win = win
            self.gender = ""
            self.age = ""

            # Button activation delay
            self.screen_start_time = None
            self.button_delay = 2.0
            self.button_active = False

            # Title
            self.title_stim = visual.TextStim(
                win=win,
                text="Demographics Confirmation",
                pos=(0, 0.35),
                height=0.06,
                color="black",
                font="Arial",
                bold=True,
            )

            # Question
            self.question_stim = visual.TextStim(
                win=win,
                text="Please confirm your details:",
                pos=(0, 0.2),
                height=0.045,
                color="black",
                font="Arial",
            )

            # Demographics display
            self.demographics_stim = visual.TextStim(
                win=win,
                text="",
                pos=(-0.2, 0.05),
                height=0.04,
                color="black",
                font="Arial",
                anchorHoriz="left",
                alignText="left",
            )

            # Buttons
            button_y = -0.3
            button_width = 0.2
            button_height = 0.08

            # "No, go back" button (left)
            self.again_button = visual.Rect(
                win=win,
                pos=(-0.15, button_y),
                width=button_width,
                height=button_height,
                fillColor="lightgrey",
                lineColor="black",
            )

            self.again_text = visual.TextStim(
                win=win,
                text="No, go back",
                pos=(-0.15, button_y),
                height=0.03,
                color="black",
                font="Arial",
            )

            # "Yes, proceed" button (right)
            self.proceed_button = visual.Rect(
                win=win,
                pos=(0.15, button_y),
                width=button_width,
                height=button_height,
                fillColor="lightgrey",
                lineColor="black",
            )

            self.proceed_text = visual.TextStim(
                win=win,
                text="Yes, proceed",
                pos=(0.15, button_y),
                height=0.03,
                color="black",
                font="Arial",
            )

            self._register(
                self.title_stim,
                self.question_stim,
                self.demographics_stim,
                self.again_button,
                self.again_text,
                self.proceed_button,
                self.proceed_text,
            )
            self.hide()

        def show(self):
            """Override show to start the timer."""
            import time

            super().show()
            self.screen_start_time = time.time()
            self.button_active = False

        def set_demographics(self, gender, age):
            """Set the demographics data to display."""
            self.gender = gender
            self.age = age
            self.demographics_stim.text = f"Gender: {gender}\nAge: {age}"

        def check_navigation(self, mouse):
            """Check for button clicks."""
            import time

            # Check if enough time has passed
            if self.screen_start_time is not None:
                current_time = time.time()
                time_elapsed = current_time - self.screen_start_time
                if time_elapsed >= self.button_delay:
                    self.button_active = True

            if mouse.getPressed()[0] and self.button_active:
                mx, my = mouse.getPos()

                # Check "No, go back" button
                if (
                    self.again_button.pos[0] - self.again_button.width / 2
                    <= mx
                    <= self.again_button.pos[0] + self.again_button.width / 2
                    and self.again_button.pos[1] - self.again_button.height / 2
                    <= my
                    <= self.again_button.pos[1] + self.again_button.height / 2
                ):
                    return "again"

                # Check "Yes, proceed" button
                if (
                    self.proceed_button.pos[0] - self.proceed_button.width / 2
                    <= mx
                    <= self.proceed_button.pos[0] + self.proceed_button.width / 2
                    and self.proceed_button.pos[1] - self.proceed_button.height / 2
                    <= my
                    <= self.proceed_button.pos[1] + self.proceed_button.height / 2
                ):
                    return "proceed"

            return "waiting"

        def draw(self):
            """Draw all elements."""
            self.title_stim.draw()
            self.question_stim.draw()
            self.demographics_stim.draw()

            # Draw buttons with appropriate state
            if self.button_active:
                self.again_button.fillColor = "lightcoral"
                self.again_text.color = "black"
                self.proceed_button.fillColor = "lightgreen"
                self.proceed_text.color = "black"
            else:
                self.again_button.fillColor = "darkgrey"
                self.again_text.color = "grey"
                self.proceed_button.fillColor = "darkgrey"
                self.proceed_text.color = "grey"

            self.again_button.draw()
            self.again_text.draw()
            self.proceed_button.draw()
            self.proceed_text.draw()

    # ------------------------------------------------------------------
    # Instruction and Navigation Components
    # ------------------------------------------------------------------

    class NbackFeedbackScreen(ExperimentComponent):
        """Feedback screen for N-back task results."""

        def __init__(self, win):
            super().__init__()
            self.win = win

            # Button activation delay
            self.screen_start_time = None
            self.button_delay = 1.0
            self.button_active = False

            # Title
            self.title_stim = visual.TextStim(
                win=win,
                text="N-Back Task Results",
                pos=(0, 0.35),
                height=0.055,
                color="black",
                font="Arial",
                bold=True,
            )

            # Performance metrics
            self.hits_stim = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0.2),
                height=0.04,
                color="black",
                font="Arial",
            )

            self.misses_stim = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0.15),
                height=0.04,
                color="black",
                font="Arial",
            )

            self.false_alarms_stim = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0.1),
                height=0.04,
                color="black",
                font="Arial",
            )

            self.correct_rejections_stim = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0.05),
                height=0.04,
                color="black",
                font="Arial",
            )

            # Total score
            self.score_stim = visual.TextStim(
                win=win,
                text="",
                pos=(0, -0.05),
                height=0.06,
                color="black",
                font="Arial",
                bold=True,
            )

            # Interpretation
            self.interpretation_stim = visual.TextStim(
                win=win,
                text="",
                pos=(0, -0.2),
                height=0.035,
                color="darkblue",
                font="Arial",
            )

            # Navigation button
            self.nav_button = visual.Rect(
                win=win,
                pos=(0, -0.4),
                width=0.2,
                height=0.08,
                fillColor="lightgrey",
                lineColor="black",
            )

            # Button text
            self.nav_text = visual.TextStim(
                win=win,
                text="Continue",
                pos=(0, -0.4),
                height=0.035,
                color="black",
                font="Arial",
            )

            self._register(
                self.title_stim,
                self.hits_stim,
                self.misses_stim,
                self.false_alarms_stim,
                self.correct_rejections_stim,
                self.score_stim,
                self.interpretation_stim,
                self.nav_button,
                self.nav_text,
            )
            self.hide()

        def show(self):
            """Override show to start the timer."""
            import time

            super().show()
            self.screen_start_time = time.time()
            self.button_active = False

        def set_feedback(self, performance_data):
            """Set the feedback content from NbackTask.get_performance()"""

            # Extract data
            hits = performance_data["hits"]
            misses = performance_data["misses"]
            false_alarms = performance_data["false_alarms"]
            correct_rejections = performance_data["correct_rejections"]
            accuracy = performance_data["accuracy"]
            n_back_level = performance_data["n_back_level"]

            # Set individual metrics
            self.hits_stim.text = f"Hits: {hits}"
            self.misses_stim.text = f"Misses: {misses}"
            self.false_alarms_stim.text = f"False Alarms: {false_alarms}"
            self.correct_rejections_stim.text = (
                f"Correct Rejections: {correct_rejections}"
            )

            # Set total score
            score_percent = int(accuracy * 100)
            self.score_stim.text = f"Total Score: {score_percent}%"

            # Set score color based on performance
            if score_percent >= 90:
                self.score_stim.color = "green"
                interpretation = "Excellent performance!"
            elif score_percent >= 75:
                self.score_stim.color = "orange"
                interpretation = "Good performance!"
            else:
                self.score_stim.color = "darkred"
                interpretation = "You could do better!"

            self.interpretation_stim.text = interpretation

        def check_navigation(self, mouse, keyboard=None):
            """Check if navigation button was clicked or escape was pressed."""
            import time

            # Check for escape key press regardless of button delay
            if keyboard is not None:
                keys = keyboard.getKeys(keyList=["escape"])
                if keys:
                    return "escape"

            if self.screen_start_time is None:
                return False

            # Check if enough time has passed
            current_time = time.time()
            time_elapsed = current_time - self.screen_start_time

            if time_elapsed >= self.button_delay:
                self.button_active = True

                if mouse.getPressed()[0]:
                    mx, my = mouse.getPos()
                    if (
                        self.nav_button.pos[0] - self.nav_button.width / 2
                        <= mx
                        <= self.nav_button.pos[0] + self.nav_button.width / 2
                        and self.nav_button.pos[1] - self.nav_button.height / 2
                        <= my
                        <= self.nav_button.pos[1] + self.nav_button.height / 2
                    ):
                        return True

            return False

        def draw(self):
            self.title_stim.draw()
            self.hits_stim.draw()
            self.misses_stim.draw()
            self.false_alarms_stim.draw()
            self.correct_rejections_stim.draw()
            self.score_stim.draw()
            self.interpretation_stim.draw()

            # Draw button with appropriate state
            if self.button_active:
                self.nav_button.fillColor = "lightgrey"
                self.nav_text.color = "black"
            else:
                self.nav_button.fillColor = "darkgrey"
                self.nav_text.color = "grey"

            self.nav_button.draw()
            self.nav_text.draw()

    class NbackInstructionsScreen(ExperimentComponent):
        """Instructions screen for N-back task."""

        def __init__(self, win, n_back_level=1):
            super().__init__()
            self.win = win
            self.n_back_level = n_back_level

            # Button activation delay
            self.screen_start_time = None
            self.button_delay = 4.0
            self.button_active = False

            # Title
            self.title_stim = visual.TextStim(
                win=win,
                text=f"{n_back_level}-Back Task Instructions",
                pos=(0, 0.35),
                height=0.055,
                color="black",
                font="Arial",
                bold=True,
            )

            # Main instructions
            main_text = f"""You will see digits appearing one by one on the screen.

            Your task: Press SPACE when the current digit matches 
            the digit that appeared {n_back_level} position{'s' if n_back_level > 1 else ''} ago.

            """

            self.main_instructions = visual.TextStim(
                win=win,
                text=main_text,
                pos=(0, 0.15),
                height=0.038,
                color="black",
                font="Arial",
                wrapWidth=1.4,
            )

            # Example image
            image_filename = f"resources/nback_{n_back_level}_example.png"

            try:
                self.example_image = visual.ImageStim(
                    win=win,
                    image=image_filename,
                    pos=(0, -0.05),
                    size=(None, 0.28),
                )
                self.image_loaded = True
            except Exception as e:
                # Fallback to text if image fails to load
                print(f"Warning: Could not load {image_filename}, using digit examples")

                # Create fallback text with digits
                if n_back_level == 1:
                    example_text = """Sequence: 1 → 2 → 2 → 3 → 4
                                          ↑
                                 Press SPACE here! 
                      (2 matches the previous digit)"""
                else:
                    example_text = """Sequence: 1 → 2 → 1 → 3 → 2 → 4
                                      ↑           ↑
                            Press SPACE here! Press SPACE here!
                      (1 matches 2 back)    (2 matches 2 back)"""

                self.example_image = visual.TextStim(
                    win=win,
                    text=example_text,
                    pos=(0, -0.05),
                    height=0.035,
                    color="darkblue",
                    font="Arial",
                    wrapWidth=1.4,
                )
                self.image_loaded = False

            # Important notes
            notes_text = """Important:
            - Only press SPACE for matches - avoid false alarms!
            - You'll get feedback on each response
            - Try to be both fast and accurate. 
            """

            self.notes_stim = visual.TextStim(
                win=win,
                text=notes_text,
                pos=(0, -0.25),
                height=0.035,
                color="black",
                font="Arial",
                wrapWidth=1.4,
            )

            # Navigation button
            self.nav_button = visual.Rect(
                win=win,
                pos=(0, -0.4),
                width=0.3,
                height=0.08,
                fillColor="lightgrey",
                lineColor="black",
            )

            # Button text
            self.nav_text = visual.TextStim(
                win=win,
                text="Start Practice",
                pos=(0, -0.4),
                height=0.035,
                color="black",
                font="Arial",
                bold=True,
            )

            # Register components
            self._register(
                self.title_stim,
                self.main_instructions,
                self.example_image,
                self.notes_stim,
                self.nav_button,
                self.nav_text,
            )
            self.hide()

        def show(self):
            """Override show to start the timer."""
            import time

            super().show()
            self.screen_start_time = time.time()
            self.button_active = False

        def check_navigation(self, mouse, keyboard=None):
            """Check if navigation button was clicked or escape was pressed."""
            import time

            # Check for escape key press regardless of button delay
            if keyboard is not None:
                keys = keyboard.getKeys(keyList=["escape"])
                if keys:
                    return "escape"

            if self.screen_start_time is None:
                return False

            # Check if enough time has passed
            current_time = time.time()
            time_elapsed = current_time - self.screen_start_time

            if time_elapsed >= self.button_delay:
                self.button_active = True

                if mouse.getPressed()[0]:
                    mx, my = mouse.getPos()
                    if (
                        self.nav_button.pos[0] - self.nav_button.width / 2
                        <= mx
                        <= self.nav_button.pos[0] + self.nav_button.width / 2
                        and self.nav_button.pos[1] - self.nav_button.height / 2
                        <= my
                        <= self.nav_button.pos[1] + self.nav_button.height / 2
                    ):
                        return True

            return False

        def draw(self):
            self.title_stim.draw()
            self.main_instructions.draw()
            self.example_image.draw()
            self.notes_stim.draw()

            # Draw button with appropriate state
            if self.button_active:
                self.nav_button.fillColor = "lightgrey"
                self.nav_text.color = "black"
            else:
                self.nav_button.fillColor = "darkgrey"
                self.nav_text.color = "grey"

            self.nav_button.draw()
            self.nav_text.draw()

    class InstructionScreen(ExperimentComponent):
        """Generic instruction screen."""

        def __init__(
            self,
            win,
            title="",
            content="",
            button_text="Continue",
            button_width=0.2,
            button_fill_color="lightgrey",
            button_delay=4.0,
            letter_height=0.03,
        ):
            super().__init__()
            self.win = win

            # Button activation delay
            self.screen_start_time = None
            self.button_delay = button_delay
            self.button_active = False

            self.button_fill_color = button_fill_color

            self.letter_height = letter_height

            # Title
            self.title_stim = visual.TextStim(
                win=win,
                text=title,
                pos=(0, 0.3),
                height=0.06,
                color="black",
                font="Arial",
                bold=True,
            )

            # Content
            self.content_stim = visual.TextStim(
                win=win,
                text=content,
                pos=(0, 0),
                height=self.letter_height,
                color="black",
                font="Arial",
                wrapWidth=1.5,
            )

            # Navigation button
            self.nav_button = visual.Rect(
                win=win,
                pos=(0, -0.4),
                width=button_width,
                height=0.08,
                fillColor=self.button_fill_color,
                lineColor="black",
            )

            # Button text
            self.nav_text = visual.TextStim(
                win=win,
                text=button_text,
                pos=(0, -0.4),
                height=0.035,
                color="black",
                font="Arial",
            )

            self._register(
                self.title_stim, self.content_stim, self.nav_button, self.nav_text
            )
            self.hide()

        def show(self):
            """Override show to start the timer."""
            import time

            super().show()
            self.screen_start_time = time.time()
            self.button_active = False

        def check_navigation(self, mouse, keyboard=None):
            """Check if navigation button was clicked or escape was pressed."""
            import time

            # Check for escape key press regardless of button delay
            if keyboard is not None:
                keys = keyboard.getKeys(keyList=["escape"])
                if keys:
                    return "escape"

            if self.screen_start_time is None:
                return False

            # Check if enough time has passed
            current_time = time.time()
            time_elapsed = current_time - self.screen_start_time

            if time_elapsed >= self.button_delay:
                self.button_active = True

                if mouse.getPressed()[0]:
                    mx, my = mouse.getPos()
                    if (
                        self.nav_button.pos[0] - self.nav_button.width / 2
                        <= mx
                        <= self.nav_button.pos[0] + self.nav_button.width / 2
                        and self.nav_button.pos[1] - self.nav_button.height / 2
                        <= my
                        <= self.nav_button.pos[1] + self.nav_button.height / 2
                    ):
                        return True

            return False

        def draw(self):
            self.title_stim.draw()
            self.content_stim.draw()

            # Draw button with appropriate state
            if self.button_active:
                self.nav_button.fillColor = self.button_fill_color
                self.nav_text.color = "black"
            else:
                self.nav_button.fillColor = "darkgrey"
                self.nav_text.color = "grey"

            self.nav_button.draw()
            self.nav_text.draw()

    class TypingFeedbackScreen(ExperimentComponent):
        """Feedback screen for typing task results."""

        def __init__(self, win):
            super().__init__()
            self.win = win

            # Button activation delay
            self.screen_start_time = None
            self.button_delay = 1.0
            self.button_active = False

            # Title
            self.title_stim = visual.TextStim(
                win=win,
                text="Trial Result",
                pos=(0, 0.3),
                height=0.06,
                color="black",
                font="Arial",
                bold=True,
            )

            # Result message
            self.result_stim = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0.1),
                height=0.05,
                color="black",
                font="Arial",
                bold=True,
            )

            # Target word
            self.target_stim = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0.0),
                height=0.05,
                color="black",
                font="Arial",
            )

            # Typed word
            self.typed_stim = visual.TextStim(
                win=win,
                text="",
                pos=(0, -0.1),
                height=0.05,
                color="black",
                font="Arial",
            )

            # Navigation button
            self.nav_button = visual.Rect(
                win=win,
                pos=(0, -0.4),
                width=0.2,
                height=0.08,
                fillColor="lightgrey",
                lineColor="black",
            )

            # Button text
            self.nav_text = visual.TextStim(
                win=win,
                text="Continue",
                pos=(0, -0.4),
                height=0.035,
                color="black",
                font="Arial",
            )

            self._register(
                self.title_stim,
                self.result_stim,
                self.target_stim,
                self.typed_stim,
                self.nav_button,
                self.nav_text,
            )
            self.hide()

        def show(self):
            """Override show to start the timer."""
            import time

            super().show()
            self.screen_start_time = time.time()
            self.button_active = False

        def set_feedback(self, target_word, typed_word):
            """Set the feedback content."""
            self.target_stim.text = f"Word: {target_word}"
            self.typed_stim.text = f"Answer: {typed_word}"

            is_correct = target_word.upper() == typed_word.upper()
            if is_correct:
                self.result_stim.text = "CORRECT!"
                self.result_stim.color = "green"
            else:
                self.result_stim.text = "INCORRECT"
                self.result_stim.color = "darkred"

        def check_navigation(self, mouse, keyboard=None):
            """Check if navigation button was clicked or escape was pressed."""
            import time

            # Check for escape key press regardless of button delay
            if keyboard is not None:
                keys = keyboard.getKeys(keyList=["escape"])
                if keys:
                    return "escape"

            if self.screen_start_time is None:
                return False

            # Check if enough time has passed
            current_time = time.time()
            time_elapsed = current_time - self.screen_start_time

            if time_elapsed >= self.button_delay:
                self.button_active = True

                if mouse.getPressed()[0]:
                    mx, my = mouse.getPos()
                    if (
                        self.nav_button.pos[0] - self.nav_button.width / 2
                        <= mx
                        <= self.nav_button.pos[0] + self.nav_button.width / 2
                        and self.nav_button.pos[1] - self.nav_button.height / 2
                        <= my
                        <= self.nav_button.pos[1] + self.nav_button.height / 2
                    ):
                        return True

            return False

        def draw(self):
            self.title_stim.draw()
            self.result_stim.draw()
            self.target_stim.draw()
            self.typed_stim.draw()

            # Draw button with appropriate state
            if self.button_active:
                self.nav_button.fillColor = "lightgrey"
                self.nav_text.color = "black"
            else:
                self.nav_button.fillColor = "darkgrey"
                self.nav_text.color = "grey"

            self.nav_button.draw()
            self.nav_text.draw()

    class CombinedTaskFeedbackScreen(ExperimentComponent):
        """Feedback screen showing results from both typing and n-back tasks."""

        def __init__(self, win):
            super().__init__()
            self.win = win

            # Button activation delay
            self.screen_start_time = None
            self.button_delay = 1.0
            self.button_active = False

            # Title
            self.title_stim = visual.TextStim(
                win=win,
                text="Task Results",
                pos=(0, 0.35),
                height=0.055,
                color="black",
                font="Arial",
                bold=True,
            )

            # Typing task section
            self.typing_title = visual.TextStim(
                win=win,
                text="Typing Task:",
                pos=(0, 0.22),
                height=0.04,
                color="black",
                font="Arial",
                bold=True,
            )

            self.typing_result = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0.16),
                height=0.035,
                color="black",
                font="Arial",
            )

            self.typing_details = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0.11),
                height=0.03,
                color="black",
                font="Arial",
            )

            # N-back task section
            self.nback_title = visual.TextStim(
                win=win,
                text="N-Back Task:",
                pos=(0, 0.03),
                height=0.04,
                color="black",
                font="Arial",
                bold=True,
            )

            self.nback_score = visual.TextStim(
                win=win,
                text="",
                pos=(0, -0.03),
                height=0.035,
                color="black",
                font="Arial",
            )

            self.nback_details = visual.TextStim(
                win=win,
                text="",
                pos=(0, -0.08),
                height=0.03,
                color="black",
                font="Arial",
            )

            # Overall performance
            self.overall_text = visual.TextStim(
                win=win,
                text="",
                pos=(0, -0.17),
                height=0.04,
                color="darkblue",
                font="Arial",
                bold=True,
            )

            # Navigation button
            self.nav_button = visual.Rect(
                win=win,
                pos=(0, -0.4),
                width=0.2,
                height=0.08,
                fillColor="lightgrey",
                lineColor="black",
            )

            self.nav_text = visual.TextStim(
                win=win,
                text="Continue",
                pos=(0, -0.4),
                height=0.035,
                color="black",
                font="Arial",
            )

            self._register(
                self.title_stim,
                self.typing_title,
                self.typing_result,
                self.typing_details,
                self.nback_title,
                self.nback_score,
                self.nback_details,
                self.overall_text,
                self.nav_button,
                self.nav_text,
            )
            self.hide()

        def show(self):
            """Override show to start the timer."""
            import time

            super().show()
            self.screen_start_time = time.time()
            self.button_active = False

        def set_feedback(self, typing_target, typing_entered, nback_performance):
            """Set feedback for both tasks."""
            # Typing feedback
            typing_correct = typing_target.upper() == typing_entered.upper()
            if typing_correct:
                self.typing_result.text = "CORRECT!"
                self.typing_result.color = "green"
            else:
                self.typing_result.text = "INCORRECT"
                self.typing_result.color = "darkred"

            self.typing_details.text = (
                f"Target: {typing_target} | Entered: {typing_entered}"
            )

            # N-back feedback
            accuracy_percent = int(nback_performance["accuracy"] * 100)
            self.nback_score.text = f"Accuracy: {accuracy_percent}%"

            if accuracy_percent >= 90:
                self.nback_score.color = "green"
            elif accuracy_percent >= 75:
                self.nback_score.color = "orange"
            else:
                self.nback_score.color = "darkred"

            self.nback_details.text = f"Hits: {nback_performance['hits']} | Misses: {nback_performance['misses']} | False Alarms: {nback_performance['false_alarms']}"

            # Overall performance
            if typing_correct and accuracy_percent >= 75:
                self.overall_text.text = "Great job on both tasks!"
                self.overall_text.color = "green"
            elif typing_correct or accuracy_percent >= 75:
                self.overall_text.text = "Good performance! Keep practicing."
                self.overall_text.color = "darkblue"
            else:
                self.overall_text.text = "Keep practicing - you'll improve!"
                self.overall_text.color = "darkred"

        def check_navigation(self, mouse, keyboard=None):
            """Check if navigation button was clicked."""
            import time

            if keyboard is not None:
                keys = keyboard.getKeys(keyList=["escape"])
                if keys:
                    return "escape"

            if self.screen_start_time is None:
                return False

            current_time = time.time()
            time_elapsed = current_time - self.screen_start_time

            if time_elapsed >= self.button_delay:
                self.button_active = True

                if mouse.getPressed()[0]:
                    mx, my = mouse.getPos()
                    if (
                        self.nav_button.pos[0] - self.nav_button.width / 2
                        <= mx
                        <= self.nav_button.pos[0] + self.nav_button.width / 2
                        and self.nav_button.pos[1] - self.nav_button.height / 2
                        <= my
                        <= self.nav_button.pos[1] + self.nav_button.height / 2
                    ):
                        return True

            return False

        def draw(self):
            """Draw all elements."""
            self.title_stim.draw()
            self.typing_title.draw()
            self.typing_result.draw()
            self.typing_details.draw()
            self.nback_title.draw()
            self.nback_score.draw()
            self.nback_details.draw()
            self.overall_text.draw()

            if self.button_active:
                self.nav_button.fillColor = "lightgrey"
                self.nav_text.color = "black"
            else:
                self.nav_button.fillColor = "darkgrey"
                self.nav_text.color = "grey"

            self.nav_button.draw()
            self.nav_text.draw()

    class TimingTaskFeedbackScreen(ExperimentComponent):
        """Feedback screen showing task results plus timing estimation accuracy."""

        def __init__(self, win):
            super().__init__()
            self.win = win

            # Button activation delay
            self.screen_start_time = None
            self.button_delay = 1.0
            self.button_active = False

            # Title
            self.title_stim = visual.TextStim(
                win=win,
                text="Results with Time Estimation",
                pos=(0, 0.35),
                height=0.055,
                color="black",
                font="Arial",
                bold=True,
            )

            # Time estimation section
            self.timing_title = visual.TextStim(
                win=win,
                text="Time Estimation:",
                pos=(0, 0.22),
                height=0.04,
                color="black",
                font="Arial",
                bold=True,
            )

            self.actual_time = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0.16),
                height=0.035,
                color="black",
                font="Arial",
            )

            self.estimated_time = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0.11),
                height=0.035,
                color="black",
                font="Arial",
            )

            self.timing_feedback = visual.TextStim(
                win=win,
                text="",
                pos=(0, 0.05),
                height=0.035,
                color="darkblue",
                font="Arial",
                bold=True,
            )

            # Divider line
            self.divider = visual.Line(
                win=win,
                start=(-0.4, -0.02),
                end=(0.4, -0.02),
                lineColor="grey",
                lineWidth=2,
            )

            # Task performance section (dynamic based on tasks)
            self.task_results = []

            # Navigation button
            self.nav_button = visual.Rect(
                win=win,
                pos=(0, -0.45),
                width=0.2,
                height=0.08,
                fillColor="lightgrey",
                lineColor="black",
            )

            self.nav_text = visual.TextStim(
                win=win,
                text="Continue",
                pos=(0, -0.45),
                height=0.035,
                color="black",
                font="Arial",
            )

            # Register base components
            self._register(
                self.title_stim,
                self.timing_title,
                self.actual_time,
                self.estimated_time,
                self.timing_feedback,
                self.divider,
                self.nav_button,
                self.nav_text,
            )
            self.hide()

        def show(self):
            """Override show to start the timer."""
            import time

            super().show()
            self.screen_start_time = time.time()
            self.button_active = False

            # Show task results
            for stim in self.task_results:
                stim.autoDraw = True

        def hide(self):
            """Override hide to also hide dynamic elements."""
            super().hide()
            for stim in self.task_results:
                stim.autoDraw = False

        def set_typing_feedback(self, target_word, entered_text, y_start=-0.1):
            """Add typing task feedback."""
            # Clear previous task results
            for stim in self.task_results:
                stim.autoDraw = False
            self.task_results = []

            # Create typing feedback
            typing_title = visual.TextStim(
                win=self.win,
                text="Typing Task:",
                pos=(0, y_start),
                height=0.04,
                color="black",
                font="Arial",
                bold=True,
            )

            typing_correct = target_word.upper() == entered_text.upper()
            result_text = "CORRECT!" if typing_correct else "INCORRECT"
            result_color = "green" if typing_correct else "darkred"

            typing_result = visual.TextStim(
                win=self.win,
                text=result_text,
                pos=(0, y_start - 0.05),
                height=0.035,
                color=result_color,
                font="Arial",
            )

            typing_details = visual.TextStim(
                win=self.win,
                text=f"Target: {target_word} | Entered: {entered_text}",
                pos=(0, y_start - 0.1),
                height=0.03,
                color="black",
                font="Arial",
            )

            self.task_results = [typing_title, typing_result, typing_details]
            self._register(typing_title, typing_result, typing_details)

        def set_nback_feedback(self, nback_performance, y_start=-0.1):
            """Add n-back task feedback."""
            # Clear previous task results
            for stim in self.task_results:
                stim.autoDraw = False
            self.task_results = []

            # Create n-back feedback
            nback_title = visual.TextStim(
                win=self.win,
                text=f"{nback_performance['n_back_level']}-Back Task:",
                pos=(0, y_start),
                height=0.04,
                color="black",
                font="Arial",
                bold=True,
            )

            accuracy_percent = int(nback_performance["accuracy"] * 100)
            score_color = (
                "green"
                if accuracy_percent >= 90
                else "orange" if accuracy_percent >= 75 else "darkred"
            )

            nback_score = visual.TextStim(
                win=self.win,
                text=f"Accuracy: {accuracy_percent}%",
                pos=(0, y_start - 0.05),
                height=0.035,
                color=score_color,
                font="Arial",
            )

            nback_details = visual.TextStim(
                win=self.win,
                text=f"Hits: {nback_performance['hits']} | Misses: {nback_performance['misses']} | False Alarms: {nback_performance['false_alarms']}",
                pos=(0, y_start - 0.1),
                height=0.03,
                color="black",
                font="Arial",
            )

            self.task_results = [nback_title, nback_score, nback_details]
            self._register(nback_title, nback_score, nback_details)

        def set_combined_feedback(
            self, typing_target, typing_entered, nback_performance
        ):
            """Add both typing and n-back feedback."""
            # Clear previous task results
            for stim in self.task_results:
                stim.autoDraw = False
            self.task_results = []

            # Typing section
            typing_title = visual.TextStim(
                win=self.win,
                text="Typing Task:",
                pos=(0, -0.1),
                height=0.04,
                color="black",
                font="Arial",
                bold=True,
            )

            typing_correct = typing_target.upper() == typing_entered.upper()
            typing_result = visual.TextStim(
                win=self.win,
                text="CORRECT!" if typing_correct else "INCORRECT",
                pos=(0, -0.15),
                height=0.035,
                color="green" if typing_correct else "darkred",
                font="Arial",
            )

            # N-back section
            nback_title = visual.TextStim(
                win=self.win,
                text=f"{nback_performance['n_back_level']}-Back Task:",
                pos=(0, -0.23),
                height=0.04,
                color="black",
                font="Arial",
                bold=True,
            )

            accuracy_percent = int(nback_performance["accuracy"] * 100)
            nback_score = visual.TextStim(
                win=self.win,
                text=f"Accuracy: {accuracy_percent}%",
                pos=(0, -0.28),
                height=0.035,
                color=(
                    "green"
                    if accuracy_percent >= 90
                    else "orange" if accuracy_percent >= 75 else "darkred"
                ),
                font="Arial",
            )

            self.task_results = [typing_title, typing_result, nback_title, nback_score]
            for stim in self.task_results:
                self._register(stim)

        def set_timing_feedback(self, actual_seconds, estimated_seconds):
            """Set the timing estimation feedback."""
            # Format times with proper rounding
            actual_min = int(actual_seconds // 60)
            actual_sec = round(actual_seconds % 60)
            est_min = int(estimated_seconds // 60)
            est_sec = round(estimated_seconds % 60)

            # Handle edge case where rounding seconds to 60
            if actual_sec == 60:
                actual_min += 1
                actual_sec = 0
            if est_sec == 60:
                est_min += 1
                est_sec = 0

            self.actual_time.text = f"Actual time: {actual_min}:{actual_sec:02d}"
            self.estimated_time.text = f"Your estimate: {est_min}:{est_sec:02d}"

            # Calculate difference
            diff_seconds = estimated_seconds - actual_seconds
            abs_diff = abs(diff_seconds)

            # Generate feedback message
            if abs_diff < 2:
                self.timing_feedback.text = "Excellent time estimation!"
                self.timing_feedback.color = "green"
            elif abs_diff < 8:
                if diff_seconds > 0:
                    self.timing_feedback.text = (
                        f"Good estimate! You were {round(abs_diff)} seconds too long."
                    )
                else:
                    self.timing_feedback.text = (
                        f"Good estimate! You were {round(abs_diff)} seconds too short."
                    )
                self.timing_feedback.color = "darkblue"
            else:
                if diff_seconds > 0:
                    self.timing_feedback.text = (
                        f"You were {round(abs_diff)} seconds too long. Keep practicing!"
                    )
                else:
                    self.timing_feedback.text = f"You were {round(abs_diff)} seconds too short. Keep practicing!"
                self.timing_feedback.color = "darkred"

        def check_navigation(self, mouse, keyboard=None):
            """Check if navigation button was clicked."""
            import time

            if keyboard is not None:
                keys = keyboard.getKeys(keyList=["escape"])
                if keys:
                    return "escape"

            if self.screen_start_time is None:
                return False

            current_time = time.time()
            time_elapsed = current_time - self.screen_start_time

            if time_elapsed >= self.button_delay:
                self.button_active = True

                if mouse.getPressed()[0]:
                    mx, my = mouse.getPos()
                    if (
                        self.nav_button.pos[0] - self.nav_button.width / 2
                        <= mx
                        <= self.nav_button.pos[0] + self.nav_button.width / 2
                        and self.nav_button.pos[1] - self.nav_button.height / 2
                        <= my
                        <= self.nav_button.pos[1] + self.nav_button.height / 2
                    ):
                        return True

            return False

        def draw(self):
            """Draw all elements."""
            self.title_stim.draw()

            # Draw timing section first
            self.timing_title.draw()
            self.actual_time.draw()
            self.estimated_time.draw()
            self.timing_feedback.draw()

            # Draw divider
            self.divider.draw()

            # Draw task results
            for stim in self.task_results:
                stim.draw()

            # Draw button
            if self.button_active:
                self.nav_button.fillColor = "lightgrey"
                self.nav_text.color = "black"
            else:
                self.nav_button.fillColor = "darkgrey"
                self.nav_text.color = "grey"

            self.nav_button.draw()
            self.nav_text.draw()

    # ------------------------------------------------------------------
    # Main Phase Decision Screen
    # ------------------------------------------------------------------
    class MainPhaseDecisionScreen(ExperimentComponent):
        """Decision screen before main phase with abort/continue options."""

        def __init__(self, win):
            super().__init__()
            self.win = win

            # Button activation delay
            self.screen_start_time = None
            self.button_delay = 30.0
            self.button_active = False

            # Title
            self.title_stim = visual.TextStim(
                win=win,
                text="Main Experiment",
                pos=(0, 0.35),
                height=0.06,
                color="black",
                font="Arial",
                bold=True,
            )

            # Main content
            content_text = DECISION_SCREEN_CONTENT

            self.content_stim = visual.TextStim(
                win=win,
                text=content_text,
                pos=(0, 0),
                height=0.025,
                color="black",
                font="Arial",
                wrapWidth=1.5,
            )

            # Buttons
            button_y = -0.4
            button_width = 0.25
            button_height = 0.08
            button_spacing = 0.3

            # "Abort Experiment" button (left)
            self.abort_button = visual.Rect(
                win=win,
                pos=(-button_spacing / 2, button_y),
                width=button_width,
                height=button_height,
                fillColor="lightcoral",
                lineColor="black",
            )

            self.abort_text = visual.TextStim(
                win=win,
                text="Abort Experiment",
                pos=(-button_spacing / 2, button_y),
                height=0.03,
                color="black",
                font="Arial",
            )

            # "To Main Phase" button (right)
            self.continue_button = visual.Rect(
                win=win,
                pos=(button_spacing / 2, button_y),
                width=button_width,
                height=button_height,
                fillColor="lightgreen",
                lineColor="black",
            )

            self.continue_text = visual.TextStim(
                win=win,
                text="To Main Phase",
                pos=(button_spacing / 2, button_y),
                height=0.03,
                color="black",
                font="Arial",
            )

            self._register(
                self.title_stim,
                self.content_stim,
                self.abort_button,
                self.abort_text,
                self.continue_button,
                self.continue_text,
            )
            self.hide()

        def show(self):
            """Override show to start the timer."""
            import time

            super().show()
            self.screen_start_time = time.time()
            self.button_active = False

        def check_navigation(self, mouse, keyboard=None):
            """Check for button clicks."""
            import time

            # Check for escape key press regardless of button delay
            if keyboard is not None:
                keys = keyboard.getKeys(keyList=["escape"])
                if keys:
                    return "escape"

            if self.screen_start_time is None:
                return "waiting"

            # Check if enough time has passed
            current_time = time.time()
            time_elapsed = current_time - self.screen_start_time

            if time_elapsed >= self.button_delay:
                self.button_active = True

                if mouse.getPressed()[0]:
                    mx, my = mouse.getPos()

                    # Check "Abort Experiment" button
                    if (
                        self.abort_button.pos[0] - self.abort_button.width / 2
                        <= mx
                        <= self.abort_button.pos[0] + self.abort_button.width / 2
                        and self.abort_button.pos[1] - self.abort_button.height / 2
                        <= my
                        <= self.abort_button.pos[1] + self.abort_button.height / 2
                    ):
                        return "abort"

                    # Check "To Main Phase" button
                    if (
                        self.continue_button.pos[0] - self.continue_button.width / 2
                        <= mx
                        <= self.continue_button.pos[0] + self.continue_button.width / 2
                        and self.continue_button.pos[1]
                        - self.continue_button.height / 2
                        <= my
                        <= self.continue_button.pos[1] + self.continue_button.height / 2
                    ):
                        return "continue"

            return "waiting"

        def draw(self):
            """Draw all elements."""
            self.title_stim.draw()
            self.content_stim.draw()

            # Draw buttons with appropriate state
            if self.button_active:
                self.abort_button.fillColor = "lightcoral"
                self.abort_text.color = "black"
                self.continue_button.fillColor = "lightgreen"
                self.continue_text.color = "black"
            else:
                self.abort_button.fillColor = "darkgrey"
                self.abort_text.color = "grey"
                self.continue_button.fillColor = "darkgrey"
                self.continue_text.color = "grey"

            self.abort_button.draw()
            self.abort_text.draw()
            self.continue_button.draw()
            self.continue_text.draw()

    # ------------------------------------------------------------------
    # DataLogger Class
    # ------------------------------------------------------------------
    class DataLogger:
        """Clean interface for logging experiment data with cm_ prefix."""

        def __init__(self, thisExp):
            self.thisExp = thisExp
            self.current_trial_data = {}

        def log(self, variable_name, value):
            """Log a variable with cm_ prefix."""
            full_name = (
                f"cm_{variable_name}"
                if not variable_name.startswith("cm_")
                else variable_name
            )
            self.thisExp.addData(full_name, value)
            self.current_trial_data[full_name] = value

        def log_multiple(self, data_dict):
            """Log multiple variables at once."""
            for var_name, value in data_dict.items():
                self.log(var_name, value)

        def get_current_data(self):
            """Get all data logged for current trial."""
            return self.current_trial_data.copy()

        def clear_trial_data(self):
            """Clear current trial data (call at start of new trial)."""
            self.current_trial_data = {}

        def log_timing_start(self, timer_name):
            """Helper to log start times."""
            timestamp = globalClock.getTime()
            self.log(f"{timer_name}_start_timestamp", timestamp)
            return timestamp

        def log_timing_end(self, timer_name, start_timestamp=None):
            """Helper to log end times and calculate duration."""
            timestamp = globalClock.getTime()
            self.log(f"{timer_name}_end_timestamp", timestamp)

            if start_timestamp is not None:
                duration = timestamp - start_timestamp
                self.log(f"{timer_name}_duration", duration)
                return timestamp, duration
            return timestamp

        # Helper functions
        def calculate_character_accuracy(self, target, entered):
            """Calculate character-level accuracy (0-1)."""
            if len(target) == 0:
                return 1.0 if len(entered) == 0 else 0.0

            # Simple character matching
            max_len = max(len(target), len(entered))
            matches = sum(
                1
                for i in range(max_len)
                if i < len(target)
                and i < len(entered)
                and target[i].upper() == entered[i].upper()
            )

            return matches / len(target)

    # ------------------------------------------------------------------
    # Initialize variables
    # ------------------------------------------------------------------
    dl = DataLogger(thisExp)

    trial_start_time = None
    typing_start_time = None
    nback_start_time = None
    nback_end_time = None
    time_estimate_string = None
    time_estimate_seconds = None

    current_nback_params = None

    # ------------------------------------------------------------------
    # Global OK button (used by PrimaryTask)
    # ------------------------------------------------------------------
    OK_BUTTON_POS = (0.4, -0.45)
    OK_BUTTON_SIZE = (0.15, 0.07)

    ok_button_rect = visual.Rect(
        win=win,
        pos=OK_BUTTON_POS,
        width=OK_BUTTON_SIZE[0],
        height=OK_BUTTON_SIZE[1],
        fillColor="lightgrey",
        lineColor="black",
    )
    ok_button_text = visual.TextStim(
        win=win, text="OK", pos=OK_BUTTON_POS, height=0.04, color="black"
    )

    okButtonPos = OK_BUTTON_POS
    okButtonWidth = OK_BUTTON_SIZE[0]
    okButtonHeight = OK_BUTTON_SIZE[1]

    # Run 'Begin Experiment' code from set_globals
    experimentData = []

    # ------------------------------------------------------------------
    # Generate conditions DataFrame
    # ------------------------------------------------------------------
    import utils.create_conditions as cc
    from argparse import Namespace
    from pathlib import Path

    args = Namespace(
        trials=40,
        practice_trials=8,
        block_size=8,
        participants=1,
        seed=None,
        text_lengths=[6, 7, 8, 9],
        nback_levels=[1, 2],
        duration_range=(6, 14),
        stim_duration=1.2,
        isi=(0.25, 0.4),
        match_rate=30,
        stimuli="digits",
        quantum=0.01,
    )
    args.duration_range = cc.parse_range_arg(list(args.duration_range))
    args.isi = cc.parse_range_arg(list(args.isi))

    pid = int(expInfo["participant"])
    df = cc.generate_participant_conditions(pid, args)

    output_dir = Path(_thisDir) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"participant_{pid:03d}.csv"
    df.to_csv(csv_path, index=False)

    currentCSV = str(csv_path)
    expInfo["csv_filename_full"] = currentCSV

    import csv, json, ast

    # Read the participant's CSV once to get all needed info
    csv_path = expInfo["csv_filename_full"]
    practice_count = 0
    main_count = 0
    practice_blocks = set()
    main_blocks = set()
    practice_trials_per_block = set()
    main_trials_per_block = set()
    NBACK_STIM_DURATION = None

    with open(csv_path, newline="") as fp:
        for row in csv.DictReader(fp):
            if row["practice"].lower() == "true":
                practice_count += 1
                practice_blocks.add(int(row["block"]))
                practice_trials_per_block.add(int(row["trial_in_block"]))
            else:
                main_count += 1
                main_blocks.add(int(row["block"]))
                main_trials_per_block.add(int(row["trial_in_block"]))

            if NBACK_STIM_DURATION is None:
                stim_durations = ast.literal_eval(row["stim_durations"])
                if stim_durations:
                    NBACK_STIM_DURATION = stim_durations[0]

    # Derive experiment parameters from collected data
    NUM_PRACTICE_TRIALS = practice_count
    NUM_BLOCKS = len(main_blocks)
    NUM_TRIALS = max(main_trials_per_block) if main_trials_per_block else 0

    # Validate the structure
    if NUM_TRIALS > 0 and main_count != NUM_BLOCKS * NUM_TRIALS:
        raise ValueError(
            f"Main trial count ({main_count}) doesn't match NUM_BLOCKS ({NUM_BLOCKS}) × NUM_TRIALS ({NUM_TRIALS})"
        )

    RUN_ORIENTATION = True  # optionally skip orientation for debugging purposes
    RUN_PRACTICE = True

    NUM_INTERRUPTIONS = 1

    # VALIDATION VALUES
    TOTAL_PRIMARY_TRIALS = NUM_BLOCKS * NUM_TRIALS
    TOTAL_N_BACK_TRIALS = NUM_INTERRUPTIONS * TOTAL_PRIMARY_TRIALS

    # SET PER TRIAL
    INTERRUPTION_CONDITION = ""

    # ------------------------------------------------------------------
    # Feature toggles & parameters
    # ------------------------------------------------------------------
    MASK_COMPLETED_COLUMNS = True
    TYPING_DELAY_ENABLED = False
    TYPING_DELAY = 0.1
    INTERRUPT_DELAY = 0.4
    NUMPAD_FEEDBACK = True
    NBACK_MASK_ENABLED = True

    SHOW_BLOCK_INSTRUCTIONS = True
    SHOW_TRIAL_INSTRUCTIONS = False

    MAX_CLOCK_VALUE = "1:59"
    PRIMARY_DIFFICULTY = "hard"  # always the "invisible" typing

    SHOW_OK_BUTTON = False

    # Fixed patterns for time estimation phases
    PHASE2_NBACK_PATTERN = ["2-back", "1-back"]
    PHASE3_NBACK_PATTERN = ["1-back", "2-back", "1-back", "2-back"]
    PHASE3_SEQUENCE_INDICES = [1, 1, 2, 2]

    trial_nback_instructions_text = ""

    experimentAborted = False

    # Practice words for typing task
    ORIENTATION_TYPING_WORDS = ["HELLO", "WORLD", "TYPING"]

    # Practice words for invisible typing task
    ORIENTATION_INVISIBLE_WORDS = ["MAGIC", "BRAIN", "FOCUS"]

    # ORIENTATION_NBACK_DATA
    ORIENTATION_NBACK_DATA = {
        "1-back": [
            {
                "n_back_level": 1,
                "num_stims": 8,
                "stim_list": ["3", "7", "7", "2", "5", "5", "1", "8"],
                "stim_durations": [1.2] * 8,
                "match_positions": [2, 5],
                "isi_times": [0.3] * 8,
                "total_duration": 12.0,
            },
            {
                "n_back_level": 1,
                "num_stims": 6,
                "stim_list": ["1", "4", "4", "9", "2", "2"],
                "stim_durations": [1.2] * 6,
                "match_positions": [2, 5],
                "isi_times": [0.3] * 6,
                "total_duration": 9.0,
            },
            {
                "n_back_level": 1,
                "num_stims": 9,
                "stim_list": ["8", "1", "1", "5", "9", "9", "3", "2", "2"],
                "stim_durations": [1.2] * 9,
                "match_positions": [2, 5, 8],
                "isi_times": [0.3] * 9,
                "total_duration": 13.5,
            },
        ],
        "2-back": [
            {
                "n_back_level": 2,
                "num_stims": 10,
                "stim_list": ["8", "3", "8", "6", "8", "1", "5", "2", "5", "7"],
                "stim_durations": [1.2] * 10,
                "match_positions": [2, 4, 8],
                "isi_times": [0.3] * 10,
                "total_duration": 15.0,
            },
            {
                "n_back_level": 2,
                "num_stims": 7,
                "stim_list": ["2", "9", "2", "4", "2", "7", "4"],
                "stim_durations": [1.2] * 7,
                "match_positions": [2, 4, 6],
                "isi_times": [0.3] * 7,
                "total_duration": 10.5,
            },
            {
                "n_back_level": 2,
                "num_stims": 8,
                "stim_list": ["5", "1", "5", "8", "5", "9", "8", "3"],
                "stim_durations": [1.2] * 8,
                "match_positions": [2, 4, 6],
                "isi_times": [0.3] * 8,
                "total_duration": 12.0,
            },
        ],
    }

    # Practice words for interruption training
    ORIENTATION_INTERRUPTION_WORDS = [
        "BASKET",
        "PLANET",
        "KITCHEN",
        "WINDOW",
        "PICTURE",
        "FOREST",
    ]

    # Interruption positions for orientation interruption training
    ORIENTATION_INTERRUPTION_POSITIONS = [
        [3],
        [4],
        [2],
        [3],
        [5],
        [2],
    ]

    # Orientation interruption nback data
    ORIENTATION_INTERRUPTION_NBACK_DATA = {
        "1-back": [
            {
                "n_back_level": 1,
                "num_stims": 7,
                "stim_list": ["8", "3", "3", "5", "9", "9", "4"],
                "stim_durations": [1.2] * 7,
                "match_positions": [2, 5],
                "isi_times": [0.3] * 7,
                "total_duration": 10.5,
            },
            {
                "n_back_level": 1,
                "num_stims": 10,
                "stim_list": ["4", "1", "1", "7", "2", "2", "6", "8", "3", "3"],
                "stim_durations": [1.2] * 10,
                "match_positions": [2, 5, 9],
                "isi_times": [0.3] * 10,
                "total_duration": 15.0,
            },
            {
                "n_back_level": 1,
                "num_stims": 6,
                "stim_list": ["6", "8", "8", "3", "5", "5"],
                "stim_durations": [1.2] * 6,
                "match_positions": [2, 5],
                "isi_times": [0.3] * 6,
                "total_duration": 9.0,
            },
        ],
        "2-back": [
            {
                "n_back_level": 2,
                "num_stims": 9,
                "stim_list": ["2", "7", "2", "4", "2", "1", "4", "3", "4"],
                "stim_durations": [1.2] * 9,
                "match_positions": [2, 4, 6, 8],
                "isi_times": [0.3] * 9,
                "total_duration": 13.5,
            },
            {
                "n_back_level": 2,
                "num_stims": 8,
                "stim_list": ["5", "3", "5", "9", "5", "6", "9", "6"],
                "stim_durations": [1.2] * 8,
                "match_positions": [2, 4, 6, 7],
                "isi_times": [0.3] * 8,
                "total_duration": 12.0,
            },
            {
                "n_back_level": 2,
                "num_stims": 7,
                "stim_list": ["1", "8", "1", "2", "1", "4", "1"],
                "stim_durations": [1.2] * 7,
                "match_positions": [2, 4, 6],
                "isi_times": [0.3] * 7,
                "total_duration": 10.5,
            },
        ],
    }

    # Practice words for time estimation training
    ORIENTATION_TIME_ESTIMATION_WORDS = [
        "TIMING",
        "CLOCKS",
        "COMBINE",
        "MIXTURE",
        "COMPLEX",
        "BALANCE",
    ]

    # Interruption positions for time estimation phase 3
    PHASE3_INTERRUPT_POSITIONS = [
        [3],
        [4],
        [2],
        [5],
    ]

    # Orientation time nback data
    ORIENTATION_TIME_NBACK_DATA = {
        "1-back": [
            {
                "n_back_level": 1,
                "num_stims": 8,
                "stim_list": ["2", "6", "6", "1", "9", "9", "3", "7"],
                "stim_durations": [1.2] * 8,
                "match_positions": [2, 5],
                "isi_times": [0.3] * 8,
                "total_duration": 12.0,
            },
            {
                "n_back_level": 1,
                "num_stims": 9,
                "stim_list": ["5", "2", "2", "8", "4", "4", "6", "1", "1"],
                "stim_durations": [1.2] * 9,
                "match_positions": [2, 5, 8],
                "isi_times": [0.3] * 9,
                "total_duration": 13.5,
            },
            {
                "n_back_level": 1,
                "num_stims": 6,
                "stim_list": ["3", "7", "7", "1", "6", "6"],
                "stim_durations": [1.2] * 6,
                "match_positions": [2, 5],
                "isi_times": [0.3] * 6,
                "total_duration": 9.0,
            },
        ],
        "2-back": [
            {
                "n_back_level": 2,
                "num_stims": 7,
                "stim_list": ["7", "3", "7", "5", "7", "1", "7"],
                "stim_durations": [1.2] * 7,
                "match_positions": [2, 4, 6],
                "isi_times": [0.3] * 7,
                "total_duration": 10.5,
            },
            {
                "n_back_level": 2,
                "num_stims": 10,
                "stim_list": ["9", "2", "9", "6", "9", "8", "6", "8", "6", "8"],
                "stim_durations": [1.2] * 10,
                "match_positions": [2, 4, 6, 7, 8, 9],
                "isi_times": [0.3] * 10,
                "total_duration": 15.0,
            },
            {
                "n_back_level": 2,
                "num_stims": 8,
                "stim_list": ["4", "1", "4", "8", "4", "5", "8", "5"],
                "stim_durations": [1.2] * 8,
                "match_positions": [2, 4, 6, 7],
                "isi_times": [0.3] * 8,
                "total_duration": 12.0,
            },
        ],
    }

    SIMPLE_WELCOME_TITLE = """Welcome to the Study 
    "Multitasking and Time Perception" """

    # ------------------------------------------------------------------
    # Screen contents
    # ------------------------------------------------------------------

    STUDY_WELCOME_TITLE = "About the Study"
    STUDY_WELCOME_CONTENT = """Hi! I am Emmanuel. 

    This experiment is part of a research project for my Master's Thesis in Human-Computer Interaction at Utrecht University. 
    
    I'd like to extend a warm thank you for participating!

    All data gathered in this experiment will be fully anonymized. No personal data will be collected, beyond basic demographics (age, gender). 

    This study is carried out by Emmanuel Fragkiadakis (e.fragkiadakis@uu.nl) under the supervision of Dr. Eelco Herder (e.herder@uu.nl) at the Department of Information and Computing Sciences. 
    If you have any concerns or questions about this research, please contact me or Eelco. 

    Taking part in this study is voluntary and you may withdraw at any time without providing a reason.
    
    Note: in any form of emergency or problem, you can press the ESC key to force terminate the experiment. 
    Please note that this will Abort the experiment completely and there is no way to resume it later."""

    CONSENT_TITLE = "Consent Form"
    CONSENT_CONTENT = """For this study we are looking for adult participants up to and including 44 years old. 
    Participants should not be diagnosed with neither a form of attentional disorder nor a form of dyslexia. 

    By clicking "I consent", you affirm that: 
    
    - you have understood how your data is going to be processed and you have no objection. 
    - you understand that you can abort the experiment at any point, without giving explanation. 
    - you are not officially diagnosed with an attentional disorder or dyslexia."""

    WELCOME_TITLE = "Welcome to the Experiment"
    WELCOME_CONTENT = """This experiment will test your ability to perform multiple tasks.

    You will learn each task step by step, practice them individually,
    and then combine them in the main experiment.

    The entire session will take approximately 1 hour. 

    Try to be in a quiet and comfortable place throughout the duration of the experiment."""

    EXPERIMENT_OVERVIEW_TITLE = "Experiment Overview"
    EXPERIMENT_OVERVIEW_CONTENT = """The experiment has 2 phases:

    1. ORIENTATION PHASE
       Learn each task individually with practice trials

    2. MAIN EXPERIMENT 
       This is the main experiment with complete trials

    Let's start with the Orientation Phase."""

    ORIENTATION_INSTRUCTIONS_TITLE = "Orientation Phase"
    ORIENTATION_INSTRUCTIONS_CONTENT = """In this phase, you will learn 4 different tasks:

    1. TYPING TASK - Type words using the on-screen keyboard
    2. N-BACK TASK - Remember and identify repeated digits 
    3. INTERRUPTIONS - The above tasks with or without interruptions
    4. TIME ESTIMATION - Estimate how long tasks take

    Each task will be explained and practiced separately.

    Ready to begin?"""

    TYPING_REMINDER_MESSAGE = """REMINDER: The task gets submitted automatically when the length of the word is reached. 
    If you are lost, just keep typing what you remember."""

    TYPING_TASK_INTRO_TITLE = "Easy Typing Task"
    TYPING_TASK_INTRO_CONTENT = """You will see a target word and type it using the on-screen keyboard.

    Click letters with your mouse to type. When you complete the word, it automatically submits.

    Example: If the word is FOX, when you type 3 letters, your answer is automatically submitted.
    If you mistype a letter, just continue with the rest of the word.

    Let's practice with a few words!"""

    INVISIBLE_TYPING_INTRO_TITLE = "Hard Typing Task"
    INVISIBLE_TYPING_INTRO_CONTENT = """Now you will practice the same typing task, but with a twist:

    The letters you type will be INVISIBLE - you won't see them appear on screen.

    This makes the task more challenging as you must rely on memory to track your progress.

    Click letters with your mouse to type. When you complete the word, it automatically submits.

    Example: If the word is FOX, when you type 3 letters, your answer is automatically submitted.
    If you mistype a letter, just continue with the rest of the word.

    Let's practice with a few words!"""

    TYPING_TASK_OVERVIEW_TITLE = "1. TYPING TASK"
    TYPING_TASK_OVERVIEW_CONTENT = """You will see a target word and type it using the on-screen keyboard.

    There are two versions of this task:
    - EASY VERSION: Letters appear as you type them (visible typing)
    - HARD VERSION: Letters do not appear as you type (invisible typing)

    In the main experiment, only the HARD VERSION will be used.

    You will practice both versions to understand how each works."""

    NBACK_TASK_OVERVIEW_TITLE = "2. N-BACK TASK"
    NBACK_TASK_OVERVIEW_CONTENT = """Digits will appear one by one. 
    Press SPACE when the current digit matches the digit shown N positions back.

    There are two versions of this task:
    - 1-BACK: Match the digit that appeared 1 position ago
    - 2-BACK: Match the digit that appeared 2 positions ago

    You will practice both versions to understand how each works."""

    INTERRUPTION_TRAINING_INTRO_TITLE = "3. INTERRUPTIONS"
    INTERRUPTION_TRAINING_INTRO_CONTENT = """Now you'll learn how the tasks work together.

    Sometimes while typing, you'll be INTERRUPTED by an N-back task.
    After completing the N-back, you'll return to finish typing.

    Other times, you'll complete typing first, THEN do the N-back task.

    Let's start by understanding how interruptions work and how they feel."""

    INTERRUPTION_UNDERSTANDING_TITLE = "Understanding Interruptions"
    INTERRUPTION_UNDERSTANDING_CONTENT = """In the next 2 trials, you'll experience INTERRUPTIONS.

    While typing, the screen will suddenly switch to an N-back task.
    Complete the N-back, then you'll return to finish typing where you left off.

    Try to be focused and remember where you left off."""

    INTERRUPTION_PRACTICE_TITLE = "Interruption Practice"
    INTERRUPTION_PRACTICE_CONTENT = """Good! Now let's practice with 4 trials.

    You'll experience either:
    - INTERRUPTED: Typing gets interrupted by N-back
    - SEQUENTIAL: Complete typing first, then N-back

    This matches what you'll see in the main experiment."""

    TIME_ESTIMATION_INTRO_TITLE = "4. TIME ESTIMATION"
    TIME_ESTIMATION_INTRO_CONTENT = """Final training phase: estimating how long tasks take.

    After completing tasks, you'll estimate how many seconds passed.
    This might seem challenging at first, but you will soon start feeling that you are improving!

    We'll practice with:
    1. Typing task + time estimation
    2. N-back task + time estimation  
    3. Both tasks + time estimation"""

    TIME_PHASE1_TITLE = "Time Estimation: Typing Task"
    TIME_PHASE1_CONTENT = """Let's start simple.

    Complete the typing task, then estimate how long it took.
    Try to pay attention to the passage of time while typing. 

    ATTENTION: Do not use clocks or explicitly count!

    2 practice trials coming up..."""
    TIME_PHASE2_TITLE = "Time Estimation: N-Back Task"
    TIME_PHASE2_CONTENT = """Now with the N-back task.

    Complete the N-back task, then estimate how long it took.
    Now that you know the task, there is no longer live feedback for the N-back task. 
    You will only see your score at the end of the trial.

    Try to pay attention to the passage of time while performing the task.

    ATTENTION: Do not use clocks or explicitly count!

    2 practice trials coming up..."""

    TIME_PHASE3_TITLE = "Time Estimation: Combined Tasks"
    TIME_PHASE3_CONTENT = """Final practice: both tasks together.

    Complete typing AND N-back, then estimate the TOTAL time of the trial (both tasks).

    NOTE: You will not know when the N-back task will appear. It might interrupt your typing. 
    In that case, you will have to remember where you left off. 

    This is the most challenging part, but you got this - good luck!

    Try to pay attention to the passage of time while performing the trial.
    ATTENTION: Do not use clocks or explicitly count!
    
    4 practice trials coming up..."""

    MAIN_RUN_PRACTICE_TITLE = "Practice Trials"
    MAIN_RUN_PRACTICE_CONTENT = f"""You will now perform {NUM_PRACTICE_TRIALS} practice trials.
    You will receive feedback only on your timing performance at the end of each trial.

    This is just practice to understand the experiment trials. 
    Your performance won't be recorded."""

    MAIN_RUN_MAIN_TITLE = "Main Trials"
    MAIN_RUN_MAIN_CONTENT = f"""You will now perform the main trials in blocks.
    You will not receive any type of feedback.

    You will perform {NUM_BLOCKS} blocks of {NUM_TRIALS} trials.
    There will be short breaks between the blocks.

    IMPORTANT: This is the main experiment. 
    This is the moment to be very focused and perform your best on all tasks!
    Your performance will be recorded.

    Take some moments to get ready. 
    Click on the button when you are ready to begin the main phase.

    You got this - good luck!"""

    PRACTICE_BLOCK_TITLE = "Practice Block"
    PRACTICE_BLOCK_CONTENT = """This is the practice block.

    Press Start when ready."""

    MAIN_BLOCK_TITLE = "Block {} of {}"

    MAIN_BLOCK_CONTENT_FIRST = """This is block {} out of {}.

    {} remaining.

    When you feel ready, press the Start button.
    Note: Your responses are recorded."""

    MAIN_BLOCK_CONTENT = """This is block {} out of {}.

    {} remaining.

    Please take some time to rest.

    The button will be activated in 30 seconds.

    IMPORTANT: Do not close or switch to a different screen.

    When you feel ready, press the Start button. 
    Note: Your responses are recorded."""

    BLOCK_START_BUTTON = "Start!"

    DECISION_SCREEN_CONTENT = """Good job with completing the Orientation (Training) Phase!
    Now you know exactly how the tasks of the experiment will look like and what to expect.
    For the Main Experiment you will have to perform blocks of the trials you have already learned 
    with small breaks in between. The duration until the end is around 30 minutes.
    
    There is no new learning to happen!
    
    COMPENSATION INFORMATION:
    - Base payment: As advertized (for completing the orientation) - you will receive this even if you abort now
    - Bonus payment: $14 (for completing the main phase) - you will have to go until the end
    
    Please note: Participants displaying very low or inconsistent performance or potential cheating will not be eligible for the bonus.
    We encourage you to perform at your best on all tasks.
    
    Please continue only if you plan to be focused and provide quality data.
    
    Try to be as focused as possible for the next 30 minutes."""

    THANK_YOU_TITLE = "Thank You!"
    THANK_YOU_CONTENT = """You have completed the experiment.

    Thank you for your participation!

    Your data has been saved. 



    When the experiment exits, you can safely close all windows and the application.

    ATTENTION: you must go back to the survey to upload your data and submit the form (even if you did not complete the experiment)"""

    # ----------------------------------------------------------------------

    # --- Initialize components for Routine "demographics_gender" ---
    dummy_demographics_gender = visual.TextStim(
        win=win,
        name="dummy_demographics_gender",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "demographics_age" ---
    dummy_demographics_age = visual.TextStim(
        win=win,
        name="dummy_demographics_age",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "demographics_confirmation" ---
    dummy_demographics_confirmation = visual.TextStim(
        win=win,
        name="dummy_demographics_confirmation",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "check_keyboard" ---
    dummy_9 = visual.TextStim(
        win=win,
        name="dummy_9",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=-1.0,
    )

    # --- Initialize components for Routine "load_experiment_data" ---
    dummy_4 = visual.TextStim(
        win=win,
        name="dummy_4",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "block_instructions" ---
    block_text = visual.TextStim(
        win=win,
        name="block_text",
        text="",
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="black",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "trial_instructions" ---
    trial_text = visual.TextStim(
        win=win,
        name="trial_text",
        text="",
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="black",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "primary_task" ---
    dummy_3 = visual.TextStim(
        win=win,
        name="dummy_3",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=-1.0,
    )

    # --- Initialize components for Routine "simple_welcome_screen" ---
    dummy_simple_welcome = visual.TextStim(
        win=win,
        name="dummy_simple_welcome",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "study_welcome_screen" ---
    dummy_study_welcome = visual.TextStim(
        win=win,
        name="dummy_study_welcome",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.03,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "consent_screen" ---
    dummy_consent = visual.TextStim(
        win=win,
        name="dummy_consent",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.04,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "nback_task_instructions" ---
    nback_instructions_component = visual.TextStim(
        win=win,
        name="nback_instructions_component",
        text="",
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.15 / 2,
        wrapWidth=None,
        ori=0.0,
        color="black",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        bold=True,
        depth=0.0,
    )

    # --- Initialize components for Routine "interrupting_task" ---
    dummy_2 = visual.TextStim(
        win=win,
        name="dummy_2",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=-1.0,
    )

    # --- Initialize components for Routine "time_estimate" ---
    dummy_5 = visual.TextStim(
        win=win,
        name="dummy_5",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=-1.0,
    )

    # --- Initialize components for Routine "welcome_screen" ---
    dummy_welcome = visual.TextStim(
        win=win,
        name="dummy_welcome",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "experiment_overview" ---
    dummy_overview = visual.TextStim(
        win=win,
        name="dummy_overview",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "orientation_instructions" ---
    dummy_orientation = visual.TextStim(
        win=win,
        name="dummy_orientation",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "tasks_overview" ---
    dummy_tasks = visual.TextStim(
        win=win,
        name="dummy_tasks",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "orientation_typing_trial" ---
    dummy_typing_trial = visual.TextStim(
        win=win,
        name="dummy_typing_trial",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "typing_introduction" ---
    dummy_typing_intro = visual.TextStim(
        win=win,
        name="dummy_typing_intro",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "typing_feedback" ---
    dummy_feedback = visual.TextStim(
        win=win,
        name="dummy_feedback",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "nback_instructions" ---
    dummy_nback_instructions = visual.TextStim(
        win=win,
        name="dummy_nback_instructions",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "nback_trial" ---
    dummy_nback_trial = visual.TextStim(
        win=win,
        name="dummy_nback_trial",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "nback_feedback" ---
    dummy_nback_feedback = visual.TextStim(
        win=win,
        name="dummy_nback_feedback",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "trial_countdown" ---
    fixation_cross = visual.ShapeStim(
        win=win,
        name="fixation_cross",
        vertices="cross",
        size=(0.08, 0.08),
        ori=0.0,
        pos=(0, 0),
        anchor="center",
        lineWidth=0.6,
        colorSpace="rgb",
        lineColor="black",
        fillColor="black",
        opacity=None,
        depth=0.0,
        interpolate=True,
    )

    countdown_text = visual.TextStim(
        win=win,
        name="countdown_text",
        text="",
        font="Arial",
        pos=(0, -0.2),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="black",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=-1.0,
    )

    # --- Initialize components for Routine "main_run_phase" ---
    dummy_main_run = visual.TextStim(
        win=win,
        name="dummy_main_run",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "main_phase_intro" ---
    dummy_main_phase_intro = visual.TextStim(
        win=win,
        name="dummy_main_phase_intro",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "thank_you" ---
    dummy_thank_you = visual.TextStim(
        win=win,
        name="dummy_thank_you",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # --- Initialize components for Routine "main_phase_decision" ---
    dummy_main_decision = visual.TextStim(
        win=win,
        name="dummy_main_decision",
        text=None,
        font="Arial",
        pos=(0, 0),
        draggable=False,
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=0.0,
    )

    # Define global clock
    if globalClock is None:
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        if globalClock == "float":
            globalClock = core.Clock(format="float")
        elif globalClock == "iso":
            globalClock = core.Clock(format="%Y-%m-%d_%H:%M:%S.%f%z")
        else:
            globalClock = core.Clock(format=globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()
    win.flip()
    expInfo["expStart"] = data.getDateStr(
        format="%Y-%m-%d %Hh%M.%S.%f %z", fractionalSecondDigits=6
    )

    # --- Prepare to start Routine "global_definitions" ---
    global_definitions = data.Routine(
        name="global_definitions",
        components=[dummy],
    )
    global_definitions.status = NOT_STARTED
    continueRoutine = True

    global mouse

    if "mouse" not in globals():
        mouse = event.Mouse(win=win)
        win.mouseVisible = True

    global_definitions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    global_definitions.tStart = globalClock.getTime(format="float")
    global_definitions.status = STARTED
    thisExp.addData("global_definitions.started", global_definitions.tStart)
    global_definitions.maxDuration = None
    global_definitionsComponents = global_definitions.components
    for thisComponent in global_definitions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, "status"):
            thisComponent.status = NOT_STARTED
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "global_definitions" ---
    global_definitions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1

        # *dummy* updates
        if dummy.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            dummy.frameNStart = frameN
            dummy.tStart = t
            dummy.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(dummy, "tStartRefresh")
            thisExp.timestampOnFlip(win, "dummy.started")
            dummy.status = STARTED
            dummy.setAutoDraw(True)

        if dummy.status == STARTED:
            pass

        if dummy.status == STARTED:
            if tThisFlipGlobal > dummy.tStartRefresh + 1 - frameTolerance:
                dummy.tStop = t
                dummy.tStopRefresh = tThisFlipGlobal
                dummy.frameNStop = frameN
                thisExp.timestampOnFlip(win, "dummy.stopped")
                dummy.status = FINISHED
                dummy.setAutoDraw(False)

        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, win=win, timers=[routineTimer], playbackComponents=[]
            )
            continue

        if not continueRoutine:
            global_definitions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False
        for thisComponent in global_definitions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break

        if continueRoutine:
            win.flip()

    # --- Ending Routine "global_definitions" ---
    for thisComponent in global_definitions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    global_definitions.tStop = globalClock.getTime(format="float")
    global_definitions.tStopRefresh = tThisFlipGlobal
    thisExp.addData("global_definitions.stopped", global_definitions.tStop)
    if global_definitions.maxDurationReached:
        routineTimer.addTime(-global_definitions.maxDuration)
    elif global_definitions.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()

    # -------------------------------------------------- START EXPERIMENT INTRO --------------------------------------------------

    # ---------------- ENTRY SCREEN -------------
    # --- Prepare to start Routine "simple_welcome_screen" ---
    simple_welcome_screen = data.Routine(
        name="simple_welcome_screen",
        components=[dummy_simple_welcome],
    )
    simple_welcome_screen.status = NOT_STARTED
    continueRoutine = True

    welcome_title = visual.TextStim(
        win=win,
        text=SIMPLE_WELCOME_TITLE,
        pos=(0, 0.2),
        height=0.06,
        color="black",
        font="Arial",
        bold=True,
        wrapWidth=1.4,
    )

    try:
        uu_logo = visual.ImageStim(
            win=win,
            image="resources/uni_logo.png",
            pos=(0.1, -0.1),
            size=(None, 0.4),
        )
        logo_loaded = True
    except:
        logo_loaded = False
        print("Warning: Could not load logo.png")

    start_button = visual.Rect(
        win=win,
        pos=(0, -0.4),
        width=0.2,
        height=0.08,
        fillColor="lightgrey",
        lineColor="black",
    )

    start_button_text = visual.TextStim(
        win=win, text="Start", pos=(0, -0.4), height=0.04, color="black", font="Arial"
    )

    welcome_title.autoDraw = True
    if logo_loaded:
        uu_logo.autoDraw = True
    start_button.autoDraw = True
    start_button_text.autoDraw = True

    simple_welcome_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    simple_welcome_screen.tStart = globalClock.getTime(format="float")
    simple_welcome_screen.status = STARTED
    thisExp.addData("simple_welcome_screen.started", simple_welcome_screen.tStart)
    simple_welcome_screen.maxDuration = None

    simple_welcome_screenComponents = simple_welcome_screen.components
    for thisComponent in simple_welcome_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, "status"):
            thisComponent.status = NOT_STARTED

    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "simple_welcome_screen" ---
    simple_welcome_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1

        keys = defaultKeyboard.getKeys(keyList=["escape"])
        if keys:
            thisExp.status = FINISHED
            endExperiment(thisExp, win=win)
            return

        if mouse.getPressed()[0]:
            mx, my = mouse.getPos()
            if (
                start_button.pos[0] - start_button.width / 2
                <= mx
                <= start_button.pos[0] + start_button.width / 2
                and start_button.pos[1] - start_button.height / 2
                <= my
                <= start_button.pos[1] + start_button.height / 2
            ):
                continueRoutine = False

        if (
            dummy_simple_welcome.status == NOT_STARTED
            and tThisFlip >= 0.0 - frameTolerance
        ):
            dummy_simple_welcome.frameNStart = frameN
            dummy_simple_welcome.tStart = t
            dummy_simple_welcome.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(dummy_simple_welcome, "tStartRefresh")
            thisExp.timestampOnFlip(win, "dummy_simple_welcome.started")
            dummy_simple_welcome.status = STARTED
            dummy_simple_welcome.setAutoDraw(True)

        if dummy_simple_welcome.status == STARTED:
            pass

        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return

        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, win=win, timers=[routineTimer], playbackComponents=[]
            )
            continue

        if not continueRoutine:
            simple_welcome_screen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False
        for thisComponent in simple_welcome_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break

        if continueRoutine:
            win.flip()

    # --- Ending Routine "simple_welcome_screen" ---
    for thisComponent in simple_welcome_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)

    welcome_title.autoDraw = False
    if logo_loaded:
        uu_logo.autoDraw = False
    start_button.autoDraw = False
    start_button_text.autoDraw = False

    simple_welcome_screen.tStop = globalClock.getTime(format="float")
    simple_welcome_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData("simple_welcome_screen.stopped", simple_welcome_screen.tStop)
    routineTimer.reset()
    thisExp.nextEntry()

    # ----------------- CONSENT and DEMOGRAPHICS --------------------

    # --- Prepare to start Routine "study_welcome_screen" ---
    study_welcome_screen = data.Routine(
        name="study_welcome_screen",
        components=[dummy_study_welcome],
    )
    study_welcome_screen.status = NOT_STARTED
    continueRoutine = True

    studyWelcomeScreen = InstructionScreen(
        win=win,
        title=STUDY_WELCOME_TITLE,
        content=STUDY_WELCOME_CONTENT,
        letter_height=0.025,
    )
    studyWelcomeScreen.show()

    study_welcome_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    study_welcome_screen.tStart = globalClock.getTime(format="float")
    study_welcome_screen.status = STARTED
    thisExp.addData("study_welcome_screen.started", study_welcome_screen.tStart)
    study_welcome_screen.maxDuration = None

    study_welcome_screenComponents = study_welcome_screen.components
    for thisComponent in study_welcome_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, "status"):
            thisComponent.status = NOT_STARTED

    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "study_welcome_screen" ---
    study_welcome_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1

        studyWelcomeScreen.draw()
        nav_result = studyWelcomeScreen.check_navigation(mouse, defaultKeyboard)
        if nav_result == "escape":
            thisExp.status = FINISHED
            endExperiment(thisExp, win=win)
            return
        elif nav_result:
            continueRoutine = False

        if (
            dummy_study_welcome.status == NOT_STARTED
            and tThisFlip >= 0.0 - frameTolerance
        ):
            dummy_study_welcome.frameNStart = frameN
            dummy_study_welcome.tStart = t
            dummy_study_welcome.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(dummy_study_welcome, "tStartRefresh")
            thisExp.timestampOnFlip(win, "dummy_study_welcome.started")
            dummy_study_welcome.status = STARTED
            dummy_study_welcome.setAutoDraw(True)

        if dummy_study_welcome.status == STARTED:
            pass

        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return

        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, win=win, timers=[routineTimer], playbackComponents=[]
            )
            continue

        if not continueRoutine:
            study_welcome_screen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False
        for thisComponent in study_welcome_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break

        if continueRoutine:
            win.flip()

    # --- Ending Routine "study_welcome_screen" ---
    for thisComponent in study_welcome_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    studyWelcomeScreen.hide()

    study_welcome_screen.tStop = globalClock.getTime(format="float")
    study_welcome_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData("study_welcome_screen.stopped", study_welcome_screen.tStop)
    routineTimer.reset()
    thisExp.nextEntry()

    # --------------- Consent --------------------

    # --- Prepare to start Routine "consent_screen" ---
    consent_screen = data.Routine(
        name="consent_screen",
        components=[dummy_consent],
    )
    consent_screen.status = NOT_STARTED
    continueRoutine = True

    consentScreen = InstructionScreen(
        win=win,
        title=CONSENT_TITLE,
        content=CONSENT_CONTENT,
        button_text="I consent",
        button_fill_color="lightgreen",
        letter_height=0.025,
    )
    consentScreen.show()

    consent_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    consent_screen.tStart = globalClock.getTime(format="float")
    consent_screen.status = STARTED
    thisExp.addData("consent_screen.started", consent_screen.tStart)
    consent_screen.maxDuration = None

    consent_screenComponents = consent_screen.components
    for thisComponent in consent_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, "status"):
            thisComponent.status = NOT_STARTED

    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "consent_screen" ---
    consent_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1

        consentScreen.draw()
        nav_result = consentScreen.check_navigation(mouse, defaultKeyboard)
        if nav_result == "escape":
            thisExp.status = FINISHED
            endExperiment(thisExp, win=win)
            return
        elif nav_result:
            continueRoutine = False

        if dummy_consent.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            dummy_consent.frameNStart = frameN
            dummy_consent.tStart = t
            dummy_consent.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(dummy_consent, "tStartRefresh")
            thisExp.timestampOnFlip(win, "dummy_consent.started")
            dummy_consent.status = STARTED
            dummy_consent.setAutoDraw(True)

        if dummy_consent.status == STARTED:
            pass

        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return

        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, win=win, timers=[routineTimer], playbackComponents=[]
            )
            continue

        if not continueRoutine:
            consent_screen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False
        for thisComponent in consent_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break

        if continueRoutine:
            win.flip()

    # --- Ending Routine "consent_screen" ---
    for thisComponent in consent_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    consentScreen.hide()

    consent_screen.tStop = globalClock.getTime(format="float")
    consent_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData("consent_screen.stopped", consent_screen.tStop)
    routineTimer.reset()
    thisExp.nextEntry()

    # --------------- Demographics --------------------

    # --- Prepare to start Routine "demographics_gender" ---
    demographics_gender = data.Routine(
        name="demographics_gender",
        components=[dummy_demographics_gender],
    )
    demographics_gender.status = NOT_STARTED
    continueRoutine = True

    genderScreen = RadioButtonScreen(
        win=win,
        title="Demographics: Gender",
        question="Please select your gender:",
        options=["Male", "Female", "Non-binary / third gender", "Prefer not to say"],
        button_text="Continue",
    )
    genderScreen.show()

    if "demographics_data" not in globals():
        demographics_data = {}

    demographics_gender.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demographics_gender.tStart = globalClock.getTime(format="float")
    demographics_gender.status = STARTED
    thisExp.addData("demographics_gender.started", demographics_gender.tStart)
    demographics_gender.maxDuration = None

    demographics_genderComponents = demographics_gender.components
    for thisComponent in demographics_gender.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, "status"):
            thisComponent.status = NOT_STARTED

    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "demographics_gender" ---
    demographics_gender.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1

        genderScreen.draw()
        result = genderScreen.check_selection(mouse)
        if result == "continue":
            demographics_data["gender"] = genderScreen.get_selection()
            continueRoutine = False

        if (
            dummy_demographics_gender.status == NOT_STARTED
            and tThisFlip >= 0.0 - frameTolerance
        ):
            dummy_demographics_gender.frameNStart = frameN
            dummy_demographics_gender.tStart = t
            dummy_demographics_gender.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(dummy_demographics_gender, "tStartRefresh")
            thisExp.timestampOnFlip(win, "dummy_demographics_gender.started")
            dummy_demographics_gender.status = STARTED
            dummy_demographics_gender.setAutoDraw(True)

        if dummy_demographics_gender.status == STARTED:
            pass

        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return

        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, win=win, timers=[routineTimer], playbackComponents=[]
            )
            continue

        if not continueRoutine:
            demographics_gender.forceEnded = routineForceEnded = True
            break
        continueRoutine = False
        for thisComponent in demographics_gender.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break

        if continueRoutine:
            win.flip()

    # --- Ending Routine "demographics_gender" ---
    for thisComponent in demographics_gender.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    genderScreen.hide()

    demographics_gender.tStop = globalClock.getTime(format="float")
    demographics_gender.tStopRefresh = tThisFlipGlobal
    thisExp.addData("demographics_gender.stopped", demographics_gender.tStop)
    routineTimer.reset()
    thisExp.nextEntry()

    # --- Prepare to start Routine "demographics_age" ---
    demographics_age = data.Routine(
        name="demographics_age",
        components=[dummy_demographics_age],
    )
    demographics_age.status = NOT_STARTED
    continueRoutine = True

    ageScreen = AgeInputScreen(win=win, mouse=mouse)
    ageScreen.show()

    demographics_age.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demographics_age.tStart = globalClock.getTime(format="float")
    demographics_age.status = STARTED
    thisExp.addData("demographics_age.started", demographics_age.tStart)
    demographics_age.maxDuration = None

    demographics_ageComponents = demographics_age.components
    for thisComponent in demographics_age.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, "status"):
            thisComponent.status = NOT_STARTED

    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "demographics_age" ---
    demographics_age.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1

        ageScreen.draw()
        result = ageScreen.check_input()
        if result == "continue":
            demographics_data["age"] = ageScreen.get_age()
            continueRoutine = False

        if (
            dummy_demographics_age.status == NOT_STARTED
            and tThisFlip >= 0.0 - frameTolerance
        ):
            dummy_demographics_age.frameNStart = frameN
            dummy_demographics_age.tStart = t
            dummy_demographics_age.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(dummy_demographics_age, "tStartRefresh")
            thisExp.timestampOnFlip(win, "dummy_demographics_age.started")
            dummy_demographics_age.status = STARTED
            dummy_demographics_age.setAutoDraw(True)

        if dummy_demographics_age.status == STARTED:
            pass

        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return

        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, win=win, timers=[routineTimer], playbackComponents=[]
            )
            continue

        if not continueRoutine:
            demographics_age.forceEnded = routineForceEnded = True
            break
        continueRoutine = False
        for thisComponent in demographics_age.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break

        if continueRoutine:
            win.flip()

    # --- Ending Routine "demographics_age" ---
    for thisComponent in demographics_age.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    ageScreen.hide()

    demographics_age.tStop = globalClock.getTime(format="float")
    demographics_age.tStopRefresh = tThisFlipGlobal
    thisExp.addData("demographics_age.stopped", demographics_age.tStop)
    routineTimer.reset()
    thisExp.nextEntry()

    demographicsLoop = data.TrialHandler2(
        name="demographicsLoop",
        nReps=999,
        method="sequential",
        extraInfo=expInfo,
        originPath=-1,
        trialList=[None],
        seed=None,
    )
    thisExp.addLoop(demographicsLoop)
    thisDemographicsLoop = demographicsLoop.trialList[0]

    if thisDemographicsLoop != None:
        for paramName in thisDemographicsLoop:
            globals()[paramName] = thisDemographicsLoop[paramName]

    for thisDemographicsLoop in demographicsLoop:
        currentLoop = demographicsLoop
        thisExp.timestampOnFlip(win, "thisRow.t")

        if thisDemographicsLoop != None:
            for paramName in thisDemographicsLoop:
                globals()[paramName] = thisDemographicsLoop[paramName]

        # --- Prepare to start Routine "demographics_confirmation" ---
        demographics_confirmation = data.Routine(
            name="demographics_confirmation",
            components=[dummy_demographics_confirmation],
        )
        demographics_confirmation.status = NOT_STARTED
        continueRoutine = True

        confirmationScreen = DemographicsConfirmationScreen(win=win)
        confirmationScreen.set_demographics(
            demographics_data["gender"], demographics_data["age"]
        )
        confirmationScreen.show()

        demographics_confirmation.tStartRefresh = win.getFutureFlipTime(
            clock=globalClock
        )
        demographics_confirmation.tStart = globalClock.getTime(format="float")
        demographics_confirmation.status = STARTED
        thisExp.addData(
            "demographics_confirmation.started", demographics_confirmation.tStart
        )
        demographics_confirmation.maxDuration = None

        demographics_confirmationComponents = demographics_confirmation.components
        for thisComponent in demographics_confirmation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "demographics_confirmation" ---
        if (
            isinstance(demographicsLoop, data.TrialHandler2)
            and thisDemographicsLoop.thisN != demographicsLoop.thisTrial.thisN
        ):
            continueRoutine = False
        demographics_confirmation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            confirmationScreen.draw()

            result = confirmationScreen.check_navigation(mouse)
            if result == "again":
                continueRoutine = False
            elif result == "proceed":
                thisExp.addData("cm_gender", demographics_data["gender"])
                thisExp.addData("cm_age", demographics_data["age"])
                continueRoutine = False
                demographicsLoop.finished = True

            if (
                dummy_demographics_confirmation.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_demographics_confirmation.frameNStart = frameN
                dummy_demographics_confirmation.tStart = t
                dummy_demographics_confirmation.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_demographics_confirmation, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_demographics_confirmation.started")
                dummy_demographics_confirmation.status = STARTED
                dummy_demographics_confirmation.setAutoDraw(True)

            if dummy_demographics_confirmation.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                demographics_confirmation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in demographics_confirmation.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "demographics_confirmation" ---
        for thisComponent in demographics_confirmation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        confirmationScreen.hide()

        demographics_confirmation.tStop = globalClock.getTime(format="float")
        demographics_confirmation.tStopRefresh = tThisFlipGlobal
        thisExp.addData(
            "demographics_confirmation.stopped", demographics_confirmation.tStop
        )
        routineTimer.reset()
        thisExp.nextEntry()

        if result == "again":
            demographics_data = {}

            demographics_gender_retry = data.Routine(
                name="demographics_gender_retry",
                components=[dummy_demographics_gender],
            )
            demographics_gender_retry.status = NOT_STARTED
            continueRoutine = True

            genderScreen = RadioButtonScreen(
                win=win,
                title="Demographics: Gender",
                question="Please select your gender:",
                options=[
                    "Male",
                    "Female",
                    "Non-binary / third gender",
                    "Prefer not to say",
                ],
                button_text="Continue",
            )
            genderScreen.show()

            demographics_gender_retry.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            demographics_gender_retry.tStart = globalClock.getTime(format="float")
            demographics_gender_retry.status = STARTED
            thisExp.addData(
                "demographics_gender_retry.started", demographics_gender_retry.tStart
            )
            demographics_gender_retry.maxDuration = None

            demographics_gender_retryComponents = demographics_gender_retry.components
            for thisComponent in demographics_gender_retry.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            demographics_gender_retry.forceEnded = routineForceEnded = (
                not continueRoutine
            )
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                genderScreen.draw()

                result = genderScreen.check_selection(mouse)
                if result == "continue":
                    demographics_data["gender"] = genderScreen.get_selection()
                    continueRoutine = False

                if (
                    dummy_demographics_gender.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_demographics_gender.frameNStart = frameN
                    dummy_demographics_gender.tStart = t
                    dummy_demographics_gender.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_demographics_gender, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_demographics_gender.started")
                    dummy_demographics_gender.status = STARTED
                    dummy_demographics_gender.setAutoDraw(True)

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    demographics_gender_retry.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in demographics_gender_retry.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            for thisComponent in demographics_gender_retry.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            genderScreen.hide()
            demographics_gender_retry.tStop = globalClock.getTime(format="float")
            demographics_gender_retry.tStopRefresh = tThisFlipGlobal
            thisExp.addData(
                "demographics_gender_retry.stopped", demographics_gender_retry.tStop
            )
            routineTimer.reset()
            thisExp.nextEntry()

            demographics_age_retry = data.Routine(
                name="demographics_age_retry",
                components=[dummy_demographics_age],
            )
            demographics_age_retry.status = NOT_STARTED
            continueRoutine = True

            ageScreen = AgeInputScreen(win=win, mouse=mouse)
            ageScreen.show()

            demographics_age_retry.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            demographics_age_retry.tStart = globalClock.getTime(format="float")
            demographics_age_retry.status = STARTED
            thisExp.addData(
                "demographics_age_retry.started", demographics_age_retry.tStart
            )
            demographics_age_retry.maxDuration = None

            demographics_age_retryComponents = demographics_age_retry.components
            for thisComponent in demographics_age_retry.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            demographics_age_retry.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                ageScreen.draw()

                result = ageScreen.check_input()
                if result == "continue":
                    demographics_data["age"] = ageScreen.get_age()
                    continueRoutine = False

                if (
                    dummy_demographics_age.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_demographics_age.frameNStart = frameN
                    dummy_demographics_age.tStart = t
                    dummy_demographics_age.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_demographics_age, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_demographics_age.started")
                    dummy_demographics_age.status = STARTED
                    dummy_demographics_age.setAutoDraw(True)

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    demographics_age_retry.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in demographics_age_retry.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            for thisComponent in demographics_age_retry.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            ageScreen.hide()
            demographics_age_retry.tStop = globalClock.getTime(format="float")
            demographics_age_retry.tStopRefresh = tThisFlipGlobal
            thisExp.addData(
                "demographics_age_retry.stopped", demographics_age_retry.tStop
            )
            routineTimer.reset()
            thisExp.nextEntry()
        else:
            break

    # -------------------------------------------------- START EXPERIMENT --------------------------------------------------

    # for debugging purposes
    if not RUN_ORIENTATION:
        pass

    else:

        # --------------- Overview --------------------
        # --- Prepare to start Routine "welcome_screen" ---
        welcome_screen = data.Routine(
            name="welcome_screen",
            components=[dummy_welcome],
        )
        welcome_screen.status = NOT_STARTED
        continueRoutine = True

        welcomeScreen = InstructionScreen(
            win=win, title=WELCOME_TITLE, content=WELCOME_CONTENT
        )
        welcomeScreen.show()

        welcome_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        welcome_screen.tStart = globalClock.getTime(format="float")
        welcome_screen.status = STARTED
        thisExp.addData("welcome_screen.started", welcome_screen.tStart)
        welcome_screen.maxDuration = None

        welcome_screenComponents = welcome_screen.components
        for thisComponent in welcome_screen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "welcome_screen" ---
        welcome_screen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            welcomeScreen.draw()
            nav_result = welcomeScreen.check_navigation(mouse, defaultKeyboard)
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_welcome.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_welcome.frameNStart = frameN
                dummy_welcome.tStart = t
                dummy_welcome.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_welcome, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_welcome.started")
                dummy_welcome.status = STARTED
                dummy_welcome.setAutoDraw(True)

            if dummy_welcome.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                welcome_screen.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in welcome_screen.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "welcome_screen" ---
        for thisComponent in welcome_screen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        welcomeScreen.hide()

        welcome_screen.tStop = globalClock.getTime(format="float")
        welcome_screen.tStopRefresh = tThisFlipGlobal
        thisExp.addData("welcome_screen.stopped", welcome_screen.tStop)
        routineTimer.reset()
        thisExp.nextEntry()

        # --- Prepare to start Routine "experiment_overview" ---
        experiment_overview = data.Routine(
            name="experiment_overview",
            components=[dummy_overview],
        )
        experiment_overview.status = NOT_STARTED
        continueRoutine = True

        overviewScreen = InstructionScreen(
            win=win,
            title=EXPERIMENT_OVERVIEW_TITLE,
            content=EXPERIMENT_OVERVIEW_CONTENT,
        )
        overviewScreen.show()

        experiment_overview.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        experiment_overview.tStart = globalClock.getTime(format="float")
        experiment_overview.status = STARTED
        thisExp.addData("experiment_overview.started", experiment_overview.tStart)
        experiment_overview.maxDuration = None

        experiment_overviewComponents = experiment_overview.components
        for thisComponent in experiment_overview.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "experiment_overview" ---
        experiment_overview.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            overviewScreen.draw()
            nav_result = overviewScreen.check_navigation(mouse, defaultKeyboard)
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_overview.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_overview.frameNStart = frameN
                dummy_overview.tStart = t
                dummy_overview.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_overview, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_overview.started")
                dummy_overview.status = STARTED
                dummy_overview.setAutoDraw(True)

            if dummy_overview.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                experiment_overview.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in experiment_overview.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "experiment_overview" ---
        for thisComponent in experiment_overview.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        overviewScreen.hide()

        experiment_overview.tStop = globalClock.getTime(format="float")
        experiment_overview.tStopRefresh = tThisFlipGlobal
        thisExp.addData("experiment_overview.stopped", experiment_overview.tStop)
        routineTimer.reset()
        thisExp.nextEntry()

        # --- Prepare to start Routine "orientation_instructions" ---
        orientation_instructions = data.Routine(
            name="orientation_instructions",
            components=[dummy_orientation],
        )
        orientation_instructions.status = NOT_STARTED
        continueRoutine = True

        orientationScreen = InstructionScreen(
            win=win,
            title=ORIENTATION_INSTRUCTIONS_TITLE,
            content=ORIENTATION_INSTRUCTIONS_CONTENT,
            button_text="Let's go!",
        )
        orientationScreen.show()

        orientation_instructions.tStartRefresh = win.getFutureFlipTime(
            clock=globalClock
        )
        orientation_instructions.tStart = globalClock.getTime(format="float")
        orientation_instructions.status = STARTED
        thisExp.addData(
            "orientation_instructions.started", orientation_instructions.tStart
        )
        orientation_instructions.maxDuration = None

        orientation_instructionsComponents = orientation_instructions.components
        for thisComponent in orientation_instructions.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "orientation_instructions" ---
        orientation_instructions.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            orientationScreen.draw()
            nav_result = orientationScreen.check_navigation(mouse, defaultKeyboard)
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_orientation.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_orientation.frameNStart = frameN
                dummy_orientation.tStart = t
                dummy_orientation.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_orientation, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_orientation.started")
                dummy_orientation.status = STARTED
                dummy_orientation.setAutoDraw(True)

            if dummy_orientation.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                orientation_instructions.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in orientation_instructions.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "orientation_instructions" ---
        for thisComponent in orientation_instructions.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        orientationScreen.hide()

        orientation_instructions.tStop = globalClock.getTime(format="float")
        orientation_instructions.tStopRefresh = tThisFlipGlobal
        thisExp.addData(
            "orientation_instructions.stopped", orientation_instructions.tStop
        )
        routineTimer.reset()
        thisExp.nextEntry()

        # -------------------------------------------------- START ORIENTATION --------------------------------------------------

        # -------------------------------------------------- ORIENTATION: TYPING --------------------------------------------------

        # --- Prepare to start Routine "typing_task_overview" ---
        typing_task_overview = data.Routine(
            name="typing_task_overview",
            components=[dummy_typing_intro],
        )
        typing_task_overview.status = NOT_STARTED
        continueRoutine = True

        typingOverviewScreen = InstructionScreen(
            win=win,
            title=TYPING_TASK_OVERVIEW_TITLE,
            content=TYPING_TASK_OVERVIEW_CONTENT,
            button_text="Begin",
        )
        typingOverviewScreen.show()

        typing_task_overview.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        typing_task_overview.tStart = globalClock.getTime(format="float")
        typing_task_overview.status = STARTED
        thisExp.addData("typing_task_overview.started", typing_task_overview.tStart)
        typing_task_overview.maxDuration = None

        typing_task_overviewComponents = typing_task_overview.components
        for thisComponent in typing_task_overview.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "typing_task_overview" ---
        typing_task_overview.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            typingOverviewScreen.draw()
            nav_result = typingOverviewScreen.check_navigation(mouse, defaultKeyboard)
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_typing_intro.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_typing_intro.frameNStart = frameN
                dummy_typing_intro.tStart = t
                dummy_typing_intro.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_typing_intro, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_typing_intro.started")
                dummy_typing_intro.status = STARTED
                dummy_typing_intro.setAutoDraw(True)

            if dummy_typing_intro.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                typing_task_overview.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in typing_task_overview.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "typing_task_overview" ---
        for thisComponent in typing_task_overview.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        typingOverviewScreen.hide()

        typing_task_overview.tStop = globalClock.getTime(format="float")
        typing_task_overview.tStopRefresh = tThisFlipGlobal
        thisExp.addData("typing_task_overview.stopped", typing_task_overview.tStop)
        routineTimer.reset()
        thisExp.nextEntry()

        # --- Prepare to start Routine "typing_introduction" ---
        typing_introduction = data.Routine(
            name="typing_introduction",
            components=[dummy_typing_intro],
        )
        typing_introduction.status = NOT_STARTED
        continueRoutine = True

        typingIntroScreen = InstructionScreen(
            win=win,
            title=TYPING_TASK_INTRO_TITLE,
            content=TYPING_TASK_INTRO_CONTENT,
            button_text="Start Practice",
            button_width=0.3,
        )

        typingIntroScreen.show()

        typing_introduction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        typing_introduction.tStart = globalClock.getTime(format="float")
        typing_introduction.status = STARTED
        thisExp.addData("typing_introduction.started", typing_introduction.tStart)
        typing_introduction.maxDuration = None

        typing_introductionComponents = typing_introduction.components
        for thisComponent in typing_introduction.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "typing_introduction" ---
        typing_introduction.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            typingIntroScreen.draw()
            nav_result = typingIntroScreen.check_navigation(mouse, defaultKeyboard)
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_typing_intro.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_typing_intro.frameNStart = frameN
                dummy_typing_intro.tStart = t
                dummy_typing_intro.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_typing_intro, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_typing_intro.started")
                dummy_typing_intro.status = STARTED
                dummy_typing_intro.setAutoDraw(True)

            if dummy_typing_intro.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                typing_introduction.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in typing_introduction.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "typing_introduction" ---
        for thisComponent in typing_introduction.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        typingIntroScreen.hide()

        typing_introduction.tStop = globalClock.getTime(format="float")
        typing_introduction.tStopRefresh = tThisFlipGlobal
        thisExp.addData("typing_introduction.stopped", typing_introduction.tStop)
        routineTimer.reset()
        thisExp.nextEntry()

        orientationTypingLoop = data.TrialHandler2(
            name="orientationTypingLoop",
            nReps=3,
            method="sequential",
            extraInfo=expInfo,
            originPath=-1,
            trialList=[None],
            seed=None,
        )
        thisExp.addLoop(orientationTypingLoop)
        thisOrientationTypingLoop = orientationTypingLoop.trialList[0]

        if thisOrientationTypingLoop != None:
            for paramName in thisOrientationTypingLoop:
                globals()[paramName] = thisOrientationTypingLoop[paramName]

        for thisOrientationTypingLoop in orientationTypingLoop:
            currentLoop = orientationTypingLoop
            thisExp.timestampOnFlip(win, "thisRow.t")

            if thisOrientationTypingLoop != None:
                for paramName in thisOrientationTypingLoop:
                    globals()[paramName] = thisOrientationTypingLoop[paramName]

            # --- Prepare to start Routine "orientation_typing_trial" ---
            orientation_typing_trial = data.Routine(
                name="orientation_typing_trial",
                components=[dummy_typing_trial],
            )
            orientation_typing_trial.status = NOT_STARTED
            continueRoutine = True

            current_word = ORIENTATION_TYPING_WORDS[orientationTypingLoop.thisN]

            orientation_keyboard = Keyboard(win=win)
            orientationTypingTask = PrimaryTask(
                win=win,
                keyboard=orientation_keyboard,
                mouse=mouse,
                mode="easy",
                ok_rect=ok_button_rect,
                ok_text=ok_button_text,
                show_reminder=True,
            )

            orientationTypingTask.set_target_word(current_word)
            orientationTypingTask.reset()
            orientationTypingTask.show()

            orientation_typing_trial.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            orientation_typing_trial.tStart = globalClock.getTime(format="float")
            orientation_typing_trial.status = STARTED
            thisExp.addData(
                "orientation_typing_trial.started", orientation_typing_trial.tStart
            )
            orientation_typing_trial.maxDuration = None

            orientation_typing_trialComponents = orientation_typing_trial.components
            for thisComponent in orientation_typing_trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "orientation_typing_trial" ---
            if (
                isinstance(orientationTypingLoop, data.TrialHandler2)
                and thisOrientationTypingLoop.thisN
                != orientationTypingLoop.thisTrial.thisN
            ):
                continueRoutine = False
            orientation_typing_trial.forceEnded = routineForceEnded = (
                not continueRoutine
            )
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                orientationTypingTask.draw()
                status = orientationTypingTask.check_input()
                if status == "submitted":
                    orientationTypingTask.hide()
                    continueRoutine = False

                if (
                    dummy_typing_trial.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_typing_trial.frameNStart = frameN
                    dummy_typing_trial.tStart = t
                    dummy_typing_trial.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_typing_trial, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_typing_trial.started")
                    dummy_typing_trial.status = STARTED
                    dummy_typing_trial.setAutoDraw(True)

                if dummy_typing_trial.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    orientation_typing_trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in orientation_typing_trial.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "orientation_typing_trial" ---
            for thisComponent in orientation_typing_trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            orientation_typing_trial.tStop = globalClock.getTime(format="float")
            orientation_typing_trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData(
                "orientation_typing_trial.stopped", orientation_typing_trial.tStop
            )
            routineTimer.reset()

            # --- Prepare to start Routine "typing_feedback" ---
            typing_feedback = data.Routine(
                name="typing_feedback",
                components=[dummy_feedback],
            )
            typing_feedback.status = NOT_STARTED
            continueRoutine = True

            feedbackScreen = TypingFeedbackScreen(win=win)
            feedbackScreen.set_feedback(
                current_word, orientationTypingTask.text_entry_screen.entered_text
            )
            feedbackScreen.show()

            typing_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            typing_feedback.tStart = globalClock.getTime(format="float")
            typing_feedback.status = STARTED
            thisExp.addData("typing_feedback.started", typing_feedback.tStart)
            typing_feedback.maxDuration = None

            typing_feedbackComponents = typing_feedback.components
            for thisComponent in typing_feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "typing_feedback" ---
            if (
                isinstance(orientationTypingLoop, data.TrialHandler2)
                and thisOrientationTypingLoop.thisN
                != orientationTypingLoop.thisTrial.thisN
            ):
                continueRoutine = False
            typing_feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                feedbackScreen.draw()
                nav_result = feedbackScreen.check_navigation(mouse, defaultKeyboard)
                if nav_result == "escape":
                    thisExp.status = FINISHED
                    endExperiment(thisExp, win=win)
                    return
                elif nav_result:
                    continueRoutine = False

                if (
                    dummy_feedback.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_feedback.frameNStart = frameN
                    dummy_feedback.tStart = t
                    dummy_feedback.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_feedback, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_feedback.started")
                    dummy_feedback.status = STARTED
                    dummy_feedback.setAutoDraw(True)

                if dummy_feedback.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    typing_feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in typing_feedback.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "typing_feedback" ---
            for thisComponent in typing_feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            feedbackScreen.hide()

            typing_feedback.tStop = globalClock.getTime(format="float")
            typing_feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData("typing_feedback.stopped", typing_feedback.tStop)
            routineTimer.reset()
            thisExp.nextEntry()

        # --- Prepare to start Routine "invisible_typing_introduction" ---
        invisible_typing_introduction = data.Routine(
            name="invisible_typing_introduction",
            components=[dummy_typing_intro],
        )
        invisible_typing_introduction.status = NOT_STARTED
        continueRoutine = True

        invisibleTypingIntroScreen = InstructionScreen(
            win=win,
            title=INVISIBLE_TYPING_INTRO_TITLE,
            content=INVISIBLE_TYPING_INTRO_CONTENT,
            button_text="Start Practice",
            button_width=0.3,
        )
        invisibleTypingIntroScreen.show()

        invisible_typing_introduction.tStartRefresh = win.getFutureFlipTime(
            clock=globalClock
        )
        invisible_typing_introduction.tStart = globalClock.getTime(format="float")
        invisible_typing_introduction.status = STARTED
        thisExp.addData(
            "invisible_typing_introduction.started",
            invisible_typing_introduction.tStart,
        )
        invisible_typing_introduction.maxDuration = None

        invisible_typing_introductionComponents = (
            invisible_typing_introduction.components
        )
        for thisComponent in invisible_typing_introduction.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "invisible_typing_introduction" ---
        invisible_typing_introduction.forceEnded = routineForceEnded = (
            not continueRoutine
        )
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            invisibleTypingIntroScreen.draw()
            nav_result = invisibleTypingIntroScreen.check_navigation(
                mouse, defaultKeyboard
            )
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_typing_intro.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_typing_intro.frameNStart = frameN
                dummy_typing_intro.tStart = t
                dummy_typing_intro.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_typing_intro, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_typing_intro.started")
                dummy_typing_intro.status = STARTED
                dummy_typing_intro.setAutoDraw(True)

            if dummy_typing_intro.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                invisible_typing_introduction.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in invisible_typing_introduction.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "invisible_typing_introduction" ---
        for thisComponent in invisible_typing_introduction.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        invisibleTypingIntroScreen.hide()

        invisible_typing_introduction.tStop = globalClock.getTime(format="float")
        invisible_typing_introduction.tStopRefresh = tThisFlipGlobal
        thisExp.addData(
            "invisible_typing_introduction.stopped", invisible_typing_introduction.tStop
        )
        routineTimer.reset()
        thisExp.nextEntry()

        orientationInvisibleTypingLoop = data.TrialHandler2(
            name="orientationInvisibleTypingLoop",
            nReps=3,
            method="sequential",
            extraInfo=expInfo,
            originPath=-1,
            trialList=[None],
            seed=None,
        )
        thisExp.addLoop(orientationInvisibleTypingLoop)
        thisOrientationInvisibleTypingLoop = orientationInvisibleTypingLoop.trialList[0]

        if thisOrientationInvisibleTypingLoop != None:
            for paramName in thisOrientationInvisibleTypingLoop:
                globals()[paramName] = thisOrientationInvisibleTypingLoop[paramName]

        for thisOrientationInvisibleTypingLoop in orientationInvisibleTypingLoop:
            currentLoop = orientationInvisibleTypingLoop
            thisExp.timestampOnFlip(win, "thisRow.t")

            if thisOrientationInvisibleTypingLoop != None:
                for paramName in thisOrientationInvisibleTypingLoop:
                    globals()[paramName] = thisOrientationInvisibleTypingLoop[paramName]

            # --- Prepare to start Routine "orientation_invisible_typing_trial" ---
            orientation_invisible_typing_trial = data.Routine(
                name="orientation_invisible_typing_trial",
                components=[dummy_typing_trial],
            )
            orientation_invisible_typing_trial.status = NOT_STARTED
            continueRoutine = True

            current_invisible_word = ORIENTATION_INVISIBLE_WORDS[
                orientationInvisibleTypingLoop.thisN
            ]

            orientation_invisible_keyboard = Keyboard(win=win)
            orientationInvisibleTypingTask = PrimaryTask(
                win=win,
                keyboard=orientation_invisible_keyboard,
                mouse=mouse,
                mode="hard",
                ok_rect=ok_button_rect,
                ok_text=ok_button_text,
                show_reminder=True,
            )

            orientationInvisibleTypingTask.set_target_word(current_invisible_word)
            orientationInvisibleTypingTask.reset()
            orientationInvisibleTypingTask.show()

            orientation_invisible_typing_trial.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            orientation_invisible_typing_trial.tStart = globalClock.getTime(
                format="float"
            )
            orientation_invisible_typing_trial.status = STARTED
            thisExp.addData(
                "orientation_invisible_typing_trial.started",
                orientation_invisible_typing_trial.tStart,
            )
            orientation_invisible_typing_trial.maxDuration = None

            orientation_invisible_typing_trialComponents = (
                orientation_invisible_typing_trial.components
            )
            for thisComponent in orientation_invisible_typing_trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "orientation_invisible_typing_trial" ---
            if (
                isinstance(orientationInvisibleTypingLoop, data.TrialHandler2)
                and thisOrientationInvisibleTypingLoop.thisN
                != orientationInvisibleTypingLoop.thisTrial.thisN
            ):
                continueRoutine = False
            orientation_invisible_typing_trial.forceEnded = routineForceEnded = (
                not continueRoutine
            )
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                orientationInvisibleTypingTask.draw()
                status = orientationInvisibleTypingTask.check_input()
                if status == "submitted":
                    orientationInvisibleTypingTask.hide()
                    continueRoutine = False

                if (
                    dummy_typing_trial.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_typing_trial.frameNStart = frameN
                    dummy_typing_trial.tStart = t
                    dummy_typing_trial.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_typing_trial, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_typing_trial.started")
                    dummy_typing_trial.status = STARTED
                    dummy_typing_trial.setAutoDraw(True)

                if dummy_typing_trial.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    orientation_invisible_typing_trial.forceEnded = (
                        routineForceEnded
                    ) = True
                    break
                continueRoutine = False
                for thisComponent in orientation_invisible_typing_trial.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "orientation_invisible_typing_trial" ---
            for thisComponent in orientation_invisible_typing_trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            orientation_invisible_typing_trial.tStop = globalClock.getTime(
                format="float"
            )
            orientation_invisible_typing_trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData(
                "orientation_invisible_typing_trial.stopped",
                orientation_invisible_typing_trial.tStop,
            )
            routineTimer.reset()

            # --- Prepare to start Routine "invisible_typing_feedback" ---
            invisible_typing_feedback = data.Routine(
                name="invisible_typing_feedback",
                components=[dummy_feedback],
            )
            invisible_typing_feedback.status = NOT_STARTED
            continueRoutine = True

            invisibleFeedbackScreen = TypingFeedbackScreen(win=win)
            invisibleFeedbackScreen.set_feedback(
                current_invisible_word,
                orientationInvisibleTypingTask.text_entry_screen.entered_text,
            )
            invisibleFeedbackScreen.show()

            invisible_typing_feedback.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            invisible_typing_feedback.tStart = globalClock.getTime(format="float")
            invisible_typing_feedback.status = STARTED
            thisExp.addData(
                "invisible_typing_feedback.started", invisible_typing_feedback.tStart
            )
            invisible_typing_feedback.maxDuration = None

            invisible_typing_feedbackComponents = invisible_typing_feedback.components
            for thisComponent in invisible_typing_feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "invisible_typing_feedback" ---
            if (
                isinstance(orientationInvisibleTypingLoop, data.TrialHandler2)
                and thisOrientationInvisibleTypingLoop.thisN
                != orientationInvisibleTypingLoop.thisTrial.thisN
            ):
                continueRoutine = False
            invisible_typing_feedback.forceEnded = routineForceEnded = (
                not continueRoutine
            )
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                invisibleFeedbackScreen.draw()
                nav_result = invisibleFeedbackScreen.check_navigation(
                    mouse, defaultKeyboard
                )
                if nav_result == "escape":
                    thisExp.status = FINISHED
                    endExperiment(thisExp, win=win)
                    return
                elif nav_result:
                    continueRoutine = False

                if (
                    dummy_feedback.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_feedback.frameNStart = frameN
                    dummy_feedback.tStart = t
                    dummy_feedback.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_feedback, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_feedback.started")
                    dummy_feedback.status = STARTED
                    dummy_feedback.setAutoDraw(True)

                if dummy_feedback.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    invisible_typing_feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in invisible_typing_feedback.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "invisible_typing_feedback" ---
            for thisComponent in invisible_typing_feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            invisibleFeedbackScreen.hide()

            invisible_typing_feedback.tStop = globalClock.getTime(format="float")
            invisible_typing_feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData(
                "invisible_typing_feedback.stopped", invisible_typing_feedback.tStop
            )
            routineTimer.reset()
            thisExp.nextEntry()

        # -------------------------------------------------- ORIENTATION: NBACK --------------------------------------------------

        # --- Prepare to start Routine "nback_task_overview" ---
        nback_task_overview = data.Routine(
            name="nback_task_overview",
            components=[dummy_nback_instructions],
        )
        nback_task_overview.status = NOT_STARTED
        continueRoutine = True

        nbackOverviewScreen = InstructionScreen(
            win=win,
            title=NBACK_TASK_OVERVIEW_TITLE,
            content=NBACK_TASK_OVERVIEW_CONTENT,
            button_text="Begin",
        )
        nbackOverviewScreen.show()

        nback_task_overview.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        nback_task_overview.tStart = globalClock.getTime(format="float")
        nback_task_overview.status = STARTED
        thisExp.addData("nback_task_overview.started", nback_task_overview.tStart)
        nback_task_overview.maxDuration = None

        nback_task_overviewComponents = nback_task_overview.components
        for thisComponent in nback_task_overview.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "nback_task_overview" ---
        nback_task_overview.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            nbackOverviewScreen.draw()
            nav_result = nbackOverviewScreen.check_navigation(mouse, defaultKeyboard)
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_nback_instructions.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_nback_instructions.frameNStart = frameN
                dummy_nback_instructions.tStart = t
                dummy_nback_instructions.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_nback_instructions, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_nback_instructions.started")
                dummy_nback_instructions.status = STARTED
                dummy_nback_instructions.setAutoDraw(True)

            if dummy_nback_instructions.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                nback_task_overview.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in nback_task_overview.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "nback_task_overview" ---
        for thisComponent in nback_task_overview.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        nbackOverviewScreen.hide()

        nback_task_overview.tStop = globalClock.getTime(format="float")
        nback_task_overview.tStopRefresh = tThisFlipGlobal
        thisExp.addData("nback_task_overview.stopped", nback_task_overview.tStop)
        routineTimer.reset()
        thisExp.nextEntry()

        # --- Prepare to start Routine "nback_instructions" ---
        nback_instructions = data.Routine(
            name="nback_instructions",
            components=[dummy_nback_instructions],
        )
        nback_instructions.status = NOT_STARTED
        continueRoutine = True

        nbackInstructionScreen = NbackInstructionsScreen(win=win, n_back_level=1)
        nbackInstructionScreen.show()

        nback_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        nback_instructions.tStart = globalClock.getTime(format="float")
        nback_instructions.status = STARTED
        thisExp.addData("nback_instructions.started", nback_instructions.tStart)
        nback_instructions.maxDuration = None

        nback_instructionsComponents = nback_instructions.components
        for thisComponent in nback_instructions.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "nback_instructions" ---
        nback_instructions.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            nbackInstructionScreen.draw()
            nav_result = nbackInstructionScreen.check_navigation(mouse, defaultKeyboard)
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_nback_instructions.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_nback_instructions.frameNStart = frameN
                dummy_nback_instructions.tStart = t
                dummy_nback_instructions.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_nback_instructions, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_nback_instructions.started")
                dummy_nback_instructions.status = STARTED
                dummy_nback_instructions.setAutoDraw(True)

            if dummy_nback_instructions.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                nback_instructions.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in nback_instructions.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "nback_instructions" ---
        for thisComponent in nback_instructions.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        nbackInstructionScreen.hide()

        nback_instructions.tStop = globalClock.getTime(format="float")
        nback_instructions.tStopRefresh = tThisFlipGlobal
        thisExp.addData("nback_instructions.stopped", nback_instructions.tStop)
        routineTimer.reset()
        thisExp.nextEntry()

        orientationNbackLoop = data.TrialHandler2(
            name="orientationNbackLoop",
            nReps=6,
            method="sequential",
            extraInfo=expInfo,
            originPath=-1,
            trialList=[None],
            seed=None,
        )
        thisExp.addLoop(orientationNbackLoop)
        thisOrientationNbackLoop = orientationNbackLoop.trialList[0]

        if thisOrientationNbackLoop != None:
            for paramName in thisOrientationNbackLoop:
                globals()[paramName] = thisOrientationNbackLoop[paramName]

        for thisOrientationNbackLoop in orientationNbackLoop:
            currentLoop = orientationNbackLoop
            thisExp.timestampOnFlip(win, "thisRow.t")

            if thisOrientationNbackLoop != None:
                for paramName in thisOrientationNbackLoop:
                    globals()[paramName] = thisOrientationNbackLoop[paramName]

            # --- Prepare to start Routine "nback_trial" ---
            nback_trial = data.Routine(
                name="nback_trial",
                components=[dummy_nback_trial],
            )
            nback_trial.status = NOT_STARTED
            continueRoutine = True

            if orientationNbackLoop.thisN == 3:
                nback2InstructionScreen = NbackInstructionsScreen(
                    win=win, n_back_level=2
                )
                nback2InstructionScreen.show()

                instruction_continue = False
                while not instruction_continue:
                    nback2InstructionScreen.draw()
                    if nback2InstructionScreen.check_navigation(mouse):
                        instruction_continue = True
                    win.flip()

                nback2InstructionScreen.hide()

            if orientationNbackLoop.thisN < 3:
                nback_type = "1-back"
                nback_level = 1
                sequence_index = orientationNbackLoop.thisN
            else:
                nback_type = "2-back"
                nback_level = 2
                sequence_index = orientationNbackLoop.thisN - 3

            practice_data = ORIENTATION_NBACK_DATA[nback_type][sequence_index]
            orientationNbackTask = NbackTask(
                win=win,
                n_back_level=practice_data["n_back_level"],
                num_stims=practice_data["num_stims"],
                stim_list=practice_data["stim_list"],
                stim_durations=practice_data["stim_durations"],
                matches=practice_data["match_positions"],
                isi_times=practice_data["isi_times"],
                feedback_enabled=True,
                mask_enabled=True,
                show_intro_screen=True,
                show_reminder=True,
            )

            orientationNbackTask.show()
            orientationNbackTask.start()

            nback_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            nback_trial.tStart = globalClock.getTime(format="float")
            nback_trial.status = STARTED
            thisExp.addData("nback_trial.started", nback_trial.tStart)
            nback_trial.maxDuration = None

            nback_trialComponents = nback_trial.components
            for thisComponent in nback_trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "nback_trial" ---
            if (
                isinstance(orientationNbackLoop, data.TrialHandler2)
                and thisOrientationNbackLoop.thisN
                != orientationNbackLoop.thisTrial.thisN
            ):
                continueRoutine = False
            nback_trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                orientationNbackTask.draw()
                status = orientationNbackTask.check_key()

                if status == "end":
                    orientationNbackTask.hide()
                    continueRoutine = False

                if (
                    dummy_nback_trial.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_nback_trial.frameNStart = frameN
                    dummy_nback_trial.tStart = t
                    dummy_nback_trial.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_nback_trial, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_nback_trial.started")
                    dummy_nback_trial.status = STARTED
                    dummy_nback_trial.setAutoDraw(True)

                if dummy_nback_trial.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    nback_trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in nback_trial.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "nback_trial" ---
            for thisComponent in nback_trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            current_nback_performance = orientationNbackTask.get_performance()

            nback_trial.tStop = globalClock.getTime(format="float")
            nback_trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData("nback_trial.stopped", nback_trial.tStop)
            routineTimer.reset()

            # --- Prepare to start Routine "nback_trial_feedback" ---
            nback_trial_feedback = data.Routine(
                name="nback_trial_feedback",
                components=[dummy_nback_feedback],
            )
            nback_trial_feedback.status = NOT_STARTED
            continueRoutine = True

            nbackTrialFeedbackScreen = NbackFeedbackScreen(win=win)
            nbackTrialFeedbackScreen.set_feedback(current_nback_performance)
            nbackTrialFeedbackScreen.show()

            nback_trial_feedback.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            nback_trial_feedback.tStart = globalClock.getTime(format="float")
            nback_trial_feedback.status = STARTED
            thisExp.addData("nback_trial_feedback.started", nback_trial_feedback.tStart)
            nback_trial_feedback.maxDuration = None

            nback_trial_feedbackComponents = nback_trial_feedback.components
            for thisComponent in nback_trial_feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "nback_trial_feedback" ---
            if (
                isinstance(orientationNbackLoop, data.TrialHandler2)
                and thisOrientationNbackLoop.thisN
                != orientationNbackLoop.thisTrial.thisN
            ):
                continueRoutine = False
            nback_trial_feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                nbackTrialFeedbackScreen.draw()
                nav_result = nbackTrialFeedbackScreen.check_navigation(
                    mouse, defaultKeyboard
                )
                if nav_result == "escape":
                    thisExp.status = FINISHED
                    endExperiment(thisExp, win=win)
                    return
                elif nav_result:
                    continueRoutine = False

                if (
                    dummy_nback_feedback.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_nback_feedback.frameNStart = frameN
                    dummy_nback_feedback.tStart = t
                    dummy_nback_feedback.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_nback_feedback, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_nback_feedback.started")
                    dummy_nback_feedback.status = STARTED
                    dummy_nback_feedback.setAutoDraw(True)

                if dummy_nback_feedback.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    nback_trial_feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in nback_trial_feedback.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "nback_trial_feedback" ---
            for thisComponent in nback_trial_feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            nbackTrialFeedbackScreen.hide()

            nback_trial_feedback.tStop = globalClock.getTime(format="float")
            nback_trial_feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData("nback_trial_feedback.stopped", nback_trial_feedback.tStop)
            routineTimer.reset()
            thisExp.nextEntry()

        # -------------------------------------------------- ORIENTATION: INTERRUPTION TRAINING --------------------------------------------------

        # --- Prepare to start Routine "interruption_training_intro" ---
        interruption_training_intro = data.Routine(
            name="interruption_training_intro",
            components=[dummy_typing_intro],
        )
        interruption_training_intro.status = NOT_STARTED
        continueRoutine = True

        interruptionIntroScreen = InstructionScreen(
            win=win,
            title=INTERRUPTION_TRAINING_INTRO_TITLE,
            content=INTERRUPTION_TRAINING_INTRO_CONTENT,
            button_text="Begin",
        )
        interruptionIntroScreen.show()

        interruption_training_intro.tStartRefresh = win.getFutureFlipTime(
            clock=globalClock
        )
        interruption_training_intro.tStart = globalClock.getTime(format="float")
        interruption_training_intro.status = STARTED
        thisExp.addData(
            "interruption_training_intro.started", interruption_training_intro.tStart
        )
        interruption_training_intro.maxDuration = None

        interruption_training_introComponents = interruption_training_intro.components
        for thisComponent in interruption_training_intro.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "interruption_training_intro" ---
        interruption_training_intro.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            interruptionIntroScreen.draw()
            nav_result = interruptionIntroScreen.check_navigation(
                mouse, defaultKeyboard
            )
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_typing_intro.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_typing_intro.frameNStart = frameN
                dummy_typing_intro.tStart = t
                dummy_typing_intro.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_typing_intro, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_typing_intro.started")
                dummy_typing_intro.status = STARTED
                dummy_typing_intro.setAutoDraw(True)

            if dummy_typing_intro.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                interruption_training_intro.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in interruption_training_intro.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "interruption_training_intro" ---
        for thisComponent in interruption_training_intro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        interruptionIntroScreen.hide()

        interruption_training_intro.tStop = globalClock.getTime(format="float")
        interruption_training_intro.tStopRefresh = tThisFlipGlobal
        thisExp.addData(
            "interruption_training_intro.stopped", interruption_training_intro.tStop
        )
        routineTimer.reset()
        thisExp.nextEntry()

        # --- Prepare to start Routine "interruption_understanding_intro" ---
        interruption_understanding_intro = data.Routine(
            name="interruption_understanding_intro",
            components=[dummy_typing_intro],
        )
        interruption_understanding_intro.status = NOT_STARTED
        continueRoutine = True

        understandingIntroScreen = InstructionScreen(
            win=win,
            title=INTERRUPTION_UNDERSTANDING_TITLE,
            content=INTERRUPTION_UNDERSTANDING_CONTENT,
            button_text="Start Practice",
            button_width=0.3,
        )
        understandingIntroScreen.show()

        interruption_understanding_intro.tStartRefresh = win.getFutureFlipTime(
            clock=globalClock
        )
        interruption_understanding_intro.tStart = globalClock.getTime(format="float")
        interruption_understanding_intro.status = STARTED
        thisExp.addData(
            "interruption_understanding_intro.started",
            interruption_understanding_intro.tStart,
        )
        interruption_understanding_intro.maxDuration = None

        interruption_understanding_introComponents = (
            interruption_understanding_intro.components
        )
        for thisComponent in interruption_understanding_intro.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "interruption_understanding_intro" ---
        interruption_understanding_intro.forceEnded = routineForceEnded = (
            not continueRoutine
        )
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            understandingIntroScreen.draw()
            nav_result = understandingIntroScreen.check_navigation(
                mouse, defaultKeyboard
            )
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_typing_intro.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_typing_intro.frameNStart = frameN
                dummy_typing_intro.tStart = t
                dummy_typing_intro.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_typing_intro, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_typing_intro.started")
                dummy_typing_intro.status = STARTED
                dummy_typing_intro.setAutoDraw(True)

            if dummy_typing_intro.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                interruption_understanding_intro.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in interruption_understanding_intro.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "interruption_understanding_intro" ---
        for thisComponent in interruption_understanding_intro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        understandingIntroScreen.hide()

        interruption_understanding_intro.tStop = globalClock.getTime(format="float")
        interruption_understanding_intro.tStopRefresh = tThisFlipGlobal
        thisExp.addData(
            "interruption_understanding_intro.stopped",
            interruption_understanding_intro.tStop,
        )
        routineTimer.reset()
        thisExp.nextEntry()

        interruptionUnderstandingLoop = data.TrialHandler2(
            name="interruptionUnderstandingLoop",
            nReps=2,
            method="sequential",
            extraInfo=expInfo,
            originPath=-1,
            trialList=[None],
            seed=None,
        )
        thisExp.addLoop(interruptionUnderstandingLoop)
        thisInterruptionUnderstandingLoop = interruptionUnderstandingLoop.trialList[0]

        if thisInterruptionUnderstandingLoop != None:
            for paramName in thisInterruptionUnderstandingLoop:
                globals()[paramName] = thisInterruptionUnderstandingLoop[paramName]

        for thisInterruptionUnderstandingLoop in interruptionUnderstandingLoop:
            currentLoop = interruptionUnderstandingLoop
            thisExp.timestampOnFlip(win, "thisRow.t")

            if thisInterruptionUnderstandingLoop != None:
                for paramName in thisInterruptionUnderstandingLoop:
                    globals()[paramName] = thisInterruptionUnderstandingLoop[paramName]

            # --- Prepare to start Routine "interruption_trial" ---
            interruption_trial = data.Routine(
                name="interruption_trial",
                components=[dummy_typing_trial],
            )
            interruption_trial.status = NOT_STARTED
            continueRoutine = True

            trial_index = interruptionUnderstandingLoop.thisN
            current_word = ORIENTATION_INTERRUPTION_WORDS[trial_index]
            interrupt_position = ORIENTATION_INTERRUPTION_POSITIONS[trial_index]

            if trial_index == 0:
                nback_type = "1-back"
                sequence_index = 0
            else:
                nback_type = "2-back"
                sequence_index = 0

            nback_data = ORIENTATION_INTERRUPTION_NBACK_DATA[nback_type][sequence_index]

            interruption_keyboard = Keyboard(win=win)
            interruptionPrimaryTask = PrimaryTask(
                win=win,
                keyboard=interruption_keyboard,
                mouse=mouse,
                mode="hard",
                ok_rect=ok_button_rect,
                ok_text=ok_button_text,
            )

            interruptionTrialManager = TrialManager(interrupt_position, [nback_data])

            interruptionPrimaryTask.set_target_word(current_word)
            interruptionPrimaryTask.reset()
            interruptionPrimaryTask.show()

            interruption_occurred = False
            nback_task_created = False
            typing_performance = None
            nback_performance = None

            interruptionSequence = "none"
            interruptionTimer = core.Clock()

            interruption_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            interruption_trial.tStart = globalClock.getTime(format="float")
            interruption_trial.status = STARTED
            thisExp.addData("interruption_trial.started", interruption_trial.tStart)
            interruption_trial.maxDuration = None

            interruption_trialComponents = interruption_trial.components
            for thisComponent in interruption_trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "interruption_trial" ---
            if (
                isinstance(interruptionUnderstandingLoop, data.TrialHandler2)
                and thisInterruptionUnderstandingLoop.thisN
                != interruptionUnderstandingLoop.thisTrial.thisN
            ):
                continueRoutine = False
            interruption_trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                if not interruption_occurred:
                    interruptionPrimaryTask.draw()

                    if interruptionSequence == "none":
                        status = interruptionPrimaryTask.check_input()
                        interruptionTrialManager.update(interruptionPrimaryTask)

                        if status == "submitted":
                            typing_performance = {
                                "target": current_word,
                                "entered": interruptionPrimaryTask.text_entry_screen.entered_text,
                            }
                            interruptionPrimaryTask.hide()
                            continueRoutine = False
                        elif interruptionTrialManager.consume():
                            interruptionSequence = "masked"
                            interruptionTimer.reset()

                    elif interruptionSequence == "masked":
                        if interruptionTimer.getTime() >= INTERRUPT_DELAY:
                            interruptionSequence = "complete"
                            interruption_occurred = True
                            interruptionPrimaryTask.hide()

                            orientationInterruptionNback = NbackTask(
                                win=win,
                                n_back_level=nback_data["n_back_level"],
                                num_stims=nback_data["num_stims"],
                                stim_list=nback_data["stim_list"],
                                stim_durations=nback_data["stim_durations"],
                                matches=nback_data["match_positions"],
                                isi_times=nback_data["isi_times"],
                                feedback_enabled=True,
                                show_intro_screen=True,
                                mask_enabled=True,
                            )
                            orientationInterruptionNback.show()
                            orientationInterruptionNback.start()
                            nback_task_created = True

                else:
                    if nback_task_created:
                        orientationInterruptionNback.draw()
                        status = orientationInterruptionNback.check_key()

                        if status == "end":
                            nback_performance = (
                                orientationInterruptionNback.get_performance()
                            )
                            orientationInterruptionNback.hide()
                            interruption_occurred = False
                            nback_task_created = False
                            interruptionSequence = "none"
                            interruptionPrimaryTask.show()

                if (
                    dummy_typing_trial.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_typing_trial.frameNStart = frameN
                    dummy_typing_trial.tStart = t
                    dummy_typing_trial.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_typing_trial, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_typing_trial.started")
                    dummy_typing_trial.status = STARTED
                    dummy_typing_trial.setAutoDraw(True)

                if dummy_typing_trial.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    interruption_trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in interruption_trial.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "interruption_trial" ---
            for thisComponent in interruption_trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            if typing_performance is None:
                typing_performance = {
                    "target": current_word,
                    "entered": interruptionPrimaryTask.text_entry_screen.entered_text,
                }

            interruption_trial.tStop = globalClock.getTime(format="float")
            interruption_trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData("interruption_trial.stopped", interruption_trial.tStop)
            routineTimer.reset()

            # --- Prepare to start Routine "interruption_feedback" ---
            interruption_feedback = data.Routine(
                name="interruption_feedback",
                components=[dummy_feedback],
            )
            interruption_feedback.status = NOT_STARTED
            continueRoutine = True

            interruptionFeedbackScreen = CombinedTaskFeedbackScreen(win=win)
            interruptionFeedbackScreen.set_feedback(
                typing_performance["target"],
                typing_performance["entered"],
                nback_performance,
            )
            interruptionFeedbackScreen.show()

            interruption_feedback.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            interruption_feedback.tStart = globalClock.getTime(format="float")
            interruption_feedback.status = STARTED
            thisExp.addData(
                "interruption_feedback.started", interruption_feedback.tStart
            )
            interruption_feedback.maxDuration = None

            interruption_feedbackComponents = interruption_feedback.components
            for thisComponent in interruption_feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "interruption_feedback" ---
            if (
                isinstance(interruptionUnderstandingLoop, data.TrialHandler2)
                and thisInterruptionUnderstandingLoop.thisN
                != interruptionUnderstandingLoop.thisTrial.thisN
            ):
                continueRoutine = False
            interruption_feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                interruptionFeedbackScreen.draw()
                nav_result = interruptionFeedbackScreen.check_navigation(
                    mouse, defaultKeyboard
                )
                if nav_result == "escape":
                    thisExp.status = FINISHED
                    endExperiment(thisExp, win=win)
                    return
                elif nav_result:
                    continueRoutine = False

                if (
                    dummy_feedback.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_feedback.frameNStart = frameN
                    dummy_feedback.tStart = t
                    dummy_feedback.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_feedback, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_feedback.started")
                    dummy_feedback.status = STARTED
                    dummy_feedback.setAutoDraw(True)

                if dummy_feedback.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    interruption_feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in interruption_feedback.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "interruption_feedback" ---
            for thisComponent in interruption_feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            interruptionFeedbackScreen.hide()

            interruption_feedback.tStop = globalClock.getTime(format="float")
            interruption_feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData(
                "interruption_feedback.stopped", interruption_feedback.tStop
            )
            routineTimer.reset()
            thisExp.nextEntry()

        # --- Prepare to start Routine "interruption_practice_intro" ---
        interruption_practice_intro = data.Routine(
            name="interruption_practice_intro",
            components=[dummy_typing_intro],
        )
        interruption_practice_intro.status = NOT_STARTED
        continueRoutine = True

        practiceIntroScreen = InstructionScreen(
            win=win,
            title=INTERRUPTION_PRACTICE_TITLE,
            content=INTERRUPTION_PRACTICE_CONTENT,
            button_text="Start Practice",
            button_width=0.3,
        )
        practiceIntroScreen.show()

        interruption_practice_intro.tStartRefresh = win.getFutureFlipTime(
            clock=globalClock
        )
        interruption_practice_intro.tStart = globalClock.getTime(format="float")
        interruption_practice_intro.status = STARTED
        thisExp.addData(
            "interruption_practice_intro.started", interruption_practice_intro.tStart
        )
        interruption_practice_intro.maxDuration = None

        interruption_practice_introComponents = interruption_practice_intro.components
        for thisComponent in interruption_practice_intro.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "interruption_practice_intro" ---
        interruption_practice_intro.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            practiceIntroScreen.draw()
            nav_result = practiceIntroScreen.check_navigation(mouse, defaultKeyboard)
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_typing_intro.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_typing_intro.frameNStart = frameN
                dummy_typing_intro.tStart = t
                dummy_typing_intro.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_typing_intro, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_typing_intro.started")
                dummy_typing_intro.status = STARTED
                dummy_typing_intro.setAutoDraw(True)

            if dummy_typing_intro.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                interruption_practice_intro.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in interruption_practice_intro.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "interruption_practice_intro" ---
        for thisComponent in interruption_practice_intro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        practiceIntroScreen.hide()

        interruption_practice_intro.tStop = globalClock.getTime(format="float")
        interruption_practice_intro.tStopRefresh = tThisFlipGlobal
        thisExp.addData(
            "interruption_practice_intro.stopped", interruption_practice_intro.tStop
        )
        routineTimer.reset()
        thisExp.nextEntry()

        interruptionPracticeLoop = data.TrialHandler2(
            name="interruptionPracticeLoop",
            nReps=4,
            method="sequential",
            extraInfo=expInfo,
            originPath=-1,
            trialList=[None],
            seed=None,
        )
        thisExp.addLoop(interruptionPracticeLoop)
        thisInterruptionPracticeLoop = interruptionPracticeLoop.trialList[0]

        if thisInterruptionPracticeLoop != None:
            for paramName in thisInterruptionPracticeLoop:
                globals()[paramName] = thisInterruptionPracticeLoop[paramName]

        for thisInterruptionPracticeLoop in interruptionPracticeLoop:
            currentLoop = interruptionPracticeLoop
            thisExp.timestampOnFlip(win, "thisRow.t")

            if thisInterruptionPracticeLoop != None:
                for paramName in thisInterruptionPracticeLoop:
                    globals()[paramName] = thisInterruptionPracticeLoop[paramName]

            # --- Prepare to start Routine "interruption_practice_trial" ---
            interruption_practice_trial = data.Routine(
                name="interruption_practice_trial",
                components=[dummy_typing_trial],
            )
            interruption_practice_trial.status = NOT_STARTED
            continueRoutine = True

            trial_index = interruptionPracticeLoop.thisN + 2
            current_word = ORIENTATION_INTERRUPTION_WORDS[trial_index]

            trial_configs = [
                ("1-back", 1, "interrupted"),
                ("2-back", 1, "sequential"),
                ("2-back", 2, "interrupted"),
                ("1-back", 2, "sequential"),
            ]

            nback_type, sequence_index, practice_condition = trial_configs[
                interruptionPracticeLoop.thisN
            ]
            nback_data = ORIENTATION_INTERRUPTION_NBACK_DATA[nback_type][sequence_index]

            if practice_condition == "interrupted":
                interrupt_position = ORIENTATION_INTERRUPTION_POSITIONS[trial_index]
            else:
                interrupt_position = []

            practice_keyboard = Keyboard(win=win)
            practicePrimaryTask = PrimaryTask(
                win=win,
                keyboard=practice_keyboard,
                mouse=mouse,
                mode="hard",
                ok_rect=ok_button_rect,
                ok_text=ok_button_text,
            )

            practiceTrialManager = TrialManager(interrupt_position, [nback_data])

            practicePrimaryTask.set_target_word(current_word)
            practicePrimaryTask.reset()
            practicePrimaryTask.show()

            primary_completed = False
            nback_completed = False
            typing_performance = None
            nback_performance = None

            show_nback_next = False

            interruptionSequence = "none"
            interruptionTimer = core.Clock()

            interruption_practice_trial.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            interruption_practice_trial.tStart = globalClock.getTime(format="float")
            interruption_practice_trial.status = STARTED
            thisExp.addData(
                "interruption_practice_trial.started",
                interruption_practice_trial.tStart,
            )
            interruption_practice_trial.maxDuration = None

            interruption_practice_trialComponents = (
                interruption_practice_trial.components
            )
            for thisComponent in interruption_practice_trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "interruption_practice_trial" ---
            if (
                isinstance(interruptionPracticeLoop, data.TrialHandler2)
                and thisInterruptionPracticeLoop.thisN
                != interruptionPracticeLoop.thisTrial.thisN
            ):
                continueRoutine = False
            interruption_practice_trial.forceEnded = routineForceEnded = (
                not continueRoutine
            )
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                if practice_condition == "interrupted":
                    if not show_nback_next:
                        practicePrimaryTask.draw()

                        if interruptionSequence == "none":
                            status = practicePrimaryTask.check_input()
                            practiceTrialManager.update(practicePrimaryTask)

                            if status == "submitted":
                                typing_performance = {
                                    "target": current_word,
                                    "entered": practicePrimaryTask.text_entry_screen.entered_text,
                                }
                                primary_completed = True
                                practicePrimaryTask.hide()
                                if nback_completed:
                                    continueRoutine = False
                            elif practiceTrialManager.consume():
                                interruptionSequence = "masked"
                                interruptionTimer.reset()

                        elif interruptionSequence == "masked":
                            if interruptionTimer.getTime() >= INTERRUPT_DELAY:
                                interruptionSequence = "complete"
                                show_nback_next = True
                                practicePrimaryTask.hide()

                                practiceNbackTask = NbackTask(
                                    win=win,
                                    n_back_level=nback_data["n_back_level"],
                                    num_stims=nback_data["num_stims"],
                                    stim_list=nback_data["stim_list"],
                                    stim_durations=nback_data["stim_durations"],
                                    matches=nback_data["match_positions"],
                                    isi_times=nback_data["isi_times"],
                                    feedback_enabled=True,
                                    show_intro_screen=True,
                                    mask_enabled=True,
                                )
                                practiceNbackTask.show()
                                practiceNbackTask.start()
                    else:
                        practiceNbackTask.draw()
                        status = practiceNbackTask.check_key()

                        if status == "end":
                            nback_performance = practiceNbackTask.get_performance()
                            practiceNbackTask.hide()
                            nback_completed = True

                            if primary_completed:
                                continueRoutine = False
                            else:
                                show_nback_next = False
                                interruptionSequence = "none"
                                practicePrimaryTask.show()

                else:
                    if not primary_completed:
                        practicePrimaryTask.draw()
                        status = practicePrimaryTask.check_input()

                        if status == "submitted":
                            typing_performance = {
                                "target": current_word,
                                "entered": practicePrimaryTask.text_entry_screen.entered_text,
                            }
                            primary_completed = True
                            practicePrimaryTask.hide()

                            practiceNbackTask = NbackTask(
                                win=win,
                                n_back_level=nback_data["n_back_level"],
                                num_stims=nback_data["num_stims"],
                                stim_list=nback_data["stim_list"],
                                stim_durations=nback_data["stim_durations"],
                                matches=nback_data["match_positions"],
                                isi_times=nback_data["isi_times"],
                                feedback_enabled=True,
                                show_intro_screen=True,
                                mask_enabled=True,
                            )
                            practiceNbackTask.show()
                            practiceNbackTask.start()
                    else:
                        practiceNbackTask.draw()
                        status = practiceNbackTask.check_key()

                        if status == "end":
                            nback_performance = practiceNbackTask.get_performance()
                            practiceNbackTask.hide()
                            nback_completed = True
                            continueRoutine = False

                if (
                    dummy_typing_trial.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_typing_trial.frameNStart = frameN
                    dummy_typing_trial.tStart = t
                    dummy_typing_trial.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_typing_trial, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_typing_trial.started")
                    dummy_typing_trial.status = STARTED
                    dummy_typing_trial.setAutoDraw(True)

                if dummy_typing_trial.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    interruption_practice_trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in interruption_practice_trial.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "interruption_practice_trial" ---
            for thisComponent in interruption_practice_trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            if typing_performance is None:
                typing_performance = {
                    "target": current_word,
                    "entered": practicePrimaryTask.text_entry_screen.entered_text,
                }

            practice_condition_used = practice_condition

            interruption_practice_trial.tStop = globalClock.getTime(format="float")
            interruption_practice_trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData(
                "interruption_practice_trial.stopped", interruption_practice_trial.tStop
            )
            routineTimer.reset()

            # --- Prepare to start Routine "interruption_practice_feedback" ---
            interruption_practice_feedback = data.Routine(
                name="interruption_practice_feedback",
                components=[dummy_feedback],
            )
            interruption_practice_feedback.status = NOT_STARTED
            continueRoutine = True

            practiceFeedbackScreen = CombinedTaskFeedbackScreen(win=win)

            practiceFeedbackScreen.title_stim.text = f"Task Results"

            practiceFeedbackScreen.set_feedback(
                typing_performance["target"],
                typing_performance["entered"],
                nback_performance,
            )
            practiceFeedbackScreen.show()

            interruption_practice_feedback.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            interruption_practice_feedback.tStart = globalClock.getTime(format="float")
            interruption_practice_feedback.status = STARTED
            thisExp.addData(
                "interruption_practice_feedback.started",
                interruption_practice_feedback.tStart,
            )
            interruption_practice_feedback.maxDuration = None

            interruption_practice_feedbackComponents = (
                interruption_practice_feedback.components
            )
            for thisComponent in interruption_practice_feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "interruption_practice_feedback" ---
            if (
                isinstance(interruptionPracticeLoop, data.TrialHandler2)
                and thisInterruptionPracticeLoop.thisN
                != interruptionPracticeLoop.thisTrial.thisN
            ):
                continueRoutine = False
            interruption_practice_feedback.forceEnded = routineForceEnded = (
                not continueRoutine
            )
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                practiceFeedbackScreen.draw()
                nav_result = practiceFeedbackScreen.check_navigation(
                    mouse, defaultKeyboard
                )
                if nav_result == "escape":
                    thisExp.status = FINISHED
                    endExperiment(thisExp, win=win)
                    return
                elif nav_result:
                    continueRoutine = False

                if (
                    dummy_feedback.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_feedback.frameNStart = frameN
                    dummy_feedback.tStart = t
                    dummy_feedback.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_feedback, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_feedback.started")
                    dummy_feedback.status = STARTED
                    dummy_feedback.setAutoDraw(True)

                if dummy_feedback.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    interruption_practice_feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in interruption_practice_feedback.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "interruption_practice_feedback" ---
            for thisComponent in interruption_practice_feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            practiceFeedbackScreen.hide()

            interruption_practice_feedback.tStop = globalClock.getTime(format="float")
            interruption_practice_feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData(
                "interruption_practice_feedback.stopped",
                interruption_practice_feedback.tStop,
            )
            routineTimer.reset()
            thisExp.nextEntry()

        # ------ ORIENTATION: TIME ESTIMATION TRAINING ------

        # --- Prepare to start Routine "time_estimation_intro" ---
        time_estimation_intro = data.Routine(
            name="time_estimation_intro",
            components=[dummy_typing_intro],
        )
        time_estimation_intro.status = NOT_STARTED
        continueRoutine = True

        timeIntroScreen = InstructionScreen(
            win=win,
            title=TIME_ESTIMATION_INTRO_TITLE,
            content=TIME_ESTIMATION_INTRO_CONTENT,
            button_text="Begin",
        )
        timeIntroScreen.show()

        time_estimation_intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        time_estimation_intro.tStart = globalClock.getTime(format="float")
        time_estimation_intro.status = STARTED
        thisExp.addData("time_estimation_intro.started", time_estimation_intro.tStart)
        time_estimation_intro.maxDuration = None

        time_estimation_introComponents = time_estimation_intro.components
        for thisComponent in time_estimation_intro.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "time_estimation_intro" ---
        time_estimation_intro.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            timeIntroScreen.draw()
            nav_result = timeIntroScreen.check_navigation(mouse, defaultKeyboard)
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_typing_intro.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_typing_intro.frameNStart = frameN
                dummy_typing_intro.tStart = t
                dummy_typing_intro.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_typing_intro, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_typing_intro.started")
                dummy_typing_intro.status = STARTED
                dummy_typing_intro.setAutoDraw(True)

            if dummy_typing_intro.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                time_estimation_intro.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in time_estimation_intro.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "time_estimation_intro" ---
        for thisComponent in time_estimation_intro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        timeIntroScreen.hide()

        time_estimation_intro.tStop = globalClock.getTime(format="float")
        time_estimation_intro.tStopRefresh = tThisFlipGlobal
        thisExp.addData("time_estimation_intro.stopped", time_estimation_intro.tStop)
        routineTimer.reset()
        thisExp.nextEntry()

        # ------ SUB-PHASE 1: TYPING + TIME ESTIMATION ------

        # --- Prepare to start Routine "time_phase1_intro" ---
        time_phase1_intro = data.Routine(
            name="time_phase1_intro",
            components=[dummy_typing_intro],
        )
        time_phase1_intro.status = NOT_STARTED
        continueRoutine = True

        phase1IntroScreen = InstructionScreen(
            win=win,
            title=TIME_PHASE1_TITLE,
            content=TIME_PHASE1_CONTENT,
            button_text="Start Practice",
            button_width=0.3,
        )
        phase1IntroScreen.show()

        time_phase1_intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        time_phase1_intro.tStart = globalClock.getTime(format="float")
        time_phase1_intro.status = STARTED
        thisExp.addData("time_phase1_intro.started", time_phase1_intro.tStart)
        time_phase1_intro.maxDuration = None

        time_phase1_introComponents = time_phase1_intro.components
        for thisComponent in time_phase1_intro.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "time_phase1_intro" ---
        time_phase1_intro.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            phase1IntroScreen.draw()
            nav_result = phase1IntroScreen.check_navigation(mouse, defaultKeyboard)
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_typing_intro.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_typing_intro.frameNStart = frameN
                dummy_typing_intro.tStart = t
                dummy_typing_intro.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_typing_intro, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_typing_intro.started")
                dummy_typing_intro.status = STARTED
                dummy_typing_intro.setAutoDraw(True)

            if dummy_typing_intro.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                time_phase1_intro.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in time_phase1_intro.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "time_phase1_intro" ---
        for thisComponent in time_phase1_intro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        phase1IntroScreen.hide()

        time_phase1_intro.tStop = globalClock.getTime(format="float")
        time_phase1_intro.tStopRefresh = tThisFlipGlobal
        thisExp.addData("time_phase1_intro.stopped", time_phase1_intro.tStop)
        routineTimer.reset()
        thisExp.nextEntry()

        timePhase1Loop = data.TrialHandler2(
            name="timePhase1Loop",
            nReps=2,
            method="sequential",
            extraInfo=expInfo,
            originPath=-1,
            trialList=[None],
            seed=None,
        )
        thisExp.addLoop(timePhase1Loop)
        thisTimePhase1Loop = timePhase1Loop.trialList[0]

        if thisTimePhase1Loop != None:
            for paramName in thisTimePhase1Loop:
                globals()[paramName] = thisTimePhase1Loop[paramName]

        for thisTimePhase1Loop in timePhase1Loop:
            currentLoop = timePhase1Loop
            thisExp.timestampOnFlip(win, "thisRow.t")

            if thisTimePhase1Loop != None:
                for paramName in thisTimePhase1Loop:
                    globals()[paramName] = thisTimePhase1Loop[paramName]

            # --- Prepare to start Routine "time_phase1_trial" ---
            time_phase1_trial = data.Routine(
                name="time_phase1_trial",
                components=[dummy_typing_trial],
            )
            time_phase1_trial.status = NOT_STARTED
            continueRoutine = True

            current_word = ORIENTATION_TIME_ESTIMATION_WORDS[timePhase1Loop.thisN]

            phase1_keyboard = Keyboard(win=win)
            phase1TypingTask = PrimaryTask(
                win=win,
                keyboard=phase1_keyboard,
                mouse=mouse,
                mode="hard",
                ok_rect=ok_button_rect,
                ok_text=ok_button_text,
            )

            phase1TypingTask.set_target_word(current_word)
            phase1TypingTask.reset()
            phase1TypingTask.show()

            typing_performance = None

            dl.clear_trial_data()

            dl.log_multiple(
                {
                    "experiment_phase": "orientation_time_phase1",
                    "trial_number": timePhase1Loop.thisN + 1,
                    "target_word": current_word,
                    "target_word_length": len(current_word),
                    "typing_mode": "hard",
                }
            )

            trial_start_time = dl.log_timing_start("trial")
            typing_start_time = dl.log_timing_start("typing")

            time_phase1_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            time_phase1_trial.tStart = globalClock.getTime(format="float")
            time_phase1_trial.status = STARTED
            thisExp.addData("time_phase1_trial.started", time_phase1_trial.tStart)
            time_phase1_trial.maxDuration = None

            time_phase1_trialComponents = time_phase1_trial.components
            for thisComponent in time_phase1_trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "time_phase1_trial" ---
            if (
                isinstance(timePhase1Loop, data.TrialHandler2)
                and thisTimePhase1Loop.thisN != timePhase1Loop.thisTrial.thisN
            ):
                continueRoutine = False
            time_phase1_trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                phase1TypingTask.draw()
                status = phase1TypingTask.check_input()

                if status == "submitted":
                    typing_end_time = dl.log_timing_end("typing", typing_start_time)

                    entered_text = phase1TypingTask.text_entry_screen.entered_text
                    character_accuracy = dl.calculate_character_accuracy(
                        current_word, entered_text
                    )

                    dl.log_multiple(
                        {
                            "entered_text": entered_text,
                            "entered_text_length": len(entered_text),
                            "typing_correct": entered_text.upper()
                            == current_word.upper(),
                            "typing_accuracy": character_accuracy,
                            "typing_error_rate": 1.0 - character_accuracy,
                        }
                    )

                    typing_performance = {
                        "target": current_word,
                        "entered": entered_text,
                    }
                    phase1TypingTask.hide()
                    continueRoutine = False

                if (
                    dummy_typing_trial.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_typing_trial.frameNStart = frameN
                    dummy_typing_trial.tStart = t
                    dummy_typing_trial.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_typing_trial, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_typing_trial.started")
                    dummy_typing_trial.status = STARTED
                    dummy_typing_trial.setAutoDraw(True)

                if dummy_typing_trial.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    time_phase1_trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in time_phase1_trial.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "time_phase1_trial" ---
            for thisComponent in time_phase1_trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            trial_end_time = dl.log_timing_end("trial", trial_start_time)

            time_phase1_trial.tStop = globalClock.getTime(format="float")
            time_phase1_trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData("time_phase1_trial.stopped", time_phase1_trial.tStop)
            routineTimer.reset()

            # --- Prepare to start Routine "time_phase1_estimation" ---
            time_phase1_estimation = data.Routine(
                name="time_phase1_estimation",
                components=[dummy_typing_trial],
            )
            time_phase1_estimation.status = NOT_STARTED
            continueRoutine = True

            phase1_slider = TimeSlider(win=win)
            phase1TimeTask = TimeEstimationTask(
                win=win,
                input_device=phase1_slider,
                mouse=mouse,
                ok_rect=ok_button_rect,
                ok_text=ok_button_text,
            )
            phase1TimeTask.reset()
            phase1TimeTask.show()

            time_phase1_estimation.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            time_phase1_estimation.tStart = globalClock.getTime(format="float")
            time_phase1_estimation.status = STARTED
            thisExp.addData(
                "time_phase1_estimation.started", time_phase1_estimation.tStart
            )
            time_phase1_estimation.maxDuration = None

            time_phase1_estimationComponents = time_phase1_estimation.components
            for thisComponent in time_phase1_estimation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "time_phase1_estimation" ---
            if (
                isinstance(timePhase1Loop, data.TrialHandler2)
                and thisTimePhase1Loop.thisN != timePhase1Loop.thisTrial.thisN
            ):
                continueRoutine = False
            time_phase1_estimation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                phase1TimeTask.draw()
                status = phase1TimeTask.check_input()

                if status == "submitted":
                    time_estimate_string = phase1TimeTask.get_response()
                    minutes, seconds = phase1TimeTask.time_display.get_time()
                    estimated_seconds = minutes * 60 + seconds
                    phase1TimeTask.hide()
                    continueRoutine = False

                if (
                    dummy_typing_trial.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_typing_trial.frameNStart = frameN
                    dummy_typing_trial.tStart = t
                    dummy_typing_trial.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_typing_trial, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_typing_trial.started")
                    dummy_typing_trial.status = STARTED
                    dummy_typing_trial.setAutoDraw(True)

                if dummy_typing_trial.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    time_phase1_estimation.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in time_phase1_estimation.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "time_phase1_estimation" ---
            for thisComponent in time_phase1_estimation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            dl.log_multiple(
                {
                    "time_estimate_input": time_estimate_string,
                    "time_estimate_seconds": estimated_seconds,
                    "time_input_method": "slider",
                }
            )

            current_data = dl.get_current_data()
            actual_trial_duration = (
                current_data["cm_trial_end_timestamp"]
                - current_data["cm_trial_start_timestamp"]
            )

            # Phase 1: time on primary task equals typing duration
            time_on_primary_task = current_data.get("cm_typing_duration", 0)

            time_estimation_ratio = (
                estimated_seconds / actual_trial_duration
                if actual_trial_duration > 0
                else 0
            )
            time_estimation_error = estimated_seconds - actual_trial_duration
            time_estimation_accuracy = (
                abs(time_estimation_error) / actual_trial_duration
                if actual_trial_duration > 0
                else 0
            )

            dl.log_multiple(
                {
                    "actual_trial_duration": actual_trial_duration,
                    "total_trial_duration": actual_trial_duration,
                    "time_on_primary_task": time_on_primary_task,
                    "time_estimation_ratio": time_estimation_ratio,
                    "time_estimation_error": time_estimation_error,
                    "time_estimation_accuracy": time_estimation_accuracy,
                    "trial_completed": True,
                }
            )

            phase1_actual_duration = actual_trial_duration
            phase1_estimated_seconds = estimated_seconds

            time_phase1_estimation.tStop = globalClock.getTime(format="float")
            time_phase1_estimation.tStopRefresh = tThisFlipGlobal
            thisExp.addData(
                "time_phase1_estimation.stopped", time_phase1_estimation.tStop
            )
            routineTimer.reset()

            # --- Prepare to start Routine "time_phase1_feedback" ---
            time_phase1_feedback = data.Routine(
                name="time_phase1_feedback",
                components=[dummy_feedback],
            )
            time_phase1_feedback.status = NOT_STARTED
            continueRoutine = True

            phase1FeedbackScreen = TimingTaskFeedbackScreen(win=win)
            phase1FeedbackScreen.set_typing_feedback(
                typing_performance["target"], typing_performance["entered"]
            )
            phase1FeedbackScreen.set_timing_feedback(
                phase1_actual_duration,
                phase1_estimated_seconds,
            )
            phase1FeedbackScreen.show()

            time_phase1_feedback.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            time_phase1_feedback.tStart = globalClock.getTime(format="float")
            time_phase1_feedback.status = STARTED
            thisExp.addData("time_phase1_feedback.started", time_phase1_feedback.tStart)
            time_phase1_feedback.maxDuration = None

            time_phase1_feedbackComponents = time_phase1_feedback.components
            for thisComponent in time_phase1_feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "time_phase1_feedback" ---
            if (
                isinstance(timePhase1Loop, data.TrialHandler2)
                and thisTimePhase1Loop.thisN != timePhase1Loop.thisTrial.thisN
            ):
                continueRoutine = False
            time_phase1_feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                phase1FeedbackScreen.draw()
                nav_result = phase1FeedbackScreen.check_navigation(
                    mouse, defaultKeyboard
                )
                if nav_result == "escape":
                    thisExp.status = FINISHED
                    endExperiment(thisExp, win=win)
                    return
                elif nav_result:
                    continueRoutine = False

                if (
                    dummy_feedback.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_feedback.frameNStart = frameN
                    dummy_feedback.tStart = t
                    dummy_feedback.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_feedback, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_feedback.started")
                    dummy_feedback.status = STARTED
                    dummy_feedback.setAutoDraw(True)

                if dummy_feedback.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    time_phase1_feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in time_phase1_feedback.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "time_phase1_feedback" ---
            for thisComponent in time_phase1_feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            phase1FeedbackScreen.hide()

            time_phase1_feedback.tStop = globalClock.getTime(format="float")
            time_phase1_feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData("time_phase1_feedback.stopped", time_phase1_feedback.tStop)
            routineTimer.reset()
            thisExp.nextEntry()

        # ------ SUB-PHASE 2: N-BACK + TIME ESTIMATION ------

        # --- Prepare to start Routine "time_phase2_intro" ---
        time_phase2_intro = data.Routine(
            name="time_phase2_intro",
            components=[dummy_typing_intro],
        )
        time_phase2_intro.status = NOT_STARTED
        continueRoutine = True

        phase2IntroScreen = InstructionScreen(
            win=win,
            title=TIME_PHASE2_TITLE,
            content=TIME_PHASE2_CONTENT,
            button_text="Start Practice",
            button_width=0.3,
        )
        phase2IntroScreen.show()

        time_phase2_intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        time_phase2_intro.tStart = globalClock.getTime(format="float")
        time_phase2_intro.status = STARTED
        thisExp.addData("time_phase2_intro.started", time_phase2_intro.tStart)
        time_phase2_intro.maxDuration = None

        time_phase2_introComponents = time_phase2_intro.components
        for thisComponent in time_phase2_intro.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "time_phase2_intro" ---
        time_phase2_intro.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            phase2IntroScreen.draw()
            nav_result = phase2IntroScreen.check_navigation(mouse, defaultKeyboard)
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_typing_intro.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_typing_intro.frameNStart = frameN
                dummy_typing_intro.tStart = t
                dummy_typing_intro.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_typing_intro, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_typing_intro.started")
                dummy_typing_intro.status = STARTED
                dummy_typing_intro.setAutoDraw(True)

            if dummy_typing_intro.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                time_phase2_intro.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in time_phase2_intro.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "time_phase2_intro" ---
        for thisComponent in time_phase2_intro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        phase2IntroScreen.hide()

        time_phase2_intro.tStop = globalClock.getTime(format="float")
        time_phase2_intro.tStopRefresh = tThisFlipGlobal
        thisExp.addData("time_phase2_intro.stopped", time_phase2_intro.tStop)
        routineTimer.reset()
        thisExp.nextEntry()

        timePhase2Loop = data.TrialHandler2(
            name="timePhase2Loop",
            nReps=2,
            method="sequential",
            extraInfo=expInfo,
            originPath=-1,
            trialList=[None],
            seed=None,
        )
        thisExp.addLoop(timePhase2Loop)
        thisTimePhase2Loop = timePhase2Loop.trialList[0]

        if thisTimePhase2Loop != None:
            for paramName in thisTimePhase2Loop:
                globals()[paramName] = thisTimePhase2Loop[paramName]

        for thisTimePhase2Loop in timePhase2Loop:
            currentLoop = timePhase2Loop
            thisExp.timestampOnFlip(win, "thisRow.t")

            if thisTimePhase2Loop != None:
                for paramName in thisTimePhase2Loop:
                    globals()[paramName] = thisTimePhase2Loop[paramName]

            # --- Prepare to start Routine "time_phase2_trial" ---
            time_phase2_trial = data.Routine(
                name="time_phase2_trial",
                components=[dummy_nback_trial],
            )
            time_phase2_trial.status = NOT_STARTED
            continueRoutine = True

            nback_type = PHASE2_NBACK_PATTERN[timePhase2Loop.thisN]
            sequence_index = 0
            nback_data = ORIENTATION_TIME_NBACK_DATA[nback_type][sequence_index]

            phase2NbackTask = NbackTask(
                win=win,
                n_back_level=nback_data["n_back_level"],
                num_stims=nback_data["num_stims"],
                stim_list=nback_data["stim_list"],
                stim_durations=nback_data["stim_durations"],
                matches=nback_data["match_positions"],
                isi_times=nback_data["isi_times"],
                feedback_enabled=False,
                show_intro_screen=True,
                mask_enabled=True,
            )

            phase2NbackTask.show()
            phase2NbackTask.start()

            nback_performance = None

            dl.clear_trial_data()

            dl.log_multiple(
                {
                    "experiment_phase": "orientation_time_phase2",
                    "trial_number": timePhase2Loop.thisN + 1,
                    "nback_level": nback_data["n_back_level"],
                    "nback_num_stims": nback_data["num_stims"],
                }
            )

            trial_start_time = dl.log_timing_start("trial")
            nback_start_time = dl.log_timing_start("nback")

            time_phase2_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            time_phase2_trial.tStart = globalClock.getTime(format="float")
            time_phase2_trial.status = STARTED
            thisExp.addData("time_phase2_trial.started", time_phase2_trial.tStart)
            time_phase2_trial.maxDuration = None

            time_phase2_trialComponents = time_phase2_trial.components
            for thisComponent in time_phase2_trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "time_phase2_trial" ---
            if (
                isinstance(timePhase2Loop, data.TrialHandler2)
                and thisTimePhase2Loop.thisN != timePhase2Loop.thisTrial.thisN
            ):
                continueRoutine = False
            time_phase2_trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                phase2NbackTask.draw()
                status = phase2NbackTask.check_key()

                if status == "end":
                    nback_end_time = dl.log_timing_end("nback", nback_start_time)

                    performance = phase2NbackTask.get_performance()

                    dl.log_multiple(
                        {
                            "nback_stim_list": nback_data["stim_list"],
                            "nback_match_positions": nback_data["match_positions"],
                            "nback_participant_responses": [
                                i
                                for i, resp in enumerate(phase2NbackTask.responses)
                                if resp
                            ],
                            "nback_hits": performance["hits"],
                            "nback_misses": performance["misses"],
                            "nback_false_alarms": performance["false_alarms"],
                            "nback_correct_rejections": performance[
                                "correct_rejections"
                            ],
                            "nback_accuracy": performance["accuracy"],
                            "nback_hit_rate": (
                                performance["hits"]
                                / (performance["hits"] + performance["misses"])
                                if (performance["hits"] + performance["misses"]) > 0
                                else 0
                            ),
                            "nback_false_alarm_rate": (
                                performance["false_alarms"]
                                / (
                                    performance["false_alarms"]
                                    + performance["correct_rejections"]
                                )
                                if (
                                    performance["false_alarms"]
                                    + performance["correct_rejections"]
                                )
                                > 0
                                else 0
                            ),
                            "nback_num_stims": performance["num_stims"],
                        }
                    )

                    nback_performance = performance
                    phase2NbackTask.hide()
                    continueRoutine = False

                if (
                    dummy_nback_trial.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_nback_trial.frameNStart = frameN
                    dummy_nback_trial.tStart = t
                    dummy_nback_trial.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_nback_trial, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_nback_trial.started")
                    dummy_nback_trial.status = STARTED
                    dummy_nback_trial.setAutoDraw(True)

                if dummy_nback_trial.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    time_phase2_trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in time_phase2_trial.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "time_phase2_trial" ---
            for thisComponent in time_phase2_trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            trial_end_time = dl.log_timing_end("trial", trial_start_time)

            time_phase2_trial.tStop = globalClock.getTime(format="float")
            time_phase2_trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData("time_phase2_trial.stopped", time_phase2_trial.tStop)
            routineTimer.reset()

            # --- Prepare to start Routine "time_phase2_estimation" ---
            time_phase2_estimation = data.Routine(
                name="time_phase2_estimation",
                components=[dummy_typing_trial],
            )
            time_phase2_estimation.status = NOT_STARTED
            continueRoutine = True

            phase2_slider = TimeSlider(win=win)
            phase2TimeTask = TimeEstimationTask(
                win=win,
                input_device=phase2_slider,
                mouse=mouse,
                ok_rect=ok_button_rect,
                ok_text=ok_button_text,
            )
            phase2TimeTask.reset()
            phase2TimeTask.show()

            time_phase2_estimation.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            time_phase2_estimation.tStart = globalClock.getTime(format="float")
            time_phase2_estimation.status = STARTED
            thisExp.addData(
                "time_phase2_estimation.started", time_phase2_estimation.tStart
            )
            time_phase2_estimation.maxDuration = None

            time_phase2_estimationComponents = time_phase2_estimation.components
            for thisComponent in time_phase2_estimation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "time_phase2_estimation" ---
            if (
                isinstance(timePhase2Loop, data.TrialHandler2)
                and thisTimePhase2Loop.thisN != timePhase2Loop.thisTrial.thisN
            ):
                continueRoutine = False
            time_phase2_estimation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                phase2TimeTask.draw()
                status = phase2TimeTask.check_input()

                if status == "submitted":
                    time_estimate_string = phase2TimeTask.get_response()
                    minutes, seconds = phase2TimeTask.time_display.get_time()
                    estimated_seconds = minutes * 60 + seconds
                    phase2TimeTask.hide()
                    continueRoutine = False

                if (
                    dummy_typing_trial.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_typing_trial.frameNStart = frameN
                    dummy_typing_trial.tStart = t
                    dummy_typing_trial.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_typing_trial, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_typing_trial.started")
                    dummy_typing_trial.status = STARTED
                    dummy_typing_trial.setAutoDraw(True)

                if dummy_typing_trial.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    time_phase2_estimation.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in time_phase2_estimation.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "time_phase2_estimation" ---
            for thisComponent in time_phase2_estimation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            dl.log_multiple(
                {
                    "time_estimate_input": time_estimate_string,
                    "time_estimate_seconds": estimated_seconds,
                    "time_input_method": "slider",
                }
            )

            current_data = dl.get_current_data()
            actual_trial_duration = (
                current_data["cm_trial_end_timestamp"]
                - current_data["cm_trial_start_timestamp"]
            )

            # Phase 2: time on primary task equals nback duration
            time_on_primary_task = current_data.get("cm_nback_duration", 0)

            time_estimation_ratio = (
                estimated_seconds / actual_trial_duration
                if actual_trial_duration > 0
                else 0
            )
            time_estimation_error = estimated_seconds - actual_trial_duration
            time_estimation_accuracy = (
                abs(time_estimation_error) / actual_trial_duration
                if actual_trial_duration > 0
                else 0
            )

            dl.log_multiple(
                {
                    "actual_trial_duration": actual_trial_duration,
                    "total_trial_duration": actual_trial_duration,
                    "time_on_primary_task": time_on_primary_task,
                    "time_estimation_ratio": time_estimation_ratio,
                    "time_estimation_error": time_estimation_error,
                    "time_estimation_accuracy": time_estimation_accuracy,
                    "trial_completed": True,
                }
            )

            phase2_actual_duration = actual_trial_duration
            phase2_estimated_seconds = estimated_seconds

            time_phase2_estimation.tStop = globalClock.getTime(format="float")
            time_phase2_estimation.tStopRefresh = tThisFlipGlobal
            thisExp.addData(
                "time_phase2_estimation.stopped", time_phase2_estimation.tStop
            )
            routineTimer.reset()

            # --- Prepare to start Routine "time_phase2_feedback" ---
            time_phase2_feedback = data.Routine(
                name="time_phase2_feedback",
                components=[dummy_feedback],
            )
            time_phase2_feedback.status = NOT_STARTED
            continueRoutine = True

            phase2FeedbackScreen = TimingTaskFeedbackScreen(win=win)
            phase2FeedbackScreen.set_nback_feedback(nback_performance)
            phase2FeedbackScreen.set_timing_feedback(
                phase2_actual_duration,
                phase2_estimated_seconds,
            )
            phase2FeedbackScreen.show()

            time_phase2_feedback.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            time_phase2_feedback.tStart = globalClock.getTime(format="float")
            time_phase2_feedback.status = STARTED
            thisExp.addData("time_phase2_feedback.started", time_phase2_feedback.tStart)
            time_phase2_feedback.maxDuration = None

            time_phase2_feedbackComponents = time_phase2_feedback.components
            for thisComponent in time_phase2_feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "time_phase2_feedback" ---
            if (
                isinstance(timePhase2Loop, data.TrialHandler2)
                and thisTimePhase2Loop.thisN != timePhase2Loop.thisTrial.thisN
            ):
                continueRoutine = False
            time_phase2_feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                phase2FeedbackScreen.draw()
                nav_result = phase2FeedbackScreen.check_navigation(
                    mouse, defaultKeyboard
                )
                if nav_result == "escape":
                    thisExp.status = FINISHED
                    endExperiment(thisExp, win=win)
                    return
                elif nav_result:
                    continueRoutine = False

                if (
                    dummy_feedback.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_feedback.frameNStart = frameN
                    dummy_feedback.tStart = t
                    dummy_feedback.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_feedback, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_feedback.started")
                    dummy_feedback.status = STARTED
                    dummy_feedback.setAutoDraw(True)

                if dummy_feedback.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    time_phase2_feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in time_phase2_feedback.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "time_phase2_feedback" ---
            for thisComponent in time_phase2_feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            phase2FeedbackScreen.hide()

            time_phase2_feedback.tStop = globalClock.getTime(format="float")
            time_phase2_feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData("time_phase2_feedback.stopped", time_phase2_feedback.tStop)
            routineTimer.reset()
            thisExp.nextEntry()

        # ------ SUB-PHASE 3: COMBINED TASKS + TIME ESTIMATION ------

        # --- Prepare to start Routine "time_phase3_intro" ---
        time_phase3_intro = data.Routine(
            name="time_phase3_intro",
            components=[dummy_typing_intro],
        )
        time_phase3_intro.status = NOT_STARTED
        continueRoutine = True

        phase3IntroScreen = InstructionScreen(
            win=win,
            title=TIME_PHASE3_TITLE,
            content=TIME_PHASE3_CONTENT,
            button_text="Start Practice",
            button_width=0.3,
        )
        phase3IntroScreen.show()

        time_phase3_intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        time_phase3_intro.tStart = globalClock.getTime(format="float")
        time_phase3_intro.status = STARTED
        thisExp.addData("time_phase3_intro.started", time_phase3_intro.tStart)
        time_phase3_intro.maxDuration = None

        time_phase3_introComponents = time_phase3_intro.components
        for thisComponent in time_phase3_intro.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "time_phase3_intro" ---
        time_phase3_intro.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            phase3IntroScreen.draw()
            nav_result = phase3IntroScreen.check_navigation(mouse, defaultKeyboard)
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_typing_intro.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_typing_intro.frameNStart = frameN
                dummy_typing_intro.tStart = t
                dummy_typing_intro.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_typing_intro, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_typing_intro.started")
                dummy_typing_intro.status = STARTED
                dummy_typing_intro.setAutoDraw(True)

            if dummy_typing_intro.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                time_phase3_intro.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in time_phase3_intro.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "time_phase3_intro" ---
        for thisComponent in time_phase3_intro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        phase3IntroScreen.hide()

        time_phase3_intro.tStop = globalClock.getTime(format="float")
        time_phase3_intro.tStopRefresh = tThisFlipGlobal
        thisExp.addData("time_phase3_intro.stopped", time_phase3_intro.tStop)
        routineTimer.reset()
        thisExp.nextEntry()

        timePhase3Loop = data.TrialHandler2(
            name="timePhase3Loop",
            nReps=4,
            method="sequential",
            extraInfo=expInfo,
            originPath=-1,
            trialList=[None],
            seed=None,
        )
        thisExp.addLoop(timePhase3Loop)
        thisTimePhase3Loop = timePhase3Loop.trialList[0]

        if thisTimePhase3Loop != None:
            for paramName in thisTimePhase3Loop:
                globals()[paramName] = thisTimePhase3Loop[paramName]

        for thisTimePhase3Loop in timePhase3Loop:
            currentLoop = timePhase3Loop
            thisExp.timestampOnFlip(win, "thisRow.t")

            if thisTimePhase3Loop != None:
                for paramName in thisTimePhase3Loop:
                    globals()[paramName] = thisTimePhase3Loop[paramName]

            # --- Prepare to start Routine "time_phase3_trial" ---
            time_phase3_trial = data.Routine(
                name="time_phase3_trial",
                components=[dummy_typing_trial],
            )
            time_phase3_trial.status = NOT_STARTED
            continueRoutine = True

            trial_index = timePhase3Loop.thisN + 2
            current_word = ORIENTATION_TIME_ESTIMATION_WORDS[trial_index]

            phase3_conditions = [
                "interrupted",
                "sequential",
                "sequential",
                "interrupted",
            ]
            phase3_condition = phase3_conditions[timePhase3Loop.thisN]

            nback_type = PHASE3_NBACK_PATTERN[timePhase3Loop.thisN]
            sequence_index = PHASE3_SEQUENCE_INDICES[timePhase3Loop.thisN] - 1
            nback_data = ORIENTATION_TIME_NBACK_DATA[nback_type][sequence_index]

            if phase3_condition == "interrupted":
                interrupt_position = PHASE3_INTERRUPT_POSITIONS[timePhase3Loop.thisN]
            else:
                interrupt_position = []

            phase3_keyboard = Keyboard(win=win)
            phase3PrimaryTask = PrimaryTask(
                win=win,
                keyboard=phase3_keyboard,
                mouse=mouse,
                mode="hard",
                ok_rect=ok_button_rect,
                ok_text=ok_button_text,
            )

            phase3TrialManager = TrialManager(interrupt_position, [nback_data])

            phase3PrimaryTask.set_target_word(current_word)
            phase3PrimaryTask.reset()
            phase3PrimaryTask.show()

            primary_completed = False
            nback_completed = False
            typing_performance = None
            nback_performance = None

            show_nback_next = False

            interruptionSequence = "none"
            interruptionTimer = core.Clock()

            dl.clear_trial_data()

            dl.log_multiple(
                {
                    "experiment_phase": "orientation_time_phase3",
                    "trial_number": timePhase3Loop.thisN + 1,
                    "target_word": current_word,
                    "target_word_length": len(current_word),
                    "typing_mode": "hard",
                    "interruption_condition": phase3_condition,
                    "nback_level": nback_data["n_back_level"],
                    "interrupt_positions": (
                        interrupt_position if phase3_condition == "interrupted" else []
                    ),
                }
            )

            trial_start_time = dl.log_timing_start("trial")
            typing_start_time = dl.log_timing_start("typing")

            typing_pause_time = None
            typing_resume_time = None
            nback_start_time = None
            nback_end_time = None

            time_phase3_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            time_phase3_trial.tStart = globalClock.getTime(format="float")
            time_phase3_trial.status = STARTED
            thisExp.addData("time_phase3_trial.started", time_phase3_trial.tStart)
            time_phase3_trial.maxDuration = None

            time_phase3_trialComponents = time_phase3_trial.components
            for thisComponent in time_phase3_trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "time_phase3_trial" ---
            if (
                isinstance(timePhase3Loop, data.TrialHandler2)
                and thisTimePhase3Loop.thisN != timePhase3Loop.thisTrial.thisN
            ):
                continueRoutine = False
            time_phase3_trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                if phase3_condition == "interrupted":
                    if not show_nback_next:
                        phase3PrimaryTask.draw()

                        if interruptionSequence == "none":
                            status = phase3PrimaryTask.check_input()
                            phase3TrialManager.update(phase3PrimaryTask)

                            if status == "submitted":
                                typing_end_time = dl.log_timing_end(
                                    "typing", typing_start_time
                                )

                                entered_text = (
                                    phase3PrimaryTask.text_entry_screen.entered_text
                                )
                                character_accuracy = dl.calculate_character_accuracy(
                                    current_word, entered_text
                                )

                                dl.log_multiple(
                                    {
                                        "entered_text": entered_text,
                                        "entered_text_length": len(entered_text),
                                        "typing_correct": entered_text.upper()
                                        == current_word.upper(),
                                        "typing_accuracy": character_accuracy,
                                        "typing_error_rate": 1.0 - character_accuracy,
                                    }
                                )

                                typing_performance = {
                                    "target": current_word,
                                    "entered": entered_text,
                                }
                                primary_completed = True
                                phase3PrimaryTask.hide()
                                if nback_completed:
                                    continueRoutine = False
                            elif phase3TrialManager.consume():
                                typing_pause_time = globalClock.getTime()
                                dl.log("typing_pause_timestamp", typing_pause_time)
                                interruptionSequence = "masked"
                                interruptionTimer.reset()

                        elif interruptionSequence == "masked":
                            if interruptionTimer.getTime() >= INTERRUPT_DELAY:
                                interruptionSequence = "complete"
                                show_nback_next = True
                                phase3PrimaryTask.hide()

                                phase3NbackTask = NbackTask(
                                    win=win,
                                    n_back_level=nback_data["n_back_level"],
                                    num_stims=nback_data["num_stims"],
                                    stim_list=nback_data["stim_list"],
                                    stim_durations=nback_data["stim_durations"],
                                    matches=nback_data["match_positions"],
                                    isi_times=nback_data["isi_times"],
                                    feedback_enabled=False,
                                    show_intro_screen=True,
                                    mask_enabled=True,
                                )
                                phase3NbackTask.show()
                                phase3NbackTask.start()

                                nback_start_time = dl.log_timing_start("nback")
                    else:
                        phase3NbackTask.draw()
                        status = phase3NbackTask.check_key()

                        if status == "end":
                            nback_end_time = dl.log_timing_end(
                                "nback", nback_start_time
                            )

                            performance = phase3NbackTask.get_performance()

                            dl.log_multiple(
                                {
                                    "nback_stim_list": nback_data["stim_list"],
                                    "nback_match_positions": nback_data[
                                        "match_positions"
                                    ],
                                    "nback_participant_responses": [
                                        i
                                        for i, resp in enumerate(
                                            phase3NbackTask.responses
                                        )
                                        if resp
                                    ],
                                    "nback_hits": performance["hits"],
                                    "nback_misses": performance["misses"],
                                    "nback_false_alarms": performance["false_alarms"],
                                    "nback_correct_rejections": performance[
                                        "correct_rejections"
                                    ],
                                    "nback_accuracy": performance["accuracy"],
                                    "nback_hit_rate": (
                                        performance["hits"]
                                        / (performance["hits"] + performance["misses"])
                                        if (performance["hits"] + performance["misses"])
                                        > 0
                                        else 0
                                    ),
                                    "nback_false_alarm_rate": (
                                        performance["false_alarms"]
                                        / (
                                            performance["false_alarms"]
                                            + performance["correct_rejections"]
                                        )
                                        if (
                                            performance["false_alarms"]
                                            + performance["correct_rejections"]
                                        )
                                        > 0
                                        else 0
                                    ),
                                    "nback_num_stims": performance["num_stims"],
                                }
                            )

                            nback_performance = performance
                            phase3NbackTask.hide()
                            nback_completed = True

                            if primary_completed:
                                continueRoutine = False
                            else:
                                show_nback_next = False
                                interruptionSequence = "none"
                                phase3PrimaryTask.show()

                                dl.log("typing_resume_timestamp", globalClock.getTime())

                else:  # sequential mode
                    if not primary_completed:
                        phase3PrimaryTask.draw()
                        status = phase3PrimaryTask.check_input()

                        if status == "submitted":
                            typing_end_time = dl.log_timing_end(
                                "typing", typing_start_time
                            )

                            entered_text = (
                                phase3PrimaryTask.text_entry_screen.entered_text
                            )
                            character_accuracy = dl.calculate_character_accuracy(
                                current_word, entered_text
                            )

                            dl.log_multiple(
                                {
                                    "entered_text": entered_text,
                                    "entered_text_length": len(entered_text),
                                    "typing_correct": entered_text.upper()
                                    == current_word.upper(),
                                    "typing_accuracy": character_accuracy,
                                    "typing_error_rate": 1.0 - character_accuracy,
                                }
                            )

                            typing_performance = {
                                "target": current_word,
                                "entered": entered_text,
                            }
                            primary_completed = True
                            phase3PrimaryTask.hide()

                            phase3NbackTask = NbackTask(
                                win=win,
                                n_back_level=nback_data["n_back_level"],
                                num_stims=nback_data["num_stims"],
                                stim_list=nback_data["stim_list"],
                                stim_durations=nback_data["stim_durations"],
                                matches=nback_data["match_positions"],
                                isi_times=nback_data["isi_times"],
                                feedback_enabled=False,
                                show_intro_screen=True,
                                mask_enabled=True,
                            )
                            phase3NbackTask.show()
                            phase3NbackTask.start()

                            nback_start_time = dl.log_timing_start("nback")
                    else:
                        phase3NbackTask.draw()
                        status = phase3NbackTask.check_key()

                        if status == "end":
                            nback_end_time = dl.log_timing_end(
                                "nback", nback_start_time
                            )

                            performance = phase3NbackTask.get_performance()

                            dl.log_multiple(
                                {
                                    "nback_stim_list": nback_data["stim_list"],
                                    "nback_match_positions": nback_data[
                                        "match_positions"
                                    ],
                                    "nback_participant_responses": [
                                        i
                                        for i, resp in enumerate(
                                            phase3NbackTask.responses
                                        )
                                        if resp
                                    ],
                                    "nback_hits": performance["hits"],
                                    "nback_misses": performance["misses"],
                                    "nback_false_alarms": performance["false_alarms"],
                                    "nback_correct_rejections": performance[
                                        "correct_rejections"
                                    ],
                                    "nback_accuracy": performance["accuracy"],
                                    "nback_hit_rate": (
                                        performance["hits"]
                                        / (performance["hits"] + performance["misses"])
                                        if (performance["hits"] + performance["misses"])
                                        > 0
                                        else 0
                                    ),
                                    "nback_false_alarm_rate": (
                                        performance["false_alarms"]
                                        / (
                                            performance["false_alarms"]
                                            + performance["correct_rejections"]
                                        )
                                        if (
                                            performance["false_alarms"]
                                            + performance["correct_rejections"]
                                        )
                                        > 0
                                        else 0
                                    ),
                                    "nback_num_stims": performance["num_stims"],
                                }
                            )

                            nback_performance = performance
                            phase3NbackTask.hide()
                            nback_completed = True
                            continueRoutine = False

                if (
                    dummy_typing_trial.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_typing_trial.frameNStart = frameN
                    dummy_typing_trial.tStart = t
                    dummy_typing_trial.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_typing_trial, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_typing_trial.started")
                    dummy_typing_trial.status = STARTED
                    dummy_typing_trial.setAutoDraw(True)

                if dummy_typing_trial.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    time_phase3_trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in time_phase3_trial.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "time_phase3_trial" ---
            for thisComponent in time_phase3_trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            if typing_performance is None:
                typing_performance = {
                    "target": current_word,
                    "entered": phase3PrimaryTask.text_entry_screen.entered_text,
                }

            trial_end_time = dl.log_timing_end("trial", trial_start_time)

            phase3_condition_used = phase3_condition
            phase3_typing_performance = typing_performance
            phase3_nback_performance = nback_performance

            time_phase3_trial.tStop = globalClock.getTime(format="float")
            time_phase3_trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData("time_phase3_trial.stopped", time_phase3_trial.tStop)
            routineTimer.reset()

            # --- Prepare to start Routine "time_phase3_estimation" ---
            time_phase3_estimation = data.Routine(
                name="time_phase3_estimation",
                components=[dummy_typing_trial],
            )
            time_phase3_estimation.status = NOT_STARTED
            continueRoutine = True

            phase3_slider = TimeSlider(win=win)
            phase3TimeTask = TimeEstimationTask(
                win=win,
                input_device=phase3_slider,
                mouse=mouse,
                ok_rect=ok_button_rect,
                ok_text=ok_button_text,
            )
            phase3TimeTask.reset()
            phase3TimeTask.show()

            time_phase3_estimation.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            time_phase3_estimation.tStart = globalClock.getTime(format="float")
            time_phase3_estimation.status = STARTED
            thisExp.addData(
                "time_phase3_estimation.started", time_phase3_estimation.tStart
            )
            time_phase3_estimation.maxDuration = None

            time_phase3_estimationComponents = time_phase3_estimation.components
            for thisComponent in time_phase3_estimation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "time_phase3_estimation" ---
            if (
                isinstance(timePhase3Loop, data.TrialHandler2)
                and thisTimePhase3Loop.thisN != timePhase3Loop.thisTrial.thisN
            ):
                continueRoutine = False
            time_phase3_estimation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                phase3TimeTask.draw()
                status = phase3TimeTask.check_input()

                if status == "submitted":
                    time_estimate_string = phase3TimeTask.get_response()
                    minutes, seconds = phase3TimeTask.time_display.get_time()
                    estimated_seconds = minutes * 60 + seconds
                    phase3TimeTask.hide()
                    continueRoutine = False

                if (
                    dummy_typing_trial.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_typing_trial.frameNStart = frameN
                    dummy_typing_trial.tStart = t
                    dummy_typing_trial.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_typing_trial, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_typing_trial.started")
                    dummy_typing_trial.status = STARTED
                    dummy_typing_trial.setAutoDraw(True)

                if dummy_typing_trial.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    time_phase3_estimation.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in time_phase3_estimation.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "time_phase3_estimation" ---
            for thisComponent in time_phase3_estimation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            dl.log_multiple(
                {
                    "time_estimate_input": time_estimate_string,
                    "time_estimate_seconds": estimated_seconds,
                    "time_input_method": "slider",
                }
            )

            current_data = dl.get_current_data()
            actual_trial_duration = (
                current_data["cm_trial_end_timestamp"]
                - current_data["cm_trial_start_timestamp"]
            )

            if phase3_condition_used == "sequential":
                time_on_primary_task = current_data.get("cm_typing_duration", 0)
            else:  # interrupted mode
                typing_duration = current_data.get("cm_typing_duration", 0)
                nback_duration = current_data.get("cm_nback_duration", 0)

                if typing_duration > 0:
                    time_on_primary_task = typing_duration
                else:
                    typing_start = current_data.get("cm_typing_start_timestamp", 0)
                    typing_pause = current_data.get("cm_typing_pause_timestamp", 0)
                    typing_resume = current_data.get("cm_typing_resume_timestamp", 0)
                    trial_end = current_data.get("cm_trial_end_timestamp", 0)

                    if typing_pause > 0 and typing_resume > 0:
                        before_interruption = typing_pause - typing_start
                        after_resumption = trial_end - typing_resume
                        time_on_primary_task = before_interruption + after_resumption
                    else:
                        time_on_primary_task = actual_trial_duration - nback_duration

            time_estimation_ratio = (
                estimated_seconds / actual_trial_duration
                if actual_trial_duration > 0
                else 0
            )
            time_estimation_error = estimated_seconds - actual_trial_duration
            time_estimation_accuracy = (
                abs(time_estimation_error) / actual_trial_duration
                if actual_trial_duration > 0
                else 0
            )

            dl.log_multiple(
                {
                    "actual_trial_duration": actual_trial_duration,
                    "total_trial_duration": actual_trial_duration,
                    "time_on_primary_task": time_on_primary_task,
                    "time_estimation_ratio": time_estimation_ratio,
                    "time_estimation_error": time_estimation_error,
                    "time_estimation_accuracy": time_estimation_accuracy,
                    "trial_completed": True,
                }
            )

            phase3_actual_duration = actual_trial_duration
            phase3_estimated_seconds = estimated_seconds

            time_phase3_estimation.tStop = globalClock.getTime(format="float")
            time_phase3_estimation.tStopRefresh = tThisFlipGlobal
            thisExp.addData(
                "time_phase3_estimation.stopped", time_phase3_estimation.tStop
            )
            routineTimer.reset()

            # --- Prepare to start Routine "time_phase3_feedback" ---
            time_phase3_feedback = data.Routine(
                name="time_phase3_feedback",
                components=[dummy_feedback],
            )
            time_phase3_feedback.status = NOT_STARTED
            continueRoutine = True

            phase3FeedbackScreen = TimingTaskFeedbackScreen(win=win)

            phase3FeedbackScreen.title_stim.text = f"Results with Time Estimation"

            phase3FeedbackScreen.set_combined_feedback(
                phase3_typing_performance["target"],
                phase3_typing_performance["entered"],
                phase3_nback_performance,
            )

            phase3FeedbackScreen.set_timing_feedback(
                phase3_actual_duration,
                phase3_estimated_seconds,
            )

            phase3FeedbackScreen.show()

            time_phase3_feedback.tStartRefresh = win.getFutureFlipTime(
                clock=globalClock
            )
            time_phase3_feedback.tStart = globalClock.getTime(format="float")
            time_phase3_feedback.status = STARTED
            thisExp.addData("time_phase3_feedback.started", time_phase3_feedback.tStart)
            time_phase3_feedback.maxDuration = None

            time_phase3_feedbackComponents = time_phase3_feedback.components
            for thisComponent in time_phase3_feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "time_phase3_feedback" ---
            if (
                isinstance(timePhase3Loop, data.TrialHandler2)
                and thisTimePhase3Loop.thisN != timePhase3Loop.thisTrial.thisN
            ):
                continueRoutine = False
            time_phase3_feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                phase3FeedbackScreen.draw()
                nav_result = phase3FeedbackScreen.check_navigation(
                    mouse, defaultKeyboard
                )
                if nav_result == "escape":
                    thisExp.status = FINISHED
                    endExperiment(thisExp, win=win)
                    return
                elif nav_result:
                    continueRoutine = False

                if (
                    dummy_feedback.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_feedback.frameNStart = frameN
                    dummy_feedback.tStart = t
                    dummy_feedback.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_feedback, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_feedback.started")
                    dummy_feedback.status = STARTED
                    dummy_feedback.setAutoDraw(True)

                if dummy_feedback.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    time_phase3_feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in time_phase3_feedback.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "time_phase3_feedback" ---
            for thisComponent in time_phase3_feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            phase3FeedbackScreen.hide()

            time_phase3_feedback.tStop = globalClock.getTime(format="float")
            time_phase3_feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData("time_phase3_feedback.stopped", time_phase3_feedback.tStop)
            routineTimer.reset()
            thisExp.nextEntry()

        # --- Prepare to start Routine "orientation_complete" ---
        orientation_complete = data.Routine(
            name="orientation_complete",
            components=[dummy_typing_intro],
        )
        orientation_complete.status = NOT_STARTED
        continueRoutine = True

        orientationCompleteScreen = InstructionScreen(
            win=win,
            title="Orientation Complete!",
            content="""Excellent work! You've completed all orientation training.

        You've learned:
        ✓ How to type words using the keyboard
        ✓ How to perform N-back tasks
        ✓ How interruptions work
        ✓ How to estimate time for different tasks

        You're now ready for the Main Experiment, where you'll do trials
        just like the ones you experienced before.

        Take a moment to rest if needed.""",
            button_text="Continue to Main Experiment",
            button_width=0.5,
        )
        orientationCompleteScreen.show()

        orientation_complete.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        orientation_complete.tStart = globalClock.getTime(format="float")
        orientation_complete.status = STARTED
        thisExp.addData("orientation_complete.started", orientation_complete.tStart)
        orientation_complete.maxDuration = None

        orientation_completeComponents = orientation_complete.components
        for thisComponent in orientation_complete.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "orientation_complete" ---
        orientation_complete.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            orientationCompleteScreen.draw()
            nav_result = orientationCompleteScreen.check_navigation(
                mouse, defaultKeyboard
            )
            if nav_result == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
            elif nav_result:
                continueRoutine = False

            if (
                dummy_typing_intro.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_typing_intro.frameNStart = frameN
                dummy_typing_intro.tStart = t
                dummy_typing_intro.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_typing_intro, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_typing_intro.started")
                dummy_typing_intro.status = STARTED
                dummy_typing_intro.setAutoDraw(True)

            if dummy_typing_intro.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                orientation_complete.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in orientation_complete.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "orientation_complete" ---
        for thisComponent in orientation_complete.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        orientationCompleteScreen.hide()

        orientation_complete.tStop = globalClock.getTime(format="float")
        orientation_complete.tStopRefresh = tThisFlipGlobal
        thisExp.addData("orientation_complete.stopped", orientation_complete.tStop)
        routineTimer.reset()
        thisExp.nextEntry()

        # -------------------------------------------------- END OF ORIENTATION --------------------------------------------------

    # -------------------------------------------------- MAIN EXPERIMENT --------------------------------------------------

    # --- Decision Screen --------------------------

    # --- Prepare to start Routine "main_phase_decision" ---
    main_phase_decision = data.Routine(
        name="main_phase_decision",
        components=[dummy_main_decision],
    )
    main_phase_decision.status = NOT_STARTED
    continueRoutine = True

    decisionScreen = MainPhaseDecisionScreen(win=win)
    decisionScreen.show()

    main_phase_decision.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    main_phase_decision.tStart = globalClock.getTime(format="float")
    main_phase_decision.status = STARTED
    thisExp.addData("main_phase_decision.started", main_phase_decision.tStart)
    main_phase_decision.maxDuration = None

    main_phase_decisionComponents = main_phase_decision.components
    for thisComponent in main_phase_decision.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, "status"):
            thisComponent.status = NOT_STARTED

    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "main_phase_decision" ---
    main_phase_decision.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1

        decisionScreen.draw()
        nav_result = decisionScreen.check_navigation(mouse, defaultKeyboard)

        if nav_result == "escape" or nav_result == "abort":
            thisExp.addData("experiment_aborted", True)
            thisExp.addData("abort_at_decision", True)
            expInfo["experiment_aborted"] = True
            continueRoutine = False
        elif nav_result == "continue":
            thisExp.addData("experiment_aborted", False)
            expInfo["experiment_aborted"] = False
            continueRoutine = False

        if (
            dummy_main_decision.status == NOT_STARTED
            and tThisFlip >= 0.0 - frameTolerance
        ):
            dummy_main_decision.frameNStart = frameN
            dummy_main_decision.tStart = t
            dummy_main_decision.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(dummy_main_decision, "tStartRefresh")
            thisExp.timestampOnFlip(win, "dummy_main_decision.started")
            dummy_main_decision.status = STARTED
            dummy_main_decision.setAutoDraw(True)

        if dummy_main_decision.status == STARTED:
            pass

        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return

        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, win=win, timers=[routineTimer], playbackComponents=[]
            )
            continue

        if not continueRoutine:
            main_phase_decision.forceEnded = routineForceEnded = True
            break
        continueRoutine = False
        for thisComponent in main_phase_decision.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break

        if continueRoutine:
            win.flip()

    # --- Ending Routine "main_phase_decision" ---
    for thisComponent in main_phase_decision.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    decisionScreen.hide()

    main_phase_decision.tStop = globalClock.getTime(format="float")
    main_phase_decision.tStopRefresh = tThisFlipGlobal
    thisExp.addData("main_phase_decision.stopped", main_phase_decision.tStop)
    routineTimer.reset()

    if "experiment_aborted" in expInfo and expInfo["experiment_aborted"]:
        pass

    else:
        # --------------------------------------------- EXPERIMENT RUN STARTS ----------------------------------------------

        # --- Prepare to start Routine "main_run_phase" ---
        main_run_phase = data.Routine(
            name="main_run_phase",
            components=[dummy_main_run],
        )
        main_run_phase.status = NOT_STARTED
        continueRoutine = True

        if RUN_PRACTICE:
            mainRunScreen = InstructionScreen(
                win=win,
                title=MAIN_RUN_PRACTICE_TITLE,
                content=MAIN_RUN_PRACTICE_CONTENT,
                button_text="Begin",
                button_delay=6.0,
            )
            mainRunScreen.show()
        else:
            continueRoutine = False

        main_run_phase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        main_run_phase.tStart = globalClock.getTime(format="float")
        main_run_phase.status = STARTED
        thisExp.addData("main_run_phase.started", main_run_phase.tStart)
        main_run_phase.maxDuration = None

        main_run_phaseComponents = main_run_phase.components
        for thisComponent in main_run_phase.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "main_run_phase" ---
        main_run_phase.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            if RUN_PRACTICE:
                mainRunScreen.draw()
                nav_result = mainRunScreen.check_navigation(mouse, defaultKeyboard)
                if nav_result == "escape":
                    thisExp.status = FINISHED
                    endExperiment(thisExp, win=win)
                    return
                elif nav_result:
                    continueRoutine = False

            if (
                dummy_main_run.status == NOT_STARTED
                and tThisFlip >= 0.0 - frameTolerance
            ):
                dummy_main_run.frameNStart = frameN
                dummy_main_run.tStart = t
                dummy_main_run.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_main_run, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_main_run.started")
                dummy_main_run.status = STARTED
                dummy_main_run.setAutoDraw(True)

            if dummy_main_run.status == STARTED:
                pass

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                main_run_phase.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in main_run_phase.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "main_run_phase" ---
        for thisComponent in main_run_phase.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        if RUN_PRACTICE:
            mainRunScreen.hide()

        main_run_phase.tStop = globalClock.getTime(format="float")
        main_run_phase.tStopRefresh = tThisFlipGlobal
        thisExp.addData("main_run_phase.stopped", main_run_phase.tStop)
        routineTimer.reset()
        thisExp.nextEntry()

        # --- Prepare to start Routine "load_experiment_data" ---
        load_experiment_data = data.Routine(
            name="load_experiment_data",
            components=[dummy_4],
        )
        load_experiment_data.status = NOT_STARTED
        continueRoutine = True

        if "all_trial_rows" not in globals():
            import csv, ast, json, sys

            # ── PARTICIPANT-ASSIGNED CSV ─────────────────────────────────────
            csv_path = expInfo["csv_filename_full"]

            if not os.path.isfile(csv_path):
                sys.exit(f"CSV file not found: {csv_path}")

            all_trial_rows = []
            all_nback_rows = []
            practice_trial_rows = []
            practice_nback_rows = []

            with open(csv_path, newline="") as fp:
                for row in csv.DictReader(fp):
                    trial_data = {
                        "trial": int(row["trial"]),
                        "word": row["word"],
                        "letters": json.loads(row["letters"]),
                        "interrupt_pos": json.loads(row["interrupt_pos"]),
                        "interruption_condition": row["interruption_condition"],
                        "short_break": row["short_break"].lower() == "true",
                        "long_break": row["long_break"].lower() == "true",
                    }

                    nback_data = {
                        "n_back_level": int(row["n_back"]),
                        "stim_list": ast.literal_eval(row["stim_list"]),
                        "stim_durations": ast.literal_eval(row["stim_durations"]),
                        "match_positions": ast.literal_eval(row["match_positions"]),
                        "isi_times": ast.literal_eval(row["isi_times"]),
                        "num_stims": int(row["num_stims"]),
                        "total_duration": float(row["nback_total_duration"]),
                    }

                    if row["practice"].lower() == "true":
                        practice_trial_rows.append(trial_data)
                        practice_nback_rows.append(nback_data)
                    else:
                        all_trial_rows.append(trial_data)
                        all_nback_rows.append(nback_data)

            expected = NUM_BLOCKS * NUM_TRIALS
            if len(all_trial_rows) != expected:
                sys.exit(
                    f"CSV main trials = {len(all_trial_rows)} but NUM_BLOCKS×NUM_TRIALS = {expected}"
                )

            if len(practice_trial_rows) != NUM_PRACTICE_TRIALS:
                sys.exit(
                    f"CSV practice trials = {len(practice_trial_rows)} but NUM_PRACTICE_TRIALS = {NUM_PRACTICE_TRIALS}"
                )

            for k, r in enumerate(all_trial_rows, 1):
                if (
                    len(r["interrupt_pos"]) != 0
                    and len(r["interrupt_pos"]) != NUM_INTERRUPTIONS
                ):
                    sys.exit(f"Main trial row {k}: interrupt_pos ≠ NUM_INTERRUPTIONS")

            for k, r in enumerate(practice_trial_rows, 1):
                if (
                    len(r["interrupt_pos"]) != 0
                    and len(r["interrupt_pos"]) != NUM_INTERRUPTIONS
                ):
                    sys.exit(
                        f"Practice trial row {k}: interrupt_pos ≠ NUM_INTERRUPTIONS"
                    )

            required_main_nback = NUM_BLOCKS * NUM_TRIALS * NUM_INTERRUPTIONS
            if len(all_nback_rows) < required_main_nback:
                sys.exit(
                    f"Need ≥ {required_main_nback} main n-back rows "
                    f"but found {len(all_nback_rows)}"
                )

            required_practice_nback = NUM_PRACTICE_TRIALS * NUM_INTERRUPTIONS
            if len(practice_nback_rows) < required_practice_nback:
                sys.exit(
                    f"Need ≥ {required_practice_nback} practice n-back rows "
                    f"but found {len(practice_nback_rows)}"
                )

        load_experiment_data.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        load_experiment_data.tStart = globalClock.getTime(format="float")
        load_experiment_data.status = STARTED
        thisExp.addData("load_experiment_data.started", load_experiment_data.tStart)
        load_experiment_data.maxDuration = None

        load_experiment_dataComponents = load_experiment_data.components
        for thisComponent in load_experiment_data.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, "status"):
                thisComponent.status = NOT_STARTED

        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "load_experiment_data" ---
        load_experiment_data.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1

            if dummy_4.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                dummy_4.frameNStart = frameN
                dummy_4.tStart = t
                dummy_4.tStartRefresh = tThisFlipGlobal
                win.timeOnFlip(dummy_4, "tStartRefresh")
                thisExp.timestampOnFlip(win, "dummy_4.started")
                dummy_4.status = STARTED
                dummy_4.setAutoDraw(True)

            if dummy_4.status == STARTED:
                pass

            if dummy_4.status == STARTED:
                if tThisFlipGlobal > dummy_4.tStartRefresh + 1 - frameTolerance:
                    dummy_4.tStop = t
                    dummy_4.tStopRefresh = tThisFlipGlobal
                    dummy_4.frameNStop = frameN
                    thisExp.timestampOnFlip(win, "dummy_4.stopped")
                    dummy_4.status = FINISHED
                    dummy_4.setAutoDraw(False)

            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[],
                )
                continue

            if not continueRoutine:
                load_experiment_data.forceEnded = routineForceEnded = True
                break
            continueRoutine = False
            for thisComponent in load_experiment_data.components:
                if (
                    hasattr(thisComponent, "status")
                    and thisComponent.status != FINISHED
                ):
                    continueRoutine = True
                    break

            if continueRoutine:
                win.flip()

        # --- Ending Routine "load_experiment_data" ---
        for thisComponent in load_experiment_data.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)

        load_experiment_data.tStop = globalClock.getTime(format="float")
        load_experiment_data.tStopRefresh = tThisFlipGlobal
        thisExp.addData("load_experiment_data.stopped", load_experiment_data.tStop)

        if load_experiment_data.maxDurationReached:
            routineTimer.addTime(-load_experiment_data.maxDuration)
        elif load_experiment_data.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()

        total_iterations = NUM_BLOCKS + (1 if RUN_PRACTICE else 0)

        experimentLoop = data.TrialHandler2(
            name="experimentLoop",
            nReps=total_iterations,
            method="sequential",
            extraInfo=expInfo,
            originPath=-1,
            trialList=[None],
            seed=None,
        )

        thisExp.addLoop(experimentLoop)
        thisExperimentLoop = experimentLoop.trialList[0]

        if thisExperimentLoop != None:
            for paramName in thisExperimentLoop:
                globals()[paramName] = thisExperimentLoop[paramName]

        for thisExperimentLoop in experimentLoop:
            currentLoop = experimentLoop
            thisExp.timestampOnFlip(win, "thisRow.t")

            if thisExperimentLoop != None:
                for paramName in thisExperimentLoop:
                    globals()[paramName] = thisExperimentLoop[paramName]

            # --- Prepare to start Routine "main_phase_intro" ---
            main_phase_intro = data.Routine(
                name="main_phase_intro",
                components=[dummy_main_phase_intro],
            )
            main_phase_intro.status = NOT_STARTED
            continueRoutine = True

            is_main_phase_start = RUN_PRACTICE and experimentLoop.thisN == 1

            if is_main_phase_start:
                mainPhaseScreen = InstructionScreen(
                    win=win,
                    title=MAIN_RUN_MAIN_TITLE,
                    content=MAIN_RUN_MAIN_CONTENT,
                    button_text="Begin",
                    button_delay=6.0,
                )
                mainPhaseScreen.show()
            else:
                continueRoutine = False

            main_phase_intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            main_phase_intro.tStart = globalClock.getTime(format="float")
            main_phase_intro.status = STARTED
            thisExp.addData("main_phase_intro.started", main_phase_intro.tStart)
            main_phase_intro.maxDuration = None

            main_phase_introComponents = main_phase_intro.components
            for thisComponent in main_phase_intro.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "main_phase_intro" ---
            main_phase_intro.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                if is_main_phase_start:
                    mainPhaseScreen.draw()
                    nav_result = mainPhaseScreen.check_navigation(
                        mouse, defaultKeyboard
                    )
                    if nav_result == "escape":
                        thisExp.status = FINISHED
                        endExperiment(thisExp, win=win)
                        return
                    elif nav_result:
                        continueRoutine = False

                if (
                    dummy_main_phase_intro.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    dummy_main_phase_intro.frameNStart = frameN
                    dummy_main_phase_intro.tStart = t
                    dummy_main_phase_intro.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(dummy_main_phase_intro, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "dummy_main_phase_intro.started")
                    dummy_main_phase_intro.status = STARTED
                    dummy_main_phase_intro.setAutoDraw(True)

                if dummy_main_phase_intro.status == STARTED:
                    pass

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    main_phase_intro.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in main_phase_intro.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "main_phase_intro" ---
            for thisComponent in main_phase_intro.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            if "mainPhaseScreen" in locals() and is_main_phase_start:
                mainPhaseScreen.hide()

            main_phase_intro.tStop = globalClock.getTime(format="float")
            main_phase_intro.tStopRefresh = tThisFlipGlobal
            thisExp.addData("main_phase_intro.stopped", main_phase_intro.tStop)
            routineTimer.reset()
            thisExp.nextEntry()

            # --- Prepare to start Routine "load_block_data" ---
            load_block_data = data.Routine(
                name="load_block_data",
                components=[],
            )
            load_block_data.status = NOT_STARTED
            continueRoutine = True

            is_practice_block = RUN_PRACTICE and experimentLoop.thisN == 0

            if is_practice_block:
                if len(practice_nback_rows) > 0:
                    thisBlockNBackLevel = practice_nback_rows[0]["n_back_level"]

                print(f"Practice Block: Using {thisBlockNBackLevel}-back conditions")
            else:
                main_block_index = experimentLoop.thisN - (1 if RUN_PRACTICE else 0)
                thisBlockNBackLevel = main_block_index + 1
                print(
                    f"Block {main_block_index + 1}: Using {thisBlockNBackLevel}-back conditions"
                )

            load_block_data.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            load_block_data.tStart = globalClock.getTime(format="float")
            load_block_data.status = STARTED
            thisExp.addData("load_block_data.started", load_block_data.tStart)
            load_block_data.maxDuration = None

            load_block_dataComponents = load_block_data.components
            for thisComponent in load_block_data.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "load_block_data" ---
            if (
                isinstance(experimentLoop, data.TrialHandler2)
                and thisExperimentLoop.thisN != experimentLoop.thisTrial.thisN
            ):
                continueRoutine = False
            load_block_data.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    load_block_data.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in load_block_data.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "load_block_data" ---
            for thisComponent in load_block_data.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            load_block_data.tStop = globalClock.getTime(format="float")
            load_block_data.tStopRefresh = tThisFlipGlobal
            thisExp.addData("load_block_data.stopped", load_block_data.tStop)

            if is_practice_block:
                num_trials_this_block = NUM_PRACTICE_TRIALS
            else:
                num_trials_this_block = NUM_TRIALS

            routineTimer.reset()

            # --- Prepare to start Routine "block_instructions" ---
            block_instructions = data.Routine(
                name="block_instructions",
                components=[block_text],
            )

            block_instructions.status = NOT_STARTED
            continueRoutine = True

            is_practice_block = RUN_PRACTICE and experimentLoop.thisN == 0

            if is_practice_block:
                show_instruction_screen = False
                continueRoutine = False
            else:
                main_block_num = experimentLoop.thisN - (1 if RUN_PRACTICE else 0) + 1
                blocks_remaining = NUM_BLOCKS - main_block_num + 1

                remaining_text = (
                    f"Almost done! Just {blocks_remaining} block"
                    if blocks_remaining == 1
                    else f"{blocks_remaining} blocks"
                )

                if main_block_num == 1:
                    blockScreen = InstructionScreen(
                        win=win,
                        title=MAIN_BLOCK_TITLE.format(main_block_num, NUM_BLOCKS),
                        content=MAIN_BLOCK_CONTENT_FIRST.format(
                            main_block_num, NUM_BLOCKS, remaining_text
                        ),
                        button_text=BLOCK_START_BUTTON,
                        button_delay=6.0,
                    )
                else:
                    blockScreen = InstructionScreen(
                        win=win,
                        title=MAIN_BLOCK_TITLE.format(main_block_num, NUM_BLOCKS),
                        content=MAIN_BLOCK_CONTENT.format(
                            main_block_num, NUM_BLOCKS, remaining_text
                        ),
                        button_text=BLOCK_START_BUTTON,
                        button_delay=30.0,
                    )

                blockScreen.show()
                show_instruction_screen = True

            continueRoutine = show_instruction_screen

            block_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            block_instructions.tStart = globalClock.getTime(format="float")
            block_instructions.status = STARTED
            thisExp.addData("block_instructions.started", block_instructions.tStart)
            block_instructions.maxDuration = None

            block_instructionsComponents = block_instructions.components
            for thisComponent in block_instructions.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, "status"):
                    thisComponent.status = NOT_STARTED

            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "block_instructions" ---
            if (
                isinstance(experimentLoop, data.TrialHandler2)
                and thisExperimentLoop.thisN != experimentLoop.thisTrial.thisN
            ):
                continueRoutine = False
            block_instructions.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1

                if show_instruction_screen and "blockScreen" in locals():
                    blockScreen.draw()
                    nav_result = blockScreen.check_navigation(mouse, defaultKeyboard)

                    if nav_result == "escape":
                        thisExp.status = FINISHED
                        endExperiment(thisExp, win=win)
                        return
                    elif nav_result:
                        continueRoutine = False

                if (
                    not show_instruction_screen
                    and block_text.status == NOT_STARTED
                    and tThisFlip >= 0.0 - frameTolerance
                ):
                    block_text.frameNStart = frameN
                    block_text.tStart = t
                    block_text.tStartRefresh = tThisFlipGlobal
                    win.timeOnFlip(block_text, "tStartRefresh")
                    thisExp.timestampOnFlip(win, "block_text.started")
                    block_text.status = STARTED
                    block_text.setAutoDraw(True)

                if block_text.status == STARTED:
                    pass

                if block_text.status == STARTED:
                    if (
                        tThisFlipGlobal
                        > block_text.tStartRefresh + 1.0 - frameTolerance
                    ):
                        block_text.tStop = t
                        block_text.tStopRefresh = tThisFlipGlobal
                        block_text.frameNStop = frameN
                        thisExp.timestampOnFlip(win, "block_text.stopped")
                        block_text.status = FINISHED
                        block_text.setAutoDraw(False)
                        continueRoutine = False

                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[],
                    )
                    continue

                if not continueRoutine:
                    block_instructions.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False
                for thisComponent in block_instructions.components:
                    if (
                        hasattr(thisComponent, "status")
                        and thisComponent.status != FINISHED
                    ):
                        continueRoutine = True
                        break

                if continueRoutine:
                    win.flip()

            # --- Ending Routine "block_instructions" ---
            for thisComponent in block_instructions.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            if "mainPhaseScreen" in locals() and is_main_phase_start:
                mainPhaseScreen.hide()

            if "blockScreen" in locals() and show_instruction_screen:
                blockScreen.hide()

            block_instructions.tStop = globalClock.getTime(format="float")
            block_instructions.tStopRefresh = tThisFlipGlobal
            thisExp.addData("block_instructions.stopped", block_instructions.tStop)

            if block_instructions.maxDurationReached:
                routineTimer.addTime(-block_instructions.maxDuration)
            elif block_instructions.forceEnded:
                routineTimer.reset()
            else:
                if not show_instruction_screen:
                    routineTimer.addTime(-1.000000)

            blockLoop = data.TrialHandler2(
                name="blockLoop",
                nReps=num_trials_this_block,
                method="sequential",
                extraInfo=expInfo,
                originPath=-1,
                trialList=[None],
                seed=None,
            )
            thisExp.addLoop(blockLoop)
            thisBlockLoop = blockLoop.trialList[0]

            if thisBlockLoop != None:
                for paramName in thisBlockLoop:
                    globals()[paramName] = thisBlockLoop[paramName]
            if thisSession is not None:
                thisSession.sendExperimentData()

            for thisBlockLoop in blockLoop:
                currentLoop = blockLoop
                thisExp.timestampOnFlip(win, "thisRow.t")
                if thisSession is not None:
                    thisSession.sendExperimentData()

                if thisBlockLoop != None:
                    for paramName in thisBlockLoop:
                        globals()[paramName] = thisBlockLoop[paramName]

                # --- Prepare to start Routine "load_trial_data" ---
                load_trial_data = data.Routine(
                    name="load_trial_data",
                    components=[],
                )
                load_trial_data.status = NOT_STARTED
                continueRoutine = True

                trial_start_time = None
                trial_end_time = None
                typing_start_time = None
                typing_end_time = None
                typing_pause_time = None
                typing_resume_time = None
                nback_start_time = None
                nback_end_time = None

                nback_duration = 0
                time_estimate_string = None
                time_estimate_seconds = None
                actual_trial_duration = None
                time_on_typing_task = None

                """
                Runs once per *blockLoop* iteration.
                Each block ≙ one logical trial in which the primary task can be
                interrupted NUM_INTERRUPTIONS times (handled by the inner trialLoop).
                Output (globals for the upcoming routines)
                ------------------------------------------
                ▪ thisTrialWord        – string, the target word to be entered
                ▪ thisTrialLetters     – list[str], individual letters of the word
                ▪ trialManager         – controller for this trial
                ▪ thisTrialNbackData    – list of n-back parameters for each interruption
                """

                if is_practice_block:
                    current_trial_rows = practice_trial_rows
                    current_nback_rows = practice_nback_rows
                    row_index = blockLoop.thisN
                else:
                    current_trial_rows = all_trial_rows
                    current_nback_rows = all_nback_rows
                    main_block_index = experimentLoop.thisN - (1 if RUN_PRACTICE else 0)
                    row_index = main_block_index * NUM_TRIALS + blockLoop.thisN

                current_row = current_trial_rows[row_index]

                thisTrialWord = current_row["word"]
                thisTrialLetters = current_row["letters"]

                if "interruption_condition" in current_row:
                    INTERRUPTION_CONDITION = current_row["interruption_condition"]
                    print(
                        f"Trial {blockLoop.thisN + 1}: Using {INTERRUPTION_CONDITION} condition"
                    )
                else:
                    print(
                        f"Warning: No interruption_condition specified in CSV, using default: {INTERRUPTION_CONDITION}"
                    )

                if is_practice_block:
                    nback_start_idx = blockLoop.thisN * NUM_INTERRUPTIONS
                    nback_rows_to_use = current_nback_rows
                else:
                    main_block_index = experimentLoop.thisN - (1 if RUN_PRACTICE else 0)
                    block_offset = main_block_index * (NUM_TRIALS * NUM_INTERRUPTIONS)
                    trial_offset = blockLoop.thisN * NUM_INTERRUPTIONS
                    nback_start_idx = block_offset + trial_offset
                    nback_rows_to_use = current_nback_rows

                thisTrialNbackData = []
                for i in range(NUM_INTERRUPTIONS):
                    nback_row = nback_rows_to_use[nback_start_idx + i]
                    thisTrialNbackData.append(
                        {
                            "stim_list": nback_row["stim_list"],
                            "stim_durations": nback_row["stim_durations"],
                            "match_positions": nback_row["match_positions"],
                            "isi_times": nback_row["isi_times"],
                            "num_stims": nback_row["num_stims"],
                            "total_duration": nback_row["total_duration"],
                            "n_back_level": nback_row["n_back_level"],
                        }
                    )

                if "trialManager" not in globals():
                    trialManager = TrialManager(
                        current_row["interrupt_pos"], thisTrialNbackData
                    )
                else:
                    trialManager.reset(current_row["interrupt_pos"], thisTrialNbackData)

                if INTERRUPTION_CONDITION == "sequential" and NUM_INTERRUPTIONS > 1:
                    sys.exit(
                        "ERROR: Sequential condition cannot have more than 1 interruption. "
                        "Set NUM_INTERRUPTIONS to 1 or change INTERRUPTION_CONDITION to 'interrupted'."
                    )

                load_trial_data.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                load_trial_data.tStart = globalClock.getTime(format="float")
                load_trial_data.status = STARTED
                thisExp.addData("load_trial_data.started", load_trial_data.tStart)
                load_trial_data.maxDuration = None

                load_trial_dataComponents = load_trial_data.components
                for thisComponent in load_trial_data.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, "status"):
                        thisComponent.status = NOT_STARTED

                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1

                # --- Run Routine "load_trial_data" ---
                if (
                    isinstance(blockLoop, data.TrialHandler2)
                    and thisBlockLoop.thisN != blockLoop.thisTrial.thisN
                ):
                    continueRoutine = False
                load_trial_data.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1

                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return

                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp,
                            win=win,
                            timers=[routineTimer],
                            playbackComponents=[],
                        )
                        continue

                    if not continueRoutine:
                        load_trial_data.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False
                    for thisComponent in load_trial_data.components:
                        if (
                            hasattr(thisComponent, "status")
                            and thisComponent.status != FINISHED
                        ):
                            continueRoutine = True
                            break

                    if continueRoutine:
                        win.flip()

                # --- Ending Routine "load_trial_data" ---
                for thisComponent in load_trial_data.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)

                load_trial_data.tStop = globalClock.getTime(format="float")
                load_trial_data.tStopRefresh = tThisFlipGlobal
                thisExp.addData("load_trial_data.stopped", load_trial_data.tStop)

                dl.clear_trial_data()

                if is_practice_block:
                    global_trial_num = blockLoop.thisN + 1
                else:
                    practice_offset = NUM_PRACTICE_TRIALS if RUN_PRACTICE else 0
                    main_block_index = experimentLoop.thisN - (1 if RUN_PRACTICE else 0)
                    global_trial_num = (
                        practice_offset
                        + (main_block_index * NUM_TRIALS)
                        + (blockLoop.thisN + 1)
                    )

                dl.log_multiple(
                    {
                        "experiment_phase": "practice" if is_practice_block else "main",
                        "block_number": (
                            1
                            if is_practice_block
                            else (experimentLoop.thisN - (1 if RUN_PRACTICE else 0) + 1)
                        ),
                        "trial_number": blockLoop.thisN + 1,
                        "global_trial_number": global_trial_num,
                        "nback_level": thisBlockNBackLevel,
                        "interruption_condition": INTERRUPTION_CONDITION,
                        "num_interruptions": NUM_INTERRUPTIONS,
                        "interrupt_positions": current_row["interrupt_pos"],
                        "target_word": thisTrialWord,
                        "target_word_length": len(thisTrialWord),
                        "typing_mode": PRIMARY_DIFFICULTY,
                    }
                )
                routineTimer.reset()

                # --- Prepare to start Routine "trial_instructions" ---
                trial_instructions = data.Routine(
                    name="trial_instructions",
                    components=[trial_text],
                )

                trial_instructions.status = NOT_STARTED
                continueRoutine = True

                if is_practice_block:
                    trial_text.setText(
                        f"Practice trial {blockLoop.thisN + 1} out of {NUM_PRACTICE_TRIALS}"
                    )
                else:
                    trial_text.setText(
                        f"This is trial {blockLoop.thisN + 1} out of {NUM_TRIALS}"
                    )

                if not SHOW_TRIAL_INSTRUCTIONS:
                    continueRoutine = False

                trial_instructions.tStartRefresh = win.getFutureFlipTime(
                    clock=globalClock
                )
                trial_instructions.tStart = globalClock.getTime(format="float")
                trial_instructions.status = STARTED
                thisExp.addData("trial_instructions.started", trial_instructions.tStart)
                trial_instructions.maxDuration = None

                trial_instructionsComponents = trial_instructions.components
                for thisComponent in trial_instructions.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, "status"):
                        thisComponent.status = NOT_STARTED

                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1

                # --- Run Routine "trial_instructions" ---
                if (
                    isinstance(blockLoop, data.TrialHandler2)
                    and thisBlockLoop.thisN != blockLoop.thisTrial.thisN
                ):
                    continueRoutine = False
                trial_instructions.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 1.0:
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1

                    if (
                        trial_text.status == NOT_STARTED
                        and tThisFlip >= 0.0 - frameTolerance
                    ):
                        trial_text.frameNStart = frameN
                        trial_text.tStart = t
                        trial_text.tStartRefresh = tThisFlipGlobal
                        win.timeOnFlip(trial_text, "tStartRefresh")
                        thisExp.timestampOnFlip(win, "trial_text.started")
                        trial_text.status = STARTED
                        trial_text.setAutoDraw(True)

                    if trial_text.status == STARTED:
                        pass

                    if trial_text.status == STARTED:
                        if (
                            tThisFlipGlobal
                            > trial_text.tStartRefresh + 1.0 - frameTolerance
                        ):
                            trial_text.tStop = t
                            trial_text.tStopRefresh = tThisFlipGlobal
                            trial_text.frameNStop = frameN
                            thisExp.timestampOnFlip(win, "trial_text.stopped")
                            trial_text.status = FINISHED
                            trial_text.setAutoDraw(False)

                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return

                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp,
                            win=win,
                            timers=[routineTimer],
                            playbackComponents=[],
                        )
                        continue

                    if not continueRoutine:
                        trial_instructions.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False
                    for thisComponent in trial_instructions.components:
                        if (
                            hasattr(thisComponent, "status")
                            and thisComponent.status != FINISHED
                        ):
                            continueRoutine = True
                            break

                    if continueRoutine:
                        win.flip()

                # --- Ending Routine "trial_instructions" ---
                for thisComponent in trial_instructions.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)

                trial_instructions.tStop = globalClock.getTime(format="float")
                trial_instructions.tStopRefresh = tThisFlipGlobal
                thisExp.addData("trial_instructions.stopped", trial_instructions.tStop)

                if trial_instructions.maxDurationReached:
                    routineTimer.addTime(-trial_instructions.maxDuration)
                elif trial_instructions.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-1.000000)

                # --- Prepare to start Routine "trial_countdown" ---
                trial_countdown = data.Routine(
                    name="trial_countdown",
                    components=[fixation_cross, countdown_text],
                )
                trial_countdown.status = NOT_STARTED
                continueRoutine = True

                countdown_timer = core.Clock()
                last_shown_second = 7

                trial_countdown.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                trial_countdown.tStart = globalClock.getTime(format="float")
                trial_countdown.status = STARTED
                thisExp.addData("trial_countdown.started", trial_countdown.tStart)
                trial_countdown.maxDuration = None

                trial_countdownComponents = trial_countdown.components
                for thisComponent in trial_countdown.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, "status"):
                        thisComponent.status = NOT_STARTED

                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1

                # --- Run Routine "trial_countdown" ---
                if (
                    isinstance(blockLoop, data.TrialHandler2)
                    and thisBlockLoop.thisN != blockLoop.thisTrial.thisN
                ):
                    continueRoutine = False
                trial_countdown.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 6.0:
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1

                    if (
                        fixation_cross.status == NOT_STARTED
                        and tThisFlip >= 0.0 - frameTolerance
                    ):
                        fixation_cross.frameNStart = frameN
                        fixation_cross.tStart = t
                        fixation_cross.tStartRefresh = tThisFlipGlobal
                        win.timeOnFlip(fixation_cross, "tStartRefresh")
                        thisExp.timestampOnFlip(win, "fixation_cross.started")
                        fixation_cross.status = STARTED
                        fixation_cross.setAutoDraw(True)

                    if fixation_cross.status == STARTED:
                        pass

                    if fixation_cross.status == STARTED:
                        if (
                            tThisFlipGlobal
                            > fixation_cross.tStartRefresh + 6.0 - frameTolerance
                        ):
                            fixation_cross.tStop = t
                            fixation_cross.tStopRefresh = tThisFlipGlobal
                            fixation_cross.frameNStop = frameN
                            thisExp.timestampOnFlip(win, "fixation_cross.stopped")
                            fixation_cross.status = FINISHED
                            fixation_cross.setAutoDraw(False)

                    if (
                        countdown_text.status == NOT_STARTED
                        and tThisFlip >= 0.0 - frameTolerance
                    ):
                        countdown_text.frameNStart = frameN
                        countdown_text.tStart = t
                        countdown_text.tStartRefresh = tThisFlipGlobal
                        win.timeOnFlip(countdown_text, "tStartRefresh")
                        thisExp.timestampOnFlip(win, "countdown_text.started")
                        countdown_text.status = STARTED
                        countdown_text.setAutoDraw(True)
                        countdown_timer.reset()

                    if countdown_text.status == STARTED:
                        elapsed_time = countdown_timer.getTime()

                        if elapsed_time < 3.0:
                            current_second = 6 - int(elapsed_time)
                            if (
                                current_second != last_shown_second
                                and current_second >= 4
                            ):
                                countdown_text.text = f"New trial in {current_second}"
                                last_shown_second = current_second
                        else:
                            countdown_text.text = ""

                    if countdown_text.status == STARTED:
                        if (
                            tThisFlipGlobal
                            > countdown_text.tStartRefresh + 6.0 - frameTolerance
                        ):
                            countdown_text.tStop = t
                            countdown_text.tStopRefresh = tThisFlipGlobal
                            countdown_text.frameNStop = frameN
                            thisExp.timestampOnFlip(win, "countdown_text.stopped")
                            countdown_text.status = FINISHED
                            countdown_text.setAutoDraw(False)

                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return

                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp,
                            win=win,
                            timers=[routineTimer],
                            playbackComponents=[],
                        )
                        continue

                    if not continueRoutine:
                        trial_countdown.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False
                    for thisComponent in trial_countdown.components:
                        if (
                            hasattr(thisComponent, "status")
                            and thisComponent.status != FINISHED
                        ):
                            continueRoutine = True
                            break

                    if continueRoutine:
                        win.flip()

                # --- Ending Routine "trial_countdown" ---
                for thisComponent in trial_countdown.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)

                trial_countdown.tStop = globalClock.getTime(format="float")
                trial_countdown.tStopRefresh = tThisFlipGlobal
                thisExp.addData("trial_countdown.stopped", trial_countdown.tStop)

                if trial_countdown.maxDurationReached:
                    routineTimer.addTime(-trial_countdown.maxDuration)
                elif trial_countdown.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-6.000000)

                trialLoop = data.TrialHandler2(
                    name="trialLoop",
                    nReps=NUM_INTERRUPTIONS + 1,
                    method="sequential",
                    extraInfo=expInfo,
                    originPath=-1,
                    trialList=[None],
                    seed=None,
                )
                thisExp.addLoop(trialLoop)
                thisTrialLoop = trialLoop.trialList[0]

                if thisTrialLoop != None:
                    for paramName in thisTrialLoop:
                        globals()[paramName] = thisTrialLoop[paramName]

                for thisTrialLoop in trialLoop:
                    currentLoop = trialLoop
                    thisExp.timestampOnFlip(win, "thisRow.t")

                    if thisTrialLoop != None:
                        for paramName in thisTrialLoop:
                            globals()[paramName] = thisTrialLoop[paramName]

                    # --- Prepare to start Routine "primary_task" ---
                    primary_task = data.Routine(
                        name="primary_task",
                        components=[dummy_3],
                    )
                    primary_task.status = NOT_STARTED
                    continueRoutine = True

                    interruptionSequence = "none"
                    interruptionTimer = core.Clock()

                    if (
                        INTERRUPTION_CONDITION == "sequential"
                        and trialLoop.thisN > 0
                        and primaryTask.submitted
                    ):
                        continueRoutine = False
                        primaryTask.hide()
                    else:
                        if trialLoop.thisN == 0:
                            keyboard = Keyboard(win=win)

                            primaryTask = PrimaryTask(
                                win=win,
                                keyboard=keyboard,
                                mouse=mouse,
                                mode=PRIMARY_DIFFICULTY,
                                ok_rect=ok_button_rect,
                                ok_text=ok_button_text,
                            )

                            primaryTask.set_target_word(thisTrialWord)
                            primaryTask.reset()

                        if (
                            INTERRUPTION_CONDITION == "interrupted"
                            and trialLoop.thisN > 0
                        ):
                            primaryTask._resume_position = (
                                primaryTask.text_entry_screen.current_position
                            )
                            primaryTask._resumption_logged = False

                            reappear_time = globalClock.getTime()
                            dl.log("typing_reappear_timestamp", reappear_time)

                            primaryTask.text_entry_screen.first_key_pressed = False

                        primaryTask.show()
                        if "interruptionTask" in globals() and interruptionTask:
                            interruptionTask.hide()

                    primary_task.tStartRefresh = win.getFutureFlipTime(
                        clock=globalClock
                    )
                    primary_task.tStart = globalClock.getTime(format="float")
                    primary_task.status = STARTED
                    thisExp.addData("primary_task.started", primary_task.tStart)
                    primary_task.maxDuration = None

                    primary_taskComponents = primary_task.components
                    for thisComponent in primary_task.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, "status"):
                            thisComponent.status = NOT_STARTED

                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1

                    if trialLoop.thisN == 0:
                        trial_start_time = dl.log_timing_start("trial")
                        typing_start_time = dl.log_timing_start("typing")

                        dl.log("trial_completed", False)

                    # --- Run Routine "primary_task" ---
                    if (
                        isinstance(trialLoop, data.TrialHandler2)
                        and thisTrialLoop.thisN != trialLoop.thisTrial.thisN
                    ):
                        continueRoutine = False
                    primary_task.forceEnded = routineForceEnded = not continueRoutine
                    while continueRoutine:
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1

                        primaryTask.draw()

                        if interruptionSequence == "none":
                            status = primaryTask.check_input()
                            trialManager.update(primaryTask)

                            if (
                                INTERRUPTION_CONDITION == "interrupted"
                                and trialLoop.thisN > 0
                                and not primaryTask._resumption_logged
                                and primaryTask.text_entry_screen.first_key_pressed
                            ):

                                first_key_time = globalClock.getTime()
                                dl.log("typing_resume_timestamp", first_key_time)

                                reappear_time = dl.get_current_data()[
                                    "cm_typing_reappear_timestamp"
                                ]
                                dl.log("resumption_lag", first_key_time - reappear_time)

                                primaryTask._resumption_logged = True

                            if status == "submitted":
                                primaryTask.hide()
                                continueRoutine = False
                            elif (
                                INTERRUPTION_CONDITION == "interrupted"
                                and trialManager.consume()
                            ):
                                interruptionSequence = "masked"
                                interruptionTimer.reset()
                        elif interruptionSequence == "masked":
                            if interruptionTimer.getTime() >= INTERRUPT_DELAY:
                                interruptionSequence = "complete"
                                primaryTask.hide()
                                continueRoutine = False

                        if (
                            dummy_3.status == NOT_STARTED
                            and tThisFlip >= 0.0 - frameTolerance
                        ):
                            dummy_3.frameNStart = frameN
                            dummy_3.tStart = t
                            dummy_3.tStartRefresh = tThisFlipGlobal
                            win.timeOnFlip(dummy_3, "tStartRefresh")
                            thisExp.timestampOnFlip(win, "dummy_3.started")
                            dummy_3.status = STARTED
                            dummy_3.setAutoDraw(True)

                        if dummy_3.status == STARTED:
                            pass

                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return

                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp,
                                win=win,
                                timers=[routineTimer],
                                playbackComponents=[],
                            )
                            continue

                        if not continueRoutine:
                            primary_task.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False
                        for thisComponent in primary_task.components:
                            if (
                                hasattr(thisComponent, "status")
                                and thisComponent.status != FINISHED
                            ):
                                continueRoutine = True
                                break

                        if continueRoutine:
                            win.flip()

                    # --- Ending Routine "primary_task" ---
                    for thisComponent in primary_task.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)

                    primary_task.tStop = globalClock.getTime(format="float")
                    primary_task.tStopRefresh = tThisFlipGlobal
                    thisExp.addData("primary_task.stopped", primary_task.tStop)

                    if primaryTask.submitted:
                        primaryTask.record_data()

                    routineTimer.reset()

                    if primaryTask.submitted:
                        entered_text = primaryTask.text_entry_screen.entered_text
                        target_word = primaryTask.target_word

                        character_accuracy = dl.calculate_character_accuracy(
                            target_word, entered_text
                        )
                        dl.log_multiple(
                            {
                                "entered_text": entered_text,
                                "entered_text_length": len(entered_text),
                                "typing_correct": entered_text.upper()
                                == target_word.upper(),
                                "typing_accuracy": character_accuracy,
                                "typing_error_rate": 1.0 - character_accuracy,
                            }
                        )

                        if (
                            INTERRUPTION_CONDITION == "sequential"
                            and trialLoop.thisN == 0
                        ):
                            typing_end_time = dl.log_timing_end(
                                "typing", typing_start_time
                            )
                        else:
                            typing_submission_timestamp = globalClock.getTime()
                            dl.log(
                                "typing_submission_timestamp",
                                typing_submission_timestamp,
                            )

                    elif (
                        INTERRUPTION_CONDITION == "interrupted"
                        and interruptionSequence == "complete"
                    ):
                        typing_pause_time = globalClock.getTime()
                        dl.log("typing_pause_timestamp", typing_pause_time)

                    # --- Prepare to start Routine "load_subtrial_data" ---
                    load_subtrial_data = data.Routine(
                        name="load_subtrial_data",
                        components=[],
                    )
                    load_subtrial_data.status = NOT_STARTED
                    continueRoutine = True

                    current_nback_params = trialManager.get_current_nback_data()

                    if current_nback_params is None:
                        continueRoutine = False
                    else:
                        n_back_level = current_nback_params["n_back_level"]
                        trial_nback_instructions_text = f"{n_back_level}-back task."

                    load_subtrial_data.tStartRefresh = win.getFutureFlipTime(
                        clock=globalClock
                    )
                    load_subtrial_data.tStart = globalClock.getTime(format="float")
                    load_subtrial_data.status = STARTED
                    thisExp.addData(
                        "load_subtrial_data.started", load_subtrial_data.tStart
                    )
                    load_subtrial_data.maxDuration = None

                    load_subtrial_dataComponents = load_subtrial_data.components
                    for thisComponent in load_subtrial_data.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, "status"):
                            thisComponent.status = NOT_STARTED

                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1

                    # --- Run Routine "load_subtrial_data" ---
                    if (
                        isinstance(trialLoop, data.TrialHandler2)
                        and thisTrialLoop.thisN != trialLoop.thisTrial.thisN
                    ):
                        continueRoutine = False
                    load_subtrial_data.forceEnded = routineForceEnded = (
                        not continueRoutine
                    )
                    while continueRoutine:
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1

                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return

                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp,
                                win=win,
                                timers=[routineTimer],
                                playbackComponents=[],
                            )
                            continue

                        if not continueRoutine:
                            load_subtrial_data.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False
                        for thisComponent in load_subtrial_data.components:
                            if (
                                hasattr(thisComponent, "status")
                                and thisComponent.status != FINISHED
                            ):
                                continueRoutine = True
                                break

                        if continueRoutine:
                            win.flip()

                    # --- Ending Routine "load_subtrial_data" ---
                    for thisComponent in load_subtrial_data.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)

                    load_subtrial_data.tStop = globalClock.getTime(format="float")
                    load_subtrial_data.tStopRefresh = tThisFlipGlobal
                    thisExp.addData(
                        "load_subtrial_data.stopped", load_subtrial_data.tStop
                    )

                    routineTimer.reset()

                    # --- Prepare to start Routine "nback_task_instructions" ---
                    nback_task_instructions = data.Routine(
                        name="nback_task_instructions",
                        components=[nback_instructions_component],
                    )
                    nback_task_instructions.status = NOT_STARTED
                    continueRoutine = True

                    nback_instructions_component.setText(trial_nback_instructions_text)

                    if INTERRUPTION_CONDITION == "sequential":
                        if primaryTask.submitted and trialLoop.thisN == 0:
                            continueRoutine = True
                        else:
                            continueRoutine = False
                    else:
                        if primaryTask.submitted:
                            continueRoutine = False

                    nback_task_instructions.tStartRefresh = win.getFutureFlipTime(
                        clock=globalClock
                    )
                    nback_task_instructions.tStart = globalClock.getTime(format="float")
                    nback_task_instructions.status = STARTED
                    thisExp.addData(
                        "nback_task_instructions.started",
                        nback_task_instructions.tStart,
                    )
                    nback_task_instructions.maxDuration = None

                    nback_task_instructionsComponents = (
                        nback_task_instructions.components
                    )
                    for thisComponent in nback_task_instructions.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, "status"):
                            thisComponent.status = NOT_STARTED

                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1

                    # --- Run Routine "nback_task_instructions" ---
                    if (
                        isinstance(trialLoop, data.TrialHandler2)
                        and thisTrialLoop.thisN != trialLoop.thisTrial.thisN
                    ):
                        continueRoutine = False
                    nback_task_instructions.forceEnded = routineForceEnded = (
                        not continueRoutine
                    )
                    while continueRoutine and routineTimer.getTime() < 1.5:
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1

                        if (
                            nback_instructions_component.status == NOT_STARTED
                            and tThisFlip >= 0.0 - frameTolerance
                        ):
                            nback_instructions_component.frameNStart = frameN
                            nback_instructions_component.tStart = t
                            nback_instructions_component.tStartRefresh = tThisFlipGlobal
                            win.timeOnFlip(
                                nback_instructions_component, "tStartRefresh"
                            )
                            thisExp.timestampOnFlip(
                                win, "nback_instructions_component.started"
                            )
                            nback_instructions_component.status = STARTED
                            nback_instructions_component.setAutoDraw(True)

                        if nback_instructions_component.status == STARTED:
                            pass

                        if nback_instructions_component.status == STARTED:
                            if (
                                tThisFlipGlobal
                                > nback_instructions_component.tStartRefresh
                                + 1.5
                                - frameTolerance
                            ):
                                nback_instructions_component.tStop = t
                                nback_instructions_component.tStopRefresh = (
                                    tThisFlipGlobal
                                )
                                nback_instructions_component.frameNStop = frameN
                                thisExp.timestampOnFlip(
                                    win, "nback_instructions_component.stopped"
                                )
                                nback_instructions_component.status = FINISHED
                                nback_instructions_component.setAutoDraw(False)

                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return

                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp,
                                win=win,
                                timers=[routineTimer],
                                playbackComponents=[],
                            )
                            continue

                        if not continueRoutine:
                            nback_task_instructions.forceEnded = routineForceEnded = (
                                True
                            )
                            break
                        continueRoutine = False
                        for thisComponent in nback_task_instructions.components:
                            if (
                                hasattr(thisComponent, "status")
                                and thisComponent.status != FINISHED
                            ):
                                continueRoutine = True
                                break

                        if continueRoutine:
                            win.flip()

                    # --- Ending Routine "nback_task_instructions" ---
                    for thisComponent in nback_task_instructions.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)

                    nback_task_instructions.tStop = globalClock.getTime(format="float")
                    nback_task_instructions.tStopRefresh = tThisFlipGlobal
                    thisExp.addData(
                        "nback_task_instructions.stopped", nback_task_instructions.tStop
                    )

                    if nback_task_instructions.maxDurationReached:
                        routineTimer.addTime(-nback_task_instructions.maxDuration)
                    elif nback_task_instructions.forceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-1.500000)

                    # --- Prepare to start Routine "interrupting_task" ---
                    interrupting_task = data.Routine(
                        name="interrupting_task",
                        components=[dummy_2],
                    )
                    interrupting_task.status = NOT_STARTED
                    continueRoutine = True

                    nback_ran_this_pass = False

                    if INTERRUPTION_CONDITION == "sequential":
                        if primaryTask.submitted and trialLoop.thisN == 0:
                            continueRoutine = True
                        else:
                            continueRoutine = False
                    elif primaryTask.submitted:
                        continueRoutine = False
                    elif current_nback_params is None:
                        continueRoutine = False

                    if continueRoutine:
                        interruptionTask = NbackTask(
                            win=win,
                            n_back_level=current_nback_params["n_back_level"],
                            num_stims=current_nback_params["num_stims"],
                            stim_list=current_nback_params["stim_list"],
                            stim_durations=current_nback_params["stim_durations"],
                            matches=current_nback_params["match_positions"],
                            isi_times=current_nback_params["isi_times"],
                            stim_duration=NBACK_STIM_DURATION,
                            response_window_ratio=0.9,
                            feedback_enabled=False,
                            mask_enabled=NBACK_MASK_ENABLED,
                            hide_buffer=0,
                        )

                        interruptionTask.show()
                        primaryTask.hide()

                        nback_ran_this_pass = True

                        if hasattr(interruptionTask, "start"):
                            interruptionTask.start()

                        event.clearEvents()

                    interrupting_task.tStartRefresh = win.getFutureFlipTime(
                        clock=globalClock
                    )
                    interrupting_task.tStart = globalClock.getTime(format="float")
                    interrupting_task.status = STARTED
                    thisExp.addData(
                        "interrupting_task.started", interrupting_task.tStart
                    )
                    interrupting_task.maxDuration = None

                    interrupting_taskComponents = interrupting_task.components
                    for thisComponent in interrupting_task.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, "status"):
                            thisComponent.status = NOT_STARTED

                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1

                    if continueRoutine:
                        nback_start_time = dl.log_timing_start("nback")

                    # --- Run Routine "interrupting_task" ---
                    if (
                        isinstance(trialLoop, data.TrialHandler2)
                        and thisTrialLoop.thisN != trialLoop.thisTrial.thisN
                    ):
                        continueRoutine = False
                    interrupting_task.forceEnded = routineForceEnded = (
                        not continueRoutine
                    )
                    while continueRoutine:
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1

                        interruptionTask.draw()

                        if interruptionTask.check_key() == "end":
                            interruptionTask.hide()
                            continueRoutine = False

                        if (
                            dummy_2.status == NOT_STARTED
                            and tThisFlip >= 0.0 - frameTolerance
                        ):
                            dummy_2.frameNStart = frameN
                            dummy_2.tStart = t
                            dummy_2.tStartRefresh = tThisFlipGlobal
                            win.timeOnFlip(dummy_2, "tStartRefresh")
                            thisExp.timestampOnFlip(win, "dummy_2.started")
                            dummy_2.status = STARTED
                            dummy_2.setAutoDraw(True)

                        if dummy_2.status == STARTED:
                            pass

                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return

                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp,
                                win=win,
                                timers=[routineTimer],
                                playbackComponents=[],
                            )
                            continue

                        if not continueRoutine:
                            interrupting_task.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False
                        for thisComponent in interrupting_task.components:
                            if (
                                hasattr(thisComponent, "status")
                                and thisComponent.status != FINISHED
                            ):
                                continueRoutine = True
                                break

                        if continueRoutine:
                            win.flip()

                    # --- Ending Routine "interrupting_task" ---
                    for thisComponent in interrupting_task.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)

                    interrupting_task.tStop = globalClock.getTime(format="float")
                    interrupting_task.tStopRefresh = tThisFlipGlobal
                    thisExp.addData(
                        "interrupting_task.stopped", interrupting_task.tStop
                    )

                    event.clearEvents()

                    routineTimer.reset()

                    if nback_ran_this_pass:
                        nback_end_time = dl.log_timing_end("nback", nback_start_time)
                        performance = interruptionTask.get_performance()

                        dl.log_multiple(
                            {
                                "nback_stim_list": current_nback_params["stim_list"],
                                "nback_match_positions": current_nback_params[
                                    "match_positions"
                                ],
                                "nback_participant_responses": [
                                    i
                                    for i, resp in enumerate(interruptionTask.responses)
                                    if resp
                                ],
                                "nback_hits": performance["hits"],
                                "nback_misses": performance["misses"],
                                "nback_false_alarms": performance["false_alarms"],
                                "nback_correct_rejections": performance[
                                    "correct_rejections"
                                ],
                                "nback_accuracy": performance["accuracy"],
                                "nback_hit_rate": (
                                    performance["hits"]
                                    / (performance["hits"] + performance["misses"])
                                    if (performance["hits"] + performance["misses"]) > 0
                                    else 0
                                ),
                                "nback_false_alarm_rate": (
                                    performance["false_alarms"]
                                    / (
                                        performance["false_alarms"]
                                        + performance["correct_rejections"]
                                    )
                                    if (
                                        performance["false_alarms"]
                                        + performance["correct_rejections"]
                                    )
                                    > 0
                                    else 0
                                ),
                                "nback_num_stims": performance["num_stims"],
                            }
                        )

                # --- Prepare to start Routine "time_estimate" ---

                trial_end_time = dl.log_timing_end("trial", trial_start_time)

                time_estimate = data.Routine(
                    name="time_estimate",
                    components=[dummy_5],
                )
                time_estimate.status = NOT_STARTED
                continueRoutine = True

                slider = TimeSlider(win=win)
                timeEstimationTask = TimeEstimationTask(
                    win=win,
                    input_device=slider,
                    mouse=mouse,
                    ok_rect=ok_button_rect,
                    ok_text=ok_button_text,
                )
                timeEstimationTask.reset()
                timeEstimationTask.show()

                time_estimate.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                time_estimate.tStart = globalClock.getTime(format="float")
                time_estimate.status = STARTED
                thisExp.addData("time_estimate.started", time_estimate.tStart)
                time_estimate.maxDuration = None

                time_estimateComponents = time_estimate.components
                for thisComponent in time_estimate.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, "status"):
                        thisComponent.status = NOT_STARTED

                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1

                # --- Run Routine "time_estimate" ---
                if (
                    isinstance(blockLoop, data.TrialHandler2)
                    and thisBlockLoop.thisN != blockLoop.thisTrial.thisN
                ):
                    continueRoutine = False
                time_estimate.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1

                    timeEstimationTask.draw()
                    status = timeEstimationTask.check_input()
                    if status == "submitted":
                        timeEstimation = timeEstimationTask.get_response()
                        continueRoutine = False

                    if (
                        dummy_5.status == NOT_STARTED
                        and tThisFlip >= 0.0 - frameTolerance
                    ):
                        dummy_5.frameNStart = frameN
                        dummy_5.tStart = t
                        dummy_5.tStartRefresh = tThisFlipGlobal
                        win.timeOnFlip(dummy_5, "tStartRefresh")
                        thisExp.timestampOnFlip(win, "dummy_5.started")
                        dummy_5.status = STARTED
                        dummy_5.setAutoDraw(True)

                    if dummy_5.status == STARTED:
                        pass

                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return

                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp,
                            win=win,
                            timers=[routineTimer],
                            playbackComponents=[],
                        )
                        continue

                    if not continueRoutine:
                        time_estimate.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False
                    for thisComponent in time_estimate.components:
                        if (
                            hasattr(thisComponent, "status")
                            and thisComponent.status != FINISHED
                        ):
                            continueRoutine = True
                            break

                    if continueRoutine:
                        win.flip()

                # --- Ending Routine "time_estimate" ---
                for thisComponent in time_estimate.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)

                time_estimate.tStop = globalClock.getTime(format="float")
                time_estimate.tStopRefresh = tThisFlipGlobal
                thisExp.addData("time_estimate.stopped", time_estimate.tStop)

                timeEstimationTask.hide()

                routineTimer.reset()

                time_estimate_string = timeEstimationTask.get_response()
                minutes, seconds = timeEstimationTask.time_display.get_time()
                time_estimate_seconds = minutes * 60 + seconds

                dl.log_multiple(
                    {
                        "time_estimate_input": time_estimate_string,
                        "time_estimate_seconds": time_estimate_seconds,
                        "time_input_method": "slider",
                    }
                )

                current_data = dl.get_current_data()
                actual_trial_duration = (
                    current_data["cm_trial_end_timestamp"]
                    - current_data["cm_trial_start_timestamp"]
                )

                actual_trial_duration_sec = int(round(actual_trial_duration))

                typing_start = current_data.get("cm_typing_start_timestamp", 0)
                typing_pause = current_data.get("cm_typing_pause_timestamp")
                typing_reappear = current_data.get("cm_typing_reappear_timestamp")
                typing_submission = current_data.get("cm_typing_submission_timestamp")

                if typing_submission is None:
                    typing_submission = current_data.get("cm_typing_end_timestamp", 0)

                if typing_pause is not None and typing_reappear is not None:
                    time_on_primary_task = (typing_pause - typing_start) + (
                        typing_submission - typing_reappear
                    )
                else:
                    time_on_primary_task = current_data["cm_typing_duration"]

                time_estimation_ratio = (
                    time_estimate_seconds / actual_trial_duration
                    if actual_trial_duration > 0
                    else 0
                )
                time_estimation_error = time_estimate_seconds - actual_trial_duration
                time_estimation_accuracy = (
                    abs(time_estimation_error) / actual_trial_duration
                    if actual_trial_duration > 0
                    else 0
                )

                dl.log_multiple(
                    {
                        "actual_trial_duration": actual_trial_duration,
                        "total_trial_duration": actual_trial_duration,
                        "actual_trial_duration_sec": actual_trial_duration_sec,
                        "time_on_primary_task": time_on_primary_task,
                        "time_estimation_ratio": time_estimation_ratio,
                        "time_estimation_error": time_estimation_error,
                        "time_estimation_accuracy": time_estimation_accuracy,
                        "trial_completed": True,
                    }
                )

                thisExp.nextEntry()

                # --- Prepare to start Routine "timing_feedback" ---
                timing_feedback = data.Routine(
                    name="timing_feedback",
                    components=[dummy_feedback],
                )
                timing_feedback.status = NOT_STARTED
                continueRoutine = True

                if is_practice_block:
                    timingFeedbackScreen = TimingTaskFeedbackScreen(win=win)

                    timingFeedbackScreen.set_timing_feedback(
                        actual_trial_duration, time_estimate_seconds
                    )

                    timingFeedbackScreen.show()
                else:
                    continueRoutine = False

                timing_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                timing_feedback.tStart = globalClock.getTime(format="float")
                timing_feedback.status = STARTED
                thisExp.addData("timing_feedback.started", timing_feedback.tStart)
                timing_feedback.maxDuration = None

                timing_feedbackComponents = timing_feedback.components
                for thisComponent in timing_feedback.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, "status"):
                        thisComponent.status = NOT_STARTED

                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1

                # --- Run Routine "timing_feedback" ---
                if (
                    isinstance(blockLoop, data.TrialHandler2)
                    and thisBlockLoop.thisN != blockLoop.thisTrial.thisN
                ):
                    continueRoutine = False
                timing_feedback.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1

                    if is_practice_block:
                        timingFeedbackScreen.draw()
                        nav_result = timingFeedbackScreen.check_navigation(
                            mouse, defaultKeyboard
                        )
                        if nav_result == "escape":
                            thisExp.status = FINISHED
                            endExperiment(thisExp, win=win)
                            return
                        elif nav_result:
                            continueRoutine = False

                    if (
                        dummy_feedback.status == NOT_STARTED
                        and tThisFlip >= 0.0 - frameTolerance
                    ):
                        dummy_feedback.frameNStart = frameN
                        dummy_feedback.tStart = t
                        dummy_feedback.tStartRefresh = tThisFlipGlobal
                        win.timeOnFlip(dummy_feedback, "tStartRefresh")
                        thisExp.timestampOnFlip(win, "dummy_feedback.started")
                        dummy_feedback.status = STARTED
                        dummy_feedback.setAutoDraw(True)

                    if dummy_feedback.status == STARTED:
                        pass

                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return

                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp,
                            win=win,
                            timers=[routineTimer],
                            playbackComponents=[],
                        )
                        continue

                    if not continueRoutine:
                        timing_feedback.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False
                    for thisComponent in timing_feedback.components:
                        if (
                            hasattr(thisComponent, "status")
                            and thisComponent.status != FINISHED
                        ):
                            continueRoutine = True
                            break

                    if continueRoutine:
                        win.flip()

                # --- Ending Routine "timing_feedback" ---
                for thisComponent in timing_feedback.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                if is_practice_block:
                    timingFeedbackScreen.hide()

                timing_feedback.tStop = globalClock.getTime(format="float")
                timing_feedback.tStopRefresh = tThisFlipGlobal
                thisExp.addData("timing_feedback.stopped", timing_feedback.tStop)
                routineTimer.reset()

            if thisSession is not None:
                thisSession.sendExperimentData()

    # --- Prepare to start Routine "thank_you" ---
    thank_you = data.Routine(
        name="thank_you",
        components=[dummy_thank_you],
    )
    thank_you.status = NOT_STARTED
    continueRoutine = True

    thankYouScreen = InstructionScreen(
        win=win,
        title=THANK_YOU_TITLE,
        content=THANK_YOU_CONTENT,
        button_text="End Experiment",
        button_delay=1,
        button_width=0.3,
    )
    thankYouScreen.show()

    thank_you.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thank_you.tStart = globalClock.getTime(format="float")
    thank_you.status = STARTED
    thisExp.addData("thank_you.started", thank_you.tStart)
    thank_you.maxDuration = None

    thank_youComponents = thank_you.components
    for thisComponent in thank_you.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, "status"):
            thisComponent.status = NOT_STARTED

    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "thank_you" ---
    thank_you.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1

        thankYouScreen.draw()
        nav_result = thankYouScreen.check_navigation(mouse, defaultKeyboard)
        if nav_result:
            continueRoutine = False

        if dummy_thank_you.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            dummy_thank_you.frameNStart = frameN
            dummy_thank_you.tStart = t
            dummy_thank_you.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(dummy_thank_you, "tStartRefresh")
            thisExp.timestampOnFlip(win, "dummy_thank_you.started")
            dummy_thank_you.status = STARTED
            dummy_thank_you.setAutoDraw(True)

        if dummy_thank_you.status == STARTED:
            pass

        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return

        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, win=win, timers=[routineTimer], playbackComponents=[]
            )
            continue

        if not continueRoutine:
            thank_you.forceEnded = routineForceEnded = True
            break
        continueRoutine = False
        for thisComponent in thank_you.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break

        if continueRoutine:
            win.flip()

    # --- Ending Routine "thank_you" ---
    for thisComponent in thank_you.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thankYouScreen.hide()

    thank_you.tStop = globalClock.getTime(format="float")
    thank_you.tStopRefresh = tThisFlipGlobal
    thisExp.addData("thank_you.stopped", thank_you.tStop)
    routineTimer.reset()

    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment.

    Parameters:
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + ".csv", delim="auto")
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.

    This function does NOT close the window or end the Python process - use `quit` for this.

    Parameters:
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip()
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.

    Parameters:
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment.
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip()
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == "__main__":
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    # setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(expInfo=expInfo, thisExp=thisExp, win=win, globalClock="float")
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
