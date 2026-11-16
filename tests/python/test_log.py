# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""The log bridge: sif's messages in the logging module, under pysif.<tag>,
at the level the pysif loggers set, from any thread -- including threads
Python does not know, which must never take the GIL themselves."""

import logging

import pysif
import pysif._log as bridge
import pytest
from pysif import _pysif

from conftest import GADGET


def records(caplog, name):
    return [r for r in caplog.records if r.name == name]


def drain():
    # A call that releases the GIL empties the workers' queue on the way back.
    _pysif._log_from_threads("", 1, 0, True)


def test_message_reaches_its_logger(caplog):
    caplog.set_level(logging.WARNING, logger="pysif")
    pysif.system_init(skip_tuning=True)  # already up: warns
    got = records(caplog, "pysif.core.system")
    assert len(got) == 1
    assert got[0].levelno == logging.WARNING
    assert got[0].getMessage() == "the system state is already initialised"
    # Credited to the Python line that called into sif, not to the bridge.
    assert got[0].filename == "test_log.py"


def test_error_is_raised_not_logged(caplog):
    # The exception carries the reason; a log line would only repeat it.
    caplog.set_level(logging.WARNING, logger="pysif")
    with pytest.raises(FileNotFoundError, match="no_such_snapshot"):
        pysif.io.gadget_header(str(GADGET / "no_such_snapshot"))
    assert [r for r in caplog.records if r.name.startswith("pysif.")] == []


def test_failure_carries_the_library_reason():
    # Bindings once raised "failed; see the sif log": with errors no longer
    # logged, the message has to be the library's own.
    with pytest.raises(ValueError, match="n_cells=0"):
        pysif.CellLinkedList(0, 100.0, 10)


def test_worker_error_is_logged(caplog):
    # Recorded on a thread no exception reads, so the log is all there is.
    caplog.set_level(logging.WARNING, logger="pysif")
    _pysif._log_from_threads("worker failed", 2, 1, True, True)
    got = [r for r in records(caplog, "pysif.utils.logger")
           if r.getMessage() == "worker failed"]
    assert len(got) == 2 and all(r.levelno == logging.ERROR for r in got)


def test_level_follows_set_level_at_once():
    # No call into sif in between: the hook on logging's cache clearing is
    # what moves the library's level, for calls that keep the GIL.
    root = logging.getLogger("pysif")
    saved = root.level
    try:
        root.setLevel(logging.ERROR)
        assert _pysif._log_level() == 4
        root.setLevel(logging.DEBUG)
        assert _pysif._log_level() == 1
        logging.disable(logging.CRITICAL)
        assert _pysif._log_level() == 5
    finally:
        logging.disable(logging.NOTSET)
        root.setLevel(saved)


def test_level_follows_the_loggers(caplog):
    root = logging.getLogger("pysif")
    child = logging.getLogger("pysif.finder.exodus")
    saved = root.level, child.level
    try:
        root.setLevel(logging.ERROR)
        child.setLevel(logging.NOTSET)
        assert bridge.c_level() == 4
        # A child below its parent lowers the floor for everything, and the
        # loggers themselves keep the rest out.
        child.setLevel(logging.DEBUG)
        assert bridge.c_level() == 1
        child.setLevel(bridge.TRACE)
        assert bridge.c_level() == 0
        logging.disable(logging.WARNING)
        assert bridge.c_level() == 4
        logging.disable(logging.CRITICAL)
        assert bridge.c_level() == 5
    finally:
        logging.disable(logging.NOTSET)
        root.setLevel(saved[0])
        child.setLevel(saved[1])


def test_filtered_level_logs_nothing(caplog):
    caplog.set_level(logging.ERROR, logger="pysif")
    pysif.system_init(skip_tuning=True)  # the warning is below ERROR
    assert records(caplog, "pysif.core.system") == []


def test_trace_level_is_named():
    assert logging.getLevelName(bridge.TRACE) == "TRACE"


@pytest.mark.parametrize("release_gil", [True, False])
def test_worker_threads(caplog, release_gil):
    # Holding the GIL is the case that would deadlock if a worker took it:
    # the caller joins the workers without letting it go.
    caplog.set_level(logging.WARNING, logger="pysif")
    _pysif._log_from_threads("from a worker", 8, 25, release_gil)
    drain()
    got = [r for r in records(caplog, "pysif.utils.logger")
           if r.getMessage() == "from a worker"]
    assert len(got) == 8 * 25


def test_worker_queue_overflow_is_counted(caplog):
    caplog.set_level(logging.WARNING, logger="pysif")
    _pysif._log_from_threads("flood", 4, 1000, False)  # 4000 > the cap
    drain()
    msgs = [r.getMessage() for r in records(caplog, "pysif.utils.logger")]
    assert msgs.count("flood") == 1024
    assert any(m.startswith("2976 messages from worker threads dropped")
               for m in msgs)


def test_percent_is_not_a_format(caplog):
    caplog.set_level(logging.WARNING, logger="pysif")
    _pysif._log_from_threads("100%s %d done", 1, 1, True)
    msgs = [r.getMessage() for r in records(caplog, "pysif.utils.logger")]
    assert "100%s %d done" in msgs


def test_raising_handler_does_not_escape(monkeypatch):
    # An exception from a handler cannot go back through sif's C: it is
    # reported as unraisable, and the call returns normally.
    class Boom(logging.Handler):
        def emit(self, record):
            raise RuntimeError("handler failed")

    reported = []
    monkeypatch.setattr("sys.unraisablehook", reported.append)
    logger = logging.getLogger("pysif.core.system")
    handler = Boom()
    logger.addHandler(handler)
    try:
        pysif.system_init(skip_tuning=True)  # already up: warns
    finally:
        logger.removeHandler(handler)
    assert len(reported) == 1
    assert isinstance(reported[0].exc_value, RuntimeError)
