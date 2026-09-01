"""A failed build has to carry TensorRT's reason out with it.

No GPU: the logger is a plain callback, and what matters is which severities it
keeps. The consequence of not keeping them is that the exception can only say
"see the log above", and a caller matching on TensorRT's wording -- whisper-trt
does this to recognise an out-of-memory build -- never sees the text.
"""

import pytest

from torch2trt import trt
from torch2trt.torch2trt import _ErrorRecordingLogger


@pytest.fixture(name="log")
def logger_fixture() -> _ErrorRecordingLogger:
    return _ErrorRecordingLogger()


def test_errors_are_kept(log: _ErrorRecordingLogger) -> None:
    log.log(trt.Logger.ERROR, "Could not find any implementation for node")
    assert log.errors == ["Could not find any implementation for node"]


def test_internal_errors_are_kept(log: _ErrorRecordingLogger) -> None:
    """The severity below ERROR, and the one the buildWtsEngine assertion uses."""
    log.log(trt.Logger.INTERNAL_ERROR, "Assertion engine failed.")
    assert log.errors == ["Assertion engine failed."]


@pytest.mark.parametrize(
    "severity", [trt.Logger.WARNING, trt.Logger.INFO, trt.Logger.VERBOSE]
)
def test_everything_quieter_than_an_error_is_not_kept(
    log: _ErrorRecordingLogger, severity: object
) -> None:
    """A build logs hundreds of these; only the failure belongs in the message."""
    log.log(severity, "the logger passed into createInferRuntime differs ...")
    assert log.errors == []


def test_order_is_preserved(log: _ErrorRecordingLogger) -> None:
    """The exception reports the last few, so they have to arrive in order."""
    for i in range(5):
        log.log(trt.Logger.ERROR, f"error {i}")
    assert log.errors[-3:] == ["error 2", "error 3", "error 4"]


def test_a_clean_build_records_nothing(log: _ErrorRecordingLogger) -> None:
    assert log.errors == []
