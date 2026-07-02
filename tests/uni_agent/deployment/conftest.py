"""Shared fixtures/helpers for the mock deployment tests."""

from __future__ import annotations

import pytest

pytest.importorskip("swerex")

from swerex.runtime.abstract import BashAction  # noqa: E402


def _bash(command: str, timeout: int = 10) -> BashAction:
    """Build a ``BashAction`` for a canned command string.

    Shared across the mock test files so they don't each redefine this factory
    (with slightly different -- and sometimes buggy -- copies).
    """
    return BashAction(command=command, timeout=timeout)
