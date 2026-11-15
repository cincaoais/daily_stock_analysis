# -*- coding: utf-8 -*-
"""Chan theory (缠论) structure engine.

Code computes 分型/笔/中枢/背驰/买卖点 and hands the LLM a JSON-serialisable
snapshot; the LLM never derives structure itself.
"""

from src.chan.engine import (
    LEVEL_KEYS,
    SCHEMA_VERSION,
    SUPPORTED_TIMEFRAMES,
    build_chan_snapshot,
    build_level_snapshot,
)

__all__ = [
    "LEVEL_KEYS",
    "SCHEMA_VERSION",
    "SUPPORTED_TIMEFRAMES",
    "build_chan_snapshot",
    "build_level_snapshot",
]
