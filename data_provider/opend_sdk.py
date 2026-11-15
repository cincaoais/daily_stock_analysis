# -*- coding: utf-8 -*-
"""Lazy loader for the OpenD client SDK.

moomoo-api and futu-api expose the same client protocol against a local OpenD
process; only the package (and import) name differs. Market-data adapters must
never import either package at module import time, so this module resolves the
SDK on first use and caches the result.
"""

from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)

_SDK_MODULE_NAMES = ("moomoo", "futu")
_cached_sdk: Optional[ModuleType] = None
_cached_module_name: Optional[str] = None


def reset_opend_sdk_cache() -> None:
    """Clear the cached SDK resolution (tests and runtime re-install hooks)."""
    global _cached_sdk, _cached_module_name
    _cached_sdk = None
    _cached_module_name = None


def load_opend_sdk() -> ModuleType:
    """Return the importable OpenD SDK module, preferring ``moomoo``.

    Raises:
        ImportError: when neither ``moomoo-api`` nor ``futu-api`` is installed.
    """
    global _cached_sdk, _cached_module_name

    if (
        _cached_sdk is not None
        and _cached_module_name
        and sys.modules.get(_cached_module_name) is _cached_sdk
    ):
        # A higher-priority brand may have been installed or injected after
        # the cached resolution; keep the documented moomoo > futu preference.
        for name in _SDK_MODULE_NAMES:
            if name == _cached_module_name:
                return _cached_sdk
            if sys.modules.get(name) is not None:
                break

    for name in _SDK_MODULE_NAMES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        _cached_sdk = module
        _cached_module_name = name
        return module

    reset_opend_sdk_cache()
    raise ImportError(
        "OpenD SDK 未安装：请安装 moomoo-api 或 futu-api（二者共用本机 OpenD 协议）"
    )


def opend_sdk_brand(module: Optional[ModuleType] = None) -> str:
    """Return a short brand string (``moomoo``/``futu``) for logs."""
    if module is None:
        try:
            module = load_opend_sdk()
        except ImportError:
            return "opend"
    name = getattr(module, "__name__", "") or ""
    return name.split(".", 1)[0] or "opend"
