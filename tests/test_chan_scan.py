# -*- coding: utf-8 -*-
"""Chan index pre-screen (src/chan/scan.py) and the main.py --scan mode. No network."""

import argparse
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import src.chan.scan as scan
from src.chan.scan import (
    ScanResult,
    download_daily_frames,
    load_universe,
    parse_scan_universes,
    render_scan_markdown,
    save_scan_outputs,
    scan_frames,
)


def test_parse_scan_universes_accepts_aliases_and_rejects_unknown():
    assert parse_scan_universes("sp500, NDX,snp500") == ["sp500", "nasdaq100"]
    assert parse_scan_universes("S&P500,nasdaq") == ["sp500", "nasdaq100"]
    with pytest.raises(ValueError):
        parse_scan_universes("dow30")
    with pytest.raises(ValueError):
        parse_scan_universes(" , ")


def test_load_universe_uses_fresh_cache_refetches_stale_and_falls_back(tmp_path):
    now = datetime(2026, 11, 17, 9, 0)
    calls = []

    def fetcher(name):
        calls.append(name)
        return ["AAPL", "BRK.B"]

    assert load_universe(["sp500"], tmp_path, fetcher=fetcher, now=now) == ["AAPL", "BRK.B"]
    assert calls == ["sp500"]
    # fresh cache: no second fetch
    assert load_universe(["sp500"], tmp_path, fetcher=fetcher, now=now + timedelta(days=1)) == ["AAPL", "BRK.B"]
    assert calls == ["sp500"]

    def broken(name):
        raise OSError("403")

    # stale cache + failed fetch -> stale cache
    assert load_universe(["sp500"], tmp_path, fetcher=broken, now=now + timedelta(days=30)) == ["AAPL", "BRK.B"]
    # no cache at all + failed fetch -> error
    with pytest.raises(RuntimeError):
        load_universe(["nasdaq100"], tmp_path, fetcher=broken, now=now)


def test_download_daily_frames_maps_yahoo_symbols_back_and_skips_missing():
    idx = pd.date_range("2026-01-02", periods=3, freq="B")
    cols = pd.MultiIndex.from_product([["AAPL", "BRK-B"], ["Open", "High", "Low", "Close", "Volume"]])
    data = pd.DataFrame(np.arange(30, dtype=float).reshape(3, 10) + 1, index=idx, columns=cols)
    data.loc[idx[0], ("AAPL", "Close")] = np.nan
    seen = {}

    def downloader(symbols, **kwargs):
        seen["symbols"], seen["kwargs"] = symbols, kwargs
        return data

    frames = download_daily_frames(["AAPL", "BRK.B", "GONE"], downloader=downloader)
    assert seen["symbols"] == ["AAPL", "BRK-B", "GONE"]
    assert seen["kwargs"]["auto_adjust"] is True
    assert set(frames) == {"AAPL", "BRK.B"}
    assert len(frames["AAPL"]) == 2  # NaN close row dropped
    assert list(frames["BRK.B"].columns) == ["date", "open", "high", "low", "close", "volume"]


def _point(kind, state, bars_since, valid=True, price=10.0, invalidation=9.0):
    return {"evidence_id": f"D.{kind}.x", "type": kind, "side": "buy" if kind.endswith("B") else "sell",
            "state": state if valid else "invalidated", "still_valid": valid, "dt": "2026-11-10",
            "price": price, "invalidation_price": invalidation, "bars_since": bars_since}


def test_scan_frames_filters_ranks_and_caps(monkeypatch):
    levels = {
        "AAA": {"status": "available", "last_close": 10.0, "last_bar": "2026-11-16", "buy_sell_points": [
            _point("2B", "candidate", 1), _point("3B", "confirmed", 3, invalidation=9.5)]},
        "BBB": {"status": "available", "last_close": 10.0, "last_bar": "2026-11-16", "buy_sell_points": [
            _point("1B", "confirmed", 1, invalidation=8.0)]},
        "CCC": {"status": "available", "last_close": 10.0, "last_bar": "2026-11-16", "buy_sell_points": [
            _point("1B", "confirmed", 9), _point("2B", "confirmed", 0, valid=False), _point("1S", "confirmed", 0,
                                                                                            invalidation=11.0)]},
        "DDD": {"status": "missing", "buy_sell_points": []},
        "EEE": {"status": "available", "last_close": 10.0, "last_bar": "2026-11-16", "buy_sell_points": [
            _point("2B", "forming", 0)]},
    }
    monkeypatch.setattr(scan, "build_chan_snapshot",
                        lambda frames, code, market=None: {"levels": {"D": levels[code]}})
    result = scan_frames({code: None for code in levels}, max_bars_since=5, max_candidates=1, universes=["sp500"])

    assert result.scanned == 4 and result.unusable == ["DDD"] and result.as_of == "2026-11-16"
    # a forming point is logged but is never a candidate (the buy gate cannot act on it)
    assert [p["state"] for p in result.points if p["code"] == "EEE"] == ["forming"]
    # confirmed beats candidate; same state -> more recent; AAA keeps its confirmed 3B
    assert [(r["code"], r["type"]) for r in result.candidates] == [("BBB", "1B"), ("AAA", "3B")]
    assert result.selected_codes == ["BBB"]
    assert result.candidates[1]["risk_to_invalidation_pct"] == 5.0
    # CCC: old buy point and invalidated one dropped; its recent sell point is still recorded
    assert [(p["code"], p["side"]) for p in result.points if p["code"] == "CCC"] == [("CCC", "sell")]


def test_scan_frames_runs_the_real_engine_on_a_synthetic_series():
    closes = 100 + 10 * np.sin(np.linspace(0, 12 * np.pi, 250))
    frame = pd.DataFrame({"date": pd.date_range("2025-11-01", periods=250, freq="B"), "open": closes,
                          "high": closes + 1, "low": closes - 1, "close": closes, "volume": 1.0})
    result = scan_frames({"ZZZ": frame}, max_bars_since=250)
    assert result.scanned == 1 and not result.unusable
    assert all(p["code"] == "ZZZ" and p["bars_since"] <= 250 for p in result.points)


def test_save_scan_outputs_writes_markdown_and_json(tmp_path):
    row = {"code": "BBB", "evidence_id": "D.1B.x", "type": "1B", "side": "buy", "state": "confirmed",
           "dt": "2026-11-10", "price": 10.0, "invalidation_price": 8.0, "last_close": 10.0,
           "risk_to_invalidation_pct": 20.0, "bars_since": 1, "trend_type": "down",
           "position_vs_last_zhongshu": "below"}
    result = ScanResult(universes=["sp500"], universe_size=2, scanned=2, max_bars_since=5, max_candidates=0,
                        points=[row], candidates=[row], as_of="2026-11-16")
    paths = save_scan_outputs(result, tmp_path, language="en", now=datetime(2026, 11, 17))
    assert paths["markdown"].name == "chan_scan_20261117.md"
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "| 1 | BBB | 1B | confirmed |" in text and "no (over cap)" in text
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["selected"] == [] and payload["points"][0]["evidence_id"] == "D.1B.x"
    assert "缠论日线扫描" in render_scan_markdown(result, "zh")


def _args(**overrides):
    base = dict(scan="sp500", scan_max=20, scan_max_age=5, scan_only=False, portfolio=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_main_scan_mode_runs_analysis_only_on_selected_codes(monkeypatch):
    import main

    selected = ScanResult(universes=["sp500"], universe_size=1, scanned=1, max_bars_since=5, max_candidates=20,
                          candidates=[{"code": "AAPL"}])
    monkeypatch.setattr(scan, "run_chan_scan", lambda *a, **k: selected)
    calls = []
    monkeypatch.setattr(main, "_run_analysis_with_runtime_scheduler_lock",
                        lambda config, args, codes: calls.append(codes) or True)
    config = argparse.Namespace(report_language="en")

    assert main._run_chan_scan_mode(config, _args(scan_only=True)) == 0 and calls == []
    assert main._run_chan_scan_mode(config, _args()) == 0 and calls == [["AAPL"]]
    assert main._run_chan_scan_mode(config, _args(portfolio="futu")) == 1

    monkeypatch.setattr(scan, "run_chan_scan", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")))
    assert main._run_chan_scan_mode(config, _args()) == 1


def test_scheduled_scan_skips_closed_us_market_and_raises_on_failure(monkeypatch):
    import main
    import src.core.trading_calendar as cal

    calls = []
    monkeypatch.setattr(main, "_reload_runtime_config", lambda: argparse.Namespace(trading_day_check_enabled=True))
    monkeypatch.setattr(main, "_run_chan_scan_mode", lambda config, args: calls.append(args.scan) or 0)

    monkeypatch.setattr(cal, "get_open_markets_today", lambda: {"cn"})
    main._scheduled_chan_scan_once(_args(force_run=False))
    assert calls == []
    main._scheduled_chan_scan_once(_args(force_run=True))
    assert calls == ["sp500"]

    monkeypatch.setattr(cal, "get_open_markets_today", lambda: {"us"})
    monkeypatch.setattr(main, "_run_chan_scan_mode", lambda config, args: 1)
    with pytest.raises(RuntimeError):
        main._scheduled_chan_scan_once(_args(force_run=False))


def test_scan_schedule_hands_the_scan_task_to_the_scheduler(monkeypatch):
    import main
    import src.scheduler as scheduler

    seen = {}
    monkeypatch.setattr(scheduler, "run_with_schedule", lambda **kwargs: seen.update(kwargs))
    config = argparse.Namespace(schedule_run_immediately=True, schedule_time="06:30")
    assert main._run_chan_scan_schedule(config, _args(no_run_immediately=True)) == 0
    assert seen["schedule_time"] == "06:30" and seen["run_immediately"] is False and callable(seen["task"])
