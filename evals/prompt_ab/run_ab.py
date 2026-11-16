#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt A/B harness: run frozen Chan scenarios through the real analyzer and score them.

Compare two prompt versions by running each from its own checkout (or git worktree)
with a different ``--tag``, then ``compare`` the two result files. Each case runs
``--repeats`` times so the run-to-run noise of one prompt can be seen next to the
difference between prompts. Results keep the full parsed output, so ``rescore``
can re-apply changed metrics without new LLM calls.

Needs a configured LLM (``LITELLM_MODEL`` + provider key in ``.env``).

Usage:
    python evals/prompt_ab/run_ab.py run --tag new --repeats 3 --out ab_new.json
    python evals/prompt_ab/run_ab.py rescore ab_new.json
    python evals/prompt_ab/run_ab.py compare ab_old.json ab_new.json

Exit codes: 0 = ran, 1 = no LLM configured / bad input, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals.prompt_ab.cases import build_cases  # noqa: E402
from evals.prompt_ab.metrics import score_result  # noqa: E402


def run_cases(
    analyze: Callable[[Dict[str, Any], Optional[str]], Any],
    cases: List[Dict[str, Any]],
    repeats: int,
    gate: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Run every case ``repeats`` times through ``analyze`` and score the outputs."""
    records = []
    for case in cases:
        context = {k: v for k, v in case["context"].items() if not k.startswith("_")}
        news = case["context"].get("_news")
        for attempt in range(repeats):
            result = analyze(context, news)
            pre_gate = str(getattr(result, "decision_type", "") or "")
            if gate is not None:
                gate(result, context.get("chan_snapshot"))
            payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            payload["success"] = bool(getattr(result, "success", payload.get("success", True)))
            records.append({
                "case": case["id"], "attempt": attempt, "pre_gate_decision": pre_gate,
                "result": payload, "score": score_result(case, payload),
            })
    return records


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_case.setdefault(rec["case"], []).append(rec)
    out = {}
    for case_id, recs in sorted(by_case.items()):
        decisions = Counter(r["score"]["decision"] for r in recs)
        top = decisions.most_common(1)[0][1] if decisions else 0
        out[case_id] = {
            "n": len(recs),
            "pass_rate": round(sum(r["score"]["pass"] for r in recs) / len(recs), 3),
            "decisions": dict(decisions),
            "self_agreement": round(top / len(recs), 3),  # run-to-run noise floor
            "invented_probability": sum(r["score"]["invented_probability"] for r in recs),
            "invented_ids": sum(len(r["score"]["invented_evidence_ids"]) for r in recs),
            "gate_changed": sum(r["pre_gate_decision"] != r["score"]["decision"] for r in recs),
        }
    return out


def _cmd_run(args) -> int:
    if args.language:
        os.environ["REPORT_LANGUAGE"] = args.language
    from src.analyzer import GeminiAnalyzer, enforce_chan_buy_gate

    analyzer = GeminiAnalyzer()
    if not analyzer.is_available():
        print("No LLM configured: set LITELLM_MODEL and the provider key in .env", file=sys.stderr)
        return 1
    cases = build_cases()
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    records = run_cases(lambda ctx, news: analyzer.analyze(ctx, news_context=news), cases, args.repeats,
                        gate=enforce_chan_buy_gate)
    summary = summarise(records)
    Path(args.out).write_text(json.dumps({"tag": args.tag, "summary": summary, "records": records},
                                         ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def _cmd_rescore(args) -> int:
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in build_cases()}
    for rec in data["records"]:
        rec["score"] = score_result(cases[rec["case"]], rec["result"])
    data["summary"] = summarise(data["records"])
    Path(args.file).write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(data["summary"], ensure_ascii=False, indent=1))
    return 0


def _cmd_compare(args) -> int:
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    print(f"case\t{a.get('tag')} pass\t{b.get('tag')} pass\tdelta\t{a.get('tag')} agree\t{b.get('tag')} agree")
    for case_id in sorted(set(a["summary"]) | set(b["summary"])):
        sa, sb = a["summary"].get(case_id, {}), b["summary"].get(case_id, {})
        pa, pb = sa.get("pass_rate"), sb.get("pass_rate")
        delta = round(pb - pa, 3) if pa is not None and pb is not None else ""
        print(f"{case_id}\t{pa}\t{pb}\t{delta}\t{sa.get('self_agreement')}\t{sb.get('self_agreement')}")
    print("\nA difference smaller than 1 - self_agreement is within run-to-run noise.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--tag", required=True)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--cases", help="comma-separated case ids (default: all)")
    run.add_argument("--language", choices=["zh", "en"], help="override REPORT_LANGUAGE")
    run.add_argument("--out", required=True)
    rescore = sub.add_parser("rescore")
    rescore.add_argument("file")
    compare = sub.add_parser("compare")
    compare.add_argument("a")
    compare.add_argument("b")
    args = parser.parse_args(argv)
    return {"run": _cmd_run, "rescore": _cmd_rescore, "compare": _cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
