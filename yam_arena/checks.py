"""PASS/FAIL bookkeeping for the stage scripts. Pure Python.

Each stage records named checks with a measured value, prints them as it goes, and ends with
a summary block that is flushed *before* the Kit app is touched again: Isaac Sim can hang in
``SimulationApp.close()``, so :meth:`Checks.finish` leaves with ``os._exit``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


class Checks:
    def __init__(self, stage: str, out_dir: str | Path | None = None):
        self.stage = stage
        self.results: list[dict[str, Any]] = []
        self.info: dict[str, Any] = {}
        self.out_dir = Path(out_dir) if out_dir else None
        self.t0 = time.monotonic()

    def check(self, name: str, ok: bool, detail: str = "", **values: Any) -> bool:
        status = "PASS" if ok else "FAIL"
        self.results.append({"check": name, "status": status, "detail": detail, **_jsonable(values)})
        print(f"[{self.stage}] {status}: {name}" + (f" -- {detail}" if detail else ""), flush=True)
        return bool(ok)

    def skip(self, name: str, reason: str) -> None:
        self.results.append({"check": name, "status": "SKIPPED", "detail": reason})
        print(f"[{self.stage}] SKIPPED: {name} -- {reason}", flush=True)

    def note(self, key: str, value: Any) -> None:
        self.info[key] = _jsonable(value)
        print(f"[{self.stage}] INFO {key}: {value}", flush=True)

    def error(self, name: str, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.results.append({"check": name, "status": "FAIL", "detail": f"exception: {exc!r}"})
        print(f"[{self.stage}] FAIL: {name} -- exception {exc!r}\n{tb}", flush=True)

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r["status"] != "FAIL" for r in self.results)

    def finish(self, exit_process: bool = True) -> int:
        n_pass = sum(r["status"] == "PASS" for r in self.results)
        n_fail = sum(r["status"] == "FAIL" for r in self.results)
        n_skip = sum(r["status"] == "SKIPPED" for r in self.results)
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            "",
            f"===== {self.stage} SUMMARY: {verdict} ({n_pass} pass, {n_fail} fail, {n_skip} skipped,"
            f" {time.monotonic() - self.t0:.0f} s) =====",
        ]
        for r in self.results:
            lines.append(f"  {r['status']:7s} {r['check']}" + (f" -- {r['detail']}" if r["detail"] else ""))
        lines.append(f"===== END {self.stage} =====")
        print("\n".join(lines), flush=True)
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            summary = {"stage": self.stage, "verdict": verdict, "results": self.results, "info": self.info}
            (self.out_dir / f"{self.stage}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        sys.stdout.flush()
        sys.stderr.flush()
        code = 0 if self.passed else 1
        if exit_process:
            os._exit(code)  # Kit's app.close() can hang after a headless run
        return code


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, float):
        return round(value, 6)
    return value
