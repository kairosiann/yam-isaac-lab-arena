import json
import subprocess
import sys

import numpy as np

from yam_arena.checks import Checks


def test_summary_is_printed_and_written_as_json(tmp_path, capsys):
    checks = Checks("stage_x", tmp_path)
    assert checks.check("limits", True, worst=np.float64(0.0123456789), per_link={"link1": np.array([1.5, 2.25])})
    assert not checks.check("fk", False, "3 mm off")
    checks.skip("cameras", "no GPU")
    checks.note("seconds", 12.3456789)
    assert checks.finish(exit_process=False) == 1

    out = capsys.readouterr().out
    assert "[stage_x] FAIL: fk -- 3 mm off" in out
    assert "===== stage_x SUMMARY: FAIL (1 pass, 1 fail, 1 skipped," in out
    summary = json.loads((tmp_path / "stage_x_summary.json").read_text())
    assert summary["verdict"] == "FAIL"
    assert [r["status"] for r in summary["results"]] == ["PASS", "FAIL", "SKIPPED"]
    assert summary["results"][0]["worst"] == 0.012346
    assert summary["results"][0]["per_link"] == {"link1": [1.5, 2.25]}
    assert summary["info"] == {"seconds": 12.345679}


def test_a_stage_that_checks_nothing_fails():
    checks = Checks("empty")
    assert not checks.passed
    assert checks.finish(exit_process=False) == 1


def test_skipped_checks_do_not_fail_a_stage():
    checks = Checks("stage_y")
    checks.check("hold", True)
    checks.skip("agent run", "OPENROUTER_API_KEY is not set")
    assert checks.passed
    assert checks.finish(exit_process=False) == 0


def test_an_exception_fails_the_stage_with_its_traceback(capsys):
    checks = Checks("stage_z")
    try:
        raise ValueError("boom")
    except ValueError as exc:
        checks.error("part a", exc)
    assert not checks.passed
    assert checks.results == [{"check": "part a", "status": "FAIL", "detail": "exception: ValueError('boom')"}]
    assert "Traceback" in capsys.readouterr().out


def test_finish_flushes_the_summary_then_ends_the_process():
    script = "from yam_arena.checks import Checks; c = Checks('s'); c.check('a', False); c.finish(); print('after')"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1
    assert "===== s SUMMARY: FAIL" in proc.stdout
    assert "after" not in proc.stdout
