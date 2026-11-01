import subprocess
from pathlib import Path

import pytest

from yam_arena.inspect_embodiment import DESKTOP_DISPLAY, DESKTOP_SIZE, DESKTOP_SOCKET

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "scripts" / "desktop.sh"


@pytest.mark.parametrize("script", [ROOT / "setup.sh", DESKTOP])
def test_the_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_setup_rejects_an_unknown_option_before_installing_anything():
    proc = subprocess.run(["bash", str(ROOT / "setup.sh"), "--dekstop"], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 1
    assert "unknown option '--dekstop'" in proc.stderr


def test_the_embodiment_looks_for_the_desktop_on_its_display():
    text = DESKTOP.read_text()
    assert f"export DISPLAY={DESKTOP_DISPLAY}" in text
    assert f"SOCKET={DESKTOP_SOCKET}" in text
    assert "-screen 0 {}x{}x24".format(*DESKTOP_SIZE) in text


def test_the_desktop_has_no_password_so_it_listens_on_this_machine_only():
    text = DESKTOP.read_text()
    assert "x11vnc -display :1 -localhost " in text
    assert 'websockify --web "$WEB" "127.0.0.1:$PORT" ' in text


@pytest.mark.parametrize("script", [ROOT / "setup.sh", DESKTOP])
def test_the_shell_scripts_are_executable(script):
    assert script.stat().st_mode & 0o111
