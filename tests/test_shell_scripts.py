import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from yam_arena.inspect_embodiment import DESKTOP_DISPLAY, DESKTOP_SIZE, DESKTOP_SOCKET

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "scripts" / "desktop.sh"
LAUNCHABLE = ROOT / "scripts" / "launchable.sh"
SCRIPTS = [ROOT / "setup.sh", DESKTOP, LAUNCHABLE]


@pytest.mark.parametrize("script", SCRIPTS)
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


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_shell_scripts_are_executable(script):
    assert script.stat().st_mode & 0o111


def test_the_launchable_installs_with_the_desktop_and_loads_the_environment_setup_writes():
    text = LAUNCHABLE.read_text()
    assert 'bash "$REPO/setup.sh" --desktop' in text
    assert "export ARENA_DIR=$HOME/IsaacLab-Arena" in text
    assert 'ENV_LINE="source $ARENA_DIR/yam.env"' in text
    assert '>"$ARENA_DIR/yam.env"' in (ROOT / "setup.sh").read_text()


def test_the_launchable_fetches_itself_from_this_repository():
    text = LAUNCHABLE.read_text()
    assert "kairosiann/yam-isaac-lab-arena/$REF/scripts/launchable.sh" in text
    assert "kairosiann/yam-isaac-lab-arena/main/scripts/launchable.sh | bash" in text


def _desktop_checks_can_run():
    if not (shutil.which("apt-get") and Path("/run/systemd/system").is_dir() and shutil.which("systemctl")):
        return False
    probe = subprocess.run(["systemctl", "is-active", "--quiet", "arena-desktop.service"])
    return probe.returncode != 0  # with our own desktop running, setup.sh skips the check


@pytest.mark.skipif(not _desktop_checks_can_run(), reason="needs apt-get and systemd, without arena-desktop running")
def test_setup_desktop_refuses_a_port_another_desktop_holds_before_installing_anything():
    with socket.socket() as other_desktop:
        try:
            other_desktop.bind(("127.0.0.1", 6080))
        except OSError:
            pytest.skip("port 6080 is in use on this machine")
        other_desktop.listen()
        proc = subprocess.run(["bash", str(ROOT / "setup.sh"), "--desktop"], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 1
    assert "port 6080 already in use by another program" in proc.stderr
