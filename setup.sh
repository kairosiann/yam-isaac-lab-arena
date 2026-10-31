#!/usr/bin/env bash
# Install Isaac Lab-Arena 0.3.1 natively with uv, this package, and the Inspect Robots policies
# for Dream-1 (`dream`) and OpenRouter models (`agent`). Run it from a checkout of this repository.
# Re-running it is safe; it picks up where an interrupted run stopped.
#
#   ARENA_DIR   where Arena is cloned (default ~/IsaacLab-Arena); its .venv is the environment
#   DREAM_SPEC  where inspect-robots-dream comes from: a pip URL or a local checkout
#               (default: its GitHub repository)
set -euo pipefail

ARENA_REPO=https://github.com/isaac-sim/IsaacLab-Arena.git
ARENA_REF=release/0.3.1
ARENA_DIR=${ARENA_DIR:-$HOME/IsaacLab-Arena}
DREAM_SPEC=${DREAM_SPEC:-git+https://github.com/kairosiann/inspect-robots-dream}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

die() {
  echo "setup.sh: $*" >&2
  exit 1
}
step() { echo "==> $*"; }
# Arena's submodules use SSH URLs; fetch them over HTTPS so no GitHub key is needed.
git_https() { git -c url."https://github.com/".insteadOf=git@github.com: "$@"; }

[[ -f $HERE/pyproject.toml && -d $HERE/yam_arena ]] || die "run setup.sh from a checkout of yam-arena"
command -v git >/dev/null || die "git is required"
nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed: install the NVIDIA driver first"
PATH=$HOME/.local/bin:$PATH # where uv's installer puts uv
if ! command -v uv >/dev/null; then
  step "Installing uv"
  command -v curl >/dev/null || die "curl is required to install uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v uv >/dev/null || die "uv is installed but not on PATH; open a new shell and re-run"
fi

if ! command -v cmake >/dev/null; then
  step "Installing cmake"
  uv tool install cmake
  PATH=$(uv tool dir --bin):$PATH
  command -v cmake >/dev/null || die "cmake is installed but not on PATH"
fi

mkdir -p "$(dirname "$ARENA_DIR")"
ARENA_DIR=$(cd "$(dirname "$ARENA_DIR")" && pwd)/$(basename "$ARENA_DIR")
if [[ -e $ARENA_DIR/.git ]]; then
  branch=$(git -C "$ARENA_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null) ||
    die "$ARENA_DIR is a broken git checkout; remove it and re-run"
  [[ $branch == "$ARENA_REF" ]] || die "$ARENA_DIR is on '$branch', not $ARENA_REF; set ARENA_DIR to another directory"
elif [[ -d $ARENA_DIR && -n $(ls -A "$ARENA_DIR") ]]; then
  die "$ARENA_DIR exists and is not a git checkout; set ARENA_DIR to another directory"
else
  step "Cloning Isaac Lab-Arena ($ARENA_REF) into $ARENA_DIR"
  git_https clone --branch "$ARENA_REF" "$ARENA_REPO" "$ARENA_DIR"
fi
step "Fetching Arena's submodules"
git_https -C "$ARENA_DIR" submodule update --init --recursive

step "Installing Arena's locked environment"
# --inexact keeps the packages installed below when this is re-run. robomimic, which Arena pulls in,
# depends on egl-probe: it builds from source with CMake and declares a minimum version CMake 4 refuses.
(cd "$ARENA_DIR" && CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --frozen --inexact)
VENV=$ARENA_DIR/.venv
[[ -x $VENV/bin/python ]] || die "uv sync did not create $VENV"

step "Installing yam-arena and the dream and agent policies"
if [[ -d $DREAM_SPEC ]]; then
  DREAM_SPEC=file://$(cd "$DREAM_SPEC" && pwd)
fi
uv pip install --python "$VENV/bin/python" --override "$HERE/overrides.txt" \
  -e "$HERE" "inspect-robots-dream[client] @ $DREAM_SPEC" inspect-robots-agent

step "Putting Arena's ffmpeg (imageio-ffmpeg) on the venv's PATH, for videos in trial reports"
ffmpeg=$("$VENV/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())') ||
  die "imageio-ffmpeg did not provide an ffmpeg binary"
ln -sf "$ffmpeg" "$VENV/bin/ffmpeg"
"$VENV/bin/ffmpeg" -version >/dev/null || die "$VENV/bin/ffmpeg does not run"

printf 'source %q\nexport OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y\n' "$VENV/bin/activate" >"$ARENA_DIR/yam.env"

embodiments=$("$VENV/bin/inspect-robots" list embodiments) || die "inspect-robots does not start"
grep -Eq '^[[:space:]]*-[[:space:]]+yam_arena$' <<<"$embodiments" || die "yam_arena is not a registered embodiment"
echo "Done. Start each shell with:  source $ARENA_DIR/yam.env"
