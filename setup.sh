#!/usr/bin/env bash
# Install Isaac Lab-Arena 0.3.1 natively with uv, this package, and the Inspect Robots policies
# for Dream-1 (`dream`) and OpenRouter models (`agent`). Run it from a checkout of this repository.
#
#   ARENA_DIR   where Arena is cloned (default ~/IsaacLab-Arena); its .venv is the environment
#   DREAM_SPEC  where inspect-robots-dream comes from (default: its GitHub repository)
set -euo pipefail

ARENA_REF=release/0.3.1
ARENA_DIR=${ARENA_DIR:-$HOME/IsaacLab-Arena}
DREAM_SPEC=${DREAM_SPEC:-git+https://github.com/kairosiann/inspect-robots-dream}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

command -v nvidia-smi >/dev/null || { echo "setup.sh: no NVIDIA driver found" >&2; exit 1; }
command -v git >/dev/null || { echo "setup.sh: git is required" >&2; exit 1; }
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
fi

if [ ! -d "$ARENA_DIR/.git" ]; then
  git -c url."https://github.com/".insteadOf=git@github.com: clone --branch "$ARENA_REF" \
    --recurse-submodules https://github.com/isaac-sim/IsaacLab-Arena.git "$ARENA_DIR"
fi
(cd "$ARENA_DIR" && uv sync --frozen)

uv pip install --python "$ARENA_DIR/.venv/bin/python" --override "$HERE/overrides.txt" \
  -e "$HERE" "inspect-robots-dream[client] @ $DREAM_SPEC" inspect-robots-agent

cat > "$ARENA_DIR/yam.env" <<EOF
source "$ARENA_DIR/.venv/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
EOF

"$ARENA_DIR/.venv/bin/inspect-robots" list embodiments | grep -q yam_arena
echo "Done. Start each shell with:  source $ARENA_DIR/yam.env"
