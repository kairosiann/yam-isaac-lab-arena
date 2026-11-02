#!/usr/bin/env bash
# Setup script for a Brev launchable: turns a fresh Brev GPU VM into a ready YAM workstation.
#
# It runs `./setup.sh --desktop` (Isaac Lab-Arena, the yam_arena embodiment, the dream, agent and
# yam_scripted policies, and the browser desktop), loads the environment in every new shell, and
# plays the scripted demo once headless and once in a window. Isaac Sim's first start in each mode
# takes about 4 minutes, so those warm-up runs spare a team's first runs that wait.
#
# Everything goes to ~/yam-setup.log, which ends with "yam-setup: done" when the VM is ready.
#
# Paste this file as the launchable's setup script, or a line that fetches it:
#   curl -fsSL https://raw.githubusercontent.com/kairosiann/yam-isaac-lab-arena/main/scripts/launchable.sh | bash
# It also works on any Brev VM:  bash scripts/launchable.sh
#
#   YAM_ARENA_REF   branch, tag or commit to install when this script clones the repo (default main)
#   YAM_WARMUP=0    skip the two warm-up runs
set -euo pipefail

REPO_URL=https://github.com/kairosiann/yam-isaac-lab-arena.git
REF=${YAM_ARENA_REF:-main}
SCRIPT_URL=https://raw.githubusercontent.com/kairosiann/yam-isaac-lab-arena/$REF/scripts/launchable.sh
LOGIN_USER=ubuntu

die() {
  echo "launchable.sh: $*" >&2
  exit 1
}

# Brev may run a setup script as root. Install for the login user, who does the runs.
if [[ $(id -u) == 0 ]]; then
  id "$LOGIN_USER" >/dev/null 2>&1 || die "run this as the login user, or as root on a VM with a '$LOGIN_USER' user"
  self=/tmp/yam-launchable.sh
  if [[ -f ${BASH_SOURCE[0]:-} ]]; then
    install -m 755 "${BASH_SOURCE[0]}" "$self"
  else # piped in (`curl ... | bash`): fetch a copy the login user can run
    curl -fsSL "$SCRIPT_URL" -o "$self" && chmod 755 "$self"
  fi
  exec sudo -u "$LOGIN_USER" -H --preserve-env=YAM_ARENA_REF,YAM_WARMUP bash "$self" "$@"
fi

HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
cd "$HOME"
LOG=$HOME/yam-setup.log
exec > >(tee -a "$LOG") 2>&1
step() { echo && echo "=== yam-setup: $* ($((SECONDS / 60)) min)"; }
step "started $(date -u '+%F %T UTC') as $(id -un), log in $LOG"

# This repo: the checkout this script lives in, else the one Brev clones for the launchable, else
# a fresh clone.
is_checkout() { [[ -f $1/setup.sh && -f $1/yam_arena/inspect_embodiment.py ]]; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null && pwd || true)
if [[ -n $HERE ]] && is_checkout "$HERE/.."; then
  REPO=$(cd "$HERE/.." && pwd)
elif is_checkout "$HOME/yam-isaac-lab-arena"; then
  REPO=$HOME/yam-isaac-lab-arena
else
  REPO=$HOME/yam-isaac-lab-arena
  [[ ! -e $REPO ]] || die "$REPO exists but is not a checkout of yam-isaac-lab-arena"
  step "cloning $REPO_URL ($REF)"
  git clone --quiet "$REPO_URL" "$REPO"
  git -C "$REPO" checkout --quiet "$REF"
fi
step "installing from $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"

export ARENA_DIR=$HOME/IsaacLab-Arena
bash "$REPO/setup.sh" --desktop

step "loading the environment in every new shell"
ENV_LINE="source $ARENA_DIR/yam.env"
grep -qxF "$ENV_LINE" "$HOME/.bashrc" 2>/dev/null ||
  printf '\n# yam-isaac-lab-arena (scripts/launchable.sh)\n%s\n' "$ENV_LINE" >>"$HOME/.bashrc"

if [[ ${YAM_WARMUP:-1} != 0 ]]; then
  WARM=$(mktemp -d)
  for headless in true false; do
    step "warm-up run with headless=$headless (Isaac Sim's first start in this mode)"
    if (
      set +u
      # shellcheck disable=SC1091
      source "$ARENA_DIR/yam.env"
      timeout 1200 inspect-robots run --embodiment yam_arena --policy yam_scripted \
        -E scene=station -E object_jitter=0 -E headless=$headless \
        -E control_hz=20 -E cam_width=320 -E cam_height=176 \
        --instruction "put apple into the bowl" --scorer success_at_end \
        --max-steps 600 --log-dir "$WARM"
    ); then
      echo "warm-up run with headless=$headless finished"
    else
      echo "WARNING: the warm-up run with headless=$headless failed; the install is still usable," \
        "but the first run in this mode will take a few minutes longer"
    fi
  done
  rm -rf "$WARM"
fi

step "done. Open a new shell (brev shell <instance>) and run inspect-robots; see the README's Quickstart"
