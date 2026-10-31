#!/bin/bash
# Stage 6, CLI half: inspect-robots sees `yam_arena`; the built-in non-LLM policies are
# incompatible with it; and, only if OPENROUTER_API_KEY is already set, ONE 3-step agent run.
# Run inside the Arena venv:  bash scripts/stage6_cli.sh [out_dir]
OUT=${1:-/data/arena031/out/yam-arena}
mkdir -p "$OUT/ir-logs"
echo "--- inspect-robots list embodiments"
inspect-robots list embodiments
if inspect-robots list embodiments | grep -qE '^\s*-\s*yam_arena$'; then echo "PASS: yam_arena listed"; else echo "FAIL: yam_arena not listed"; fi
echo "--- inspect-robots list policies"
inspect-robots list policies
echo "--- inspect-robots doctor --embodiment yam_arena"
timeout 300 inspect-robots doctor --embodiment yam_arena 2>&1 | tail -20
echo "--- built-in noop policy on yam_arena (expected: rejected by the compatibility check)"
timeout 600 inspect-robots run --embodiment yam_arena --policy noop --instruction "hold still" --max-steps 2 \
  --no-prompt --grader none --scorer episode_length --log-dir "$OUT/ir-logs" > "$OUT/ir-noop.txt" 2>&1
echo "noop EXIT=$?"; grep -iE "compat|error|mismatch|control_mode" "$OUT/ir-noop.txt" | head -8
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  echo "--- agent policy, ONE 3-step run (anthropic/claude-haiku-5.5, effort=low)"
  timeout 1200 inspect-robots run --embodiment yam_arena --policy agent -P model=anthropic/claude-haiku-5.5 -P effort=low \
    --instruction "put the red cube on the plate" --max-steps 3 --no-prompt --grader none --scorer episode_length \
    --log-dir "$OUT/ir-logs"
  echo "agent EXIT=$?"
else
  echo "SKIPPED: agent run (OPENROUTER_API_KEY is not set)"
fi
