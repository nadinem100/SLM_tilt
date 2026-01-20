#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"

focals=(200000)
tilts=(10)
BBOX=(1 2 3)

echo "== SLM sweep start =="
echo "Script dir: $SCRIPT_DIR"
echo "Python: $($PYTHON_BIN -V)"

for f in "${focals[@]}"; do
  for t in "${tilts[@]}"; do
    for b in "${BBOX[@]}"; do
      echo ""
      echo "---- Running: focal_length_um=$f , tilt_angle_x=$t , bbox=$b ----"
      "$PYTHON_BIN" - <<PY
import sys
sys.path.insert(0, "$SCRIPT_DIR")
import final_run_visualize as frv
frv.FOCAL_LENGTH_UM = float(${f})
frv.TILT_ANGLE_X    = float(${t})
frv.BBOX            = float(${b})
print(f"[run_sweep] Using overrides: FOCAL_LENGTH_UM={frv.FOCAL_LENGTH_UM}, TILT_ANGLE_X={frv.TILT_ANGLE_X}, BBOX={frv.BBOX}")
frv.main()
PY
      echo "---- Done: focal_length_um=$f , tilt_angle_x=$t , bbox=$b ----"
    done   # <— this was missing
  done
done

echo ""
echo "== SLM sweep complete ✅ =="
