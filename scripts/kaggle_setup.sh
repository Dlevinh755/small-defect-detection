#!/usr/bin/env bash
# Called by the Setup cell of notebooks/sdd_end_to_end.ipynb:  !bash <repo>/scripts/kaggle_setup.sh
#
# Env vars (all optional):
#   SDD_REPO      git URL to clone (default: use the code already present in SDD_CODE_INPUT)
#   SDD_CODE_INPUT  Kaggle input folder containing this repo (uploaded as a dataset), e.g. /kaggle/input/sdd-code
#   SDD_PREV      space-separated result folders of previous sessions to resume from,
#                 e.g. "/kaggle/input/sdd-p1-session1/results"
set -euo pipefail
DEST=/kaggle/working/sdd

if [ ! -d "$DEST" ]; then
  if [ -n "${SDD_REPO:-}" ]; then
    git clone -q "$SDD_REPO" "$DEST"
  elif [ -n "${SDD_CODE_INPUT:-}" ]; then
    cp -r "$SDD_CODE_INPUT" "$DEST"
  else
    echo "Set SDD_REPO or SDD_CODE_INPUT" >&2; exit 1
  fi
fi
cd "$DEST"
pip install -q -r requirements.txt
pip install -q -e . --no-deps
python - <<'PY'
from sdd.env import versions, paths
print(versions()); print("results ->", paths().results)
PY
nvidia-smi --query-gpu=index,name,memory.total --format=csv
if [ -n "${SDD_PREV:-}" ]; then
  python scripts/sync_results.py $SDD_PREV
fi
