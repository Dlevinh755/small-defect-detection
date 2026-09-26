#!/usr/bin/env bash
# Called by the Setup cell of notebooks/sdd_end_to_end.ipynb, AFTER the repo is cloned / copied:
#   !bash <repo>/scripts/kaggle_setup.sh
# Installs the package from the repo this script lives in and syncs previous results.
#
# Env vars (optional):
#   SDD_PREV   space-separated result folders of previous sessions to resume from,
#              e.g. "/kaggle/input/sdd-e2e-v3/results"
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
pip install -q -r requirements.txt
pip install -q -e . --no-deps
python - <<'PY'
from sdd.env import versions, paths
print(versions()); print("results ->", paths().results)
PY
nvidia-smi --query-gpu=index,name,memory.total --format=csv || echo "no GPU"
if [ -n "${SDD_PREV:-}" ]; then
  python scripts/sync_results.py $SDD_PREV
fi
