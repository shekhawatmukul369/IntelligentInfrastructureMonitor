
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  echo "Create the environment first: python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ."
  exit 1
fi
source .venv/bin/activate
streamlit run app.py
