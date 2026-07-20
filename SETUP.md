# Setup & Troubleshooting

One page to get any laptop - including a locked-down corporate one - running these notebooks.
**Fastest reliable path: a virtual environment in VS Code.**

## The 4-command setup (once)
```bash
python3 -m venv .venv                 # isolated environment
source .venv/bin/activate             # macOS/Linux  (Windows: .venv\Scripts\activate)
pip install -r requirements.txt       # install deps
cp .env.example .env                  # then paste your key into .env (see below)
```
Open any group's `.ipynb`, select the `.venv` interpreter as the kernel (top-right), Run All.
First cell should show `✓ Dependencies ready` then `✓ API key verified`.

## API key
Copy the template and paste your key:
```bash
cp .env.example .env      # then open .env and paste your key after ANTHROPIC_API_KEY=
```
Or just run the Setup cell - it creates the gitignored **`.env`** for you the first time. Either way,
open `.env`, paste your key after `ANTHROPIC_API_KEY=`, save, re-run the cell. One `.env` serves all five groups.
Never paste a key into a notebook cell. A key starts with `sk-ant-`.

**Important:** a shell/kernel `ANTHROPIC_API_KEY` **overrides** `.env`. If the setup cell says a key is
rejected but you're sure `.env` is right, you probably have a stale key exported in your environment -
`unset ANTHROPIC_API_KEY` (or fix it). The rejection message tells you which source and the last 4 chars.

## Common problems
- **`externally-managed-environment` (PEP 668):** the setup cell auto-installs to user space on first run. Cleanest fix is the venv above.
- **Wrong kernel (VS Code):** click the kernel name (top-right) → Select Another Kernel → Python Environments → pick the one ending in `.venv`.
- **No admin rights:** a venv is yours, needs no admin. Otherwise the notebook's `--user` fallback handles it.
- **Corporate proxy / PyPI blocked:** `export HTTPS_PROXY=...` before `pip install`, or use your internal mirror: `pip install -r requirements.txt --index-url https://<mirror>/simple`.
- **Python version:** need 3.10+. Check `python3 --version`.
