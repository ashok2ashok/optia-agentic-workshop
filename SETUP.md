# Setup & Troubleshooting

One page to get any laptop - including a locked-down corporate one - running these notebooks.
**Fastest reliable path: a virtual environment in VS Code.**

## The 4-command setup (once)
```bash
python3 -m venv .venv                 # isolated environment
source .venv/bin/activate             # macOS/Linux  (Windows: .venv\Scripts\activate)
pip install -r requirements.txt       # install deps
cp .env.example .env                  # then paste an OpenAI or Anthropic key into .env (see below)
```
Open any group's `.ipynb`, select the `.venv` interpreter as the kernel (top-right), Run All.
First cell should show `✓ Dependencies ready` then `✓ Connected via LiteLLM - model: ...`.

## API key (OpenAI or Anthropic)
These notebooks run on either provider through **LiteLLM**. Copy the template and paste ONE key:
```bash
cp .env.example .env
```
Then open `.env` and set one of:
```
OPENAI_API_KEY=sk-...            # OpenAI keys start sk-
ANTHROPIC_API_KEY=sk-ant-...     # Anthropic keys start sk-ant-
# Optional: choose the model (any LiteLLM id). If unset it picks gpt-4o or anthropic/claude-sonnet-5.
WORKSHOP_MODEL=gpt-4o
```
Save and re-run the Setup cell; you want the `✓ Connected via LiteLLM - model: ...` banner. One `.env`
serves all five groups. Never paste a key into a notebook cell.

**Model choice:** `gpt-4o` or `anthropic/claude-sonnet-5` are the reliable defaults. `gpt-4o-mini` is
cheaper and fine for most groups, but the Roadmap-Tradeoff agent (G5) is shaky on it - prefer `gpt-4o` there.

**Important:** a matching key exported in your shell/kernel **overrides** `.env`. If a key is rejected but
`.env` looks right, unset the stale one (e.g. `unset OPENAI_API_KEY` / `unset ANTHROPIC_API_KEY`).

## Common problems
- **`externally-managed-environment` (PEP 668):** the setup cell auto-installs to user space on first run. Cleanest fix is the venv above.
- **Wrong kernel (VS Code):** click the kernel name (top-right) → Select Another Kernel → Python Environments → pick the one ending in `.venv`.
- **No admin rights:** a venv is yours, needs no admin. Otherwise the notebook's `--user` fallback handles it.
- **Corporate proxy / PyPI blocked:** `export HTTPS_PROXY=...` before `pip install`, or use your internal mirror: `pip install -r requirements.txt --index-url https://<mirror>/simple`.
- **Python version:** need 3.10+. Check `python3 --version`.
