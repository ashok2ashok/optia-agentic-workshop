# Optia Agentic Workshop - Build a Working AI Agent

> *"Optia" is a fictional eye-care company used purely as a scenario for this workshop. The product
> names, feedback, and data are all invented for teaching and are not affiliated with any real company.*

Five hands-on build-alongs. Each table builds **one** real AI agent - in Python, on the Anthropic API -
the way agents are actually built: a model that runs in a **loop** (it can call **tools**) with a
**guardrail** wrapping it, developed **evals-first** so you can *measure* it getting better.

You do not need to be an expert. Every notebook is three small edits (**✏️ YOUR TURN**): the agent's
**brief** (plain English), its **tool descriptions**, and **one line** of the guardrail. After each, you
re-run an eval and watch a progress bar climb from a weak baseline to a working, guarded agent.

## The five agents
| Group | Agent | Signature move |
|---|---|---|
| 1 | [Voice-of-Customer](group1_voice_of_customer/) | catches a safety comment and flags it for human review |
| 2 | [Requirement-to-Story](group2_requirement_to_story/) | separates what was asked from invented scope ("assumptions to confirm") |
| 3 | [Market & Competitive Scan](group3_market_scan/) | labels every item confirmed vs unverified, with sources |
| 4 | [Launch & Release Comms](group4_launch_comms/) | never reworders a locked clinical claim; compliance banner on patient copy |
| 5 | [Roadmap Tradeoff](group5_roadmap_tradeoff/) | presents scenarios with tradeoffs - AI recommends, humans decide |

## The arc (every notebook)
| Stage | You add | Eval score |
|---|---|---|
| 0 · Baseline *(given)* | naive prompt, no tools | low - the starting line |
| 1 · Instructions | the brief | ↑ |
| 2 · Tools | the tool descriptions | ↑↑ |
| 3 · Guardrail | one safety-check line | ↑↑↑ target |

## How an agent works (90 seconds)
- **The model** reasons and writes. On its own it can only *talk*.
- **Tools** are functions it can call - that's how it *acts* (look something up, record a result, escalate).
- **The loop:** reason → call a tool → see the result → repeat, until done.
- **Guardrails** wrap the loop: plain rules that check/repair the output for high-stakes cases.
- **Evals** are how you *know* it works: test tasks + graders → a baseline → measured improvement.

## Prerequisites
- **Python 3.10+**
- An **Anthropic API key** (get one at https://console.anthropic.com/). Running a full notebook makes
  real API calls and consumes a small amount of credit - a few cents per group with the default models.
- That's it. Each notebook installs its one dependency (`anthropic`) on first run.

## Run it
1. Install once - see [SETUP.md](SETUP.md) (`python3 -m venv .venv && pip install -r requirements.txt`).
2. `cp .env.example .env`, open your group's folder, open the `.ipynb`, run the Setup cell, paste your key into `.env`.
3. Run top to bottom; do the three **✏️ YOUR TURN** cells.

Each group folder has its own README. Solutions live beside each notebook as `<Name>.py` (the answer key).

## License
MIT - see [LICENSE](LICENSE). Free to fork, adapt, and run.
