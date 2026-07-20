# Roadmap Tradeoff Agent · Group 5

**Level: Intermediate · ~40 min · Build-Along**

## The pain
Prioritization is painful. Every quarter you juggle competing initiatives, limited capacity, and stakeholders who each think their thing is #1 - and the reasoning behind decisions isn't captured, so you relitigate it.

## Your mission
Build an agent that takes candidate initiatives plus constraints and lays out prioritization OPTIONS with clear rationale - so leaders decide with eyes open. The agent recommends; humans decide.

## What "great" looks like
The agent presents scenarios, not a verdict, and each names what it trades away. The clearest 'AI recommends, human decides' of the day.

## What you'll build
A model that runs in a loop (it can call **tools**) with a **guardrail** wrapping it - built the way
real agents are: **evals first**. You measure a baseline, then improve it in three stages and watch an
eval score climb.

| Stage | You add | The point |
|---|---|---|
| 0 · Baseline *(given)* | a naive prompt, no tools | prove it isn't good enough yet |
| 1 · Instructions | the **brief** (plain English) | biggest single lever |
| 2 · Tools | the tool **descriptions** | now it can *do*, not just talk |
| 3 · Guardrail | **one line** - the safety check | the healthcare-grade net |

Three small edits, all marked **✏️ YOUR TURN**. After each, run the build + eval cells - a rising bar and
green banners mean it's working.

## How to run
1. Open **`Roadmap_Tradeoff_Agent.ipynb`** in VS Code / Cursor (Python + Jupyter extensions).
2. Run the **Setup** cell; paste your Anthropic key into the `.env` it creates (never into a cell).
3. Run top to bottom. Do the three **✏️ YOUR TURN** cells. Watch the score climb.

Want the answer key? **`Roadmap_Tradeoff_Agent.py`** is the completed solution - run it straight through with
`python3 Roadmap_Tradeoff_Agent.py`.
