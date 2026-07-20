# Market & Competitive Scan Agent · Group 3

**Level: Intermediate · ~40 min · Build-Along**

## The pain
Staying current on the product area - competitor moves, market signals, regulatory shifts - is a part-time job nobody has. Insights arrive late, or dressed-up rumors get treated as fact.

## Your mission
Build an agent that monitors a set of signals and produces a short "what changed and why it matters" briefing - with every item clearly labeled confirmed vs unverified, and sourced.

## What "great" looks like
Confidence labels and the clean separation of confirmed vs unverified. You make the abstract 'explainability' point concrete: a leader can see at a glance what to trust.

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
1. Open **`Market_Scan_Agent.ipynb`** in VS Code / Cursor (Python + Jupyter extensions).
2. Run the **Setup** cell; paste your Anthropic key into the `.env` it creates (never into a cell).
3. Run top to bottom. Do the three **✏️ YOUR TURN** cells. Watch the score climb.

Want the answer key? **`Market_Scan_Agent.py`** is the completed solution - run it straight through with
`python3 Market_Scan_Agent.py`.
