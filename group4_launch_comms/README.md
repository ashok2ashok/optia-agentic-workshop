# Launch & Release Comms Agent · Group 4

**Level: Intermediate · ~40 min · Build-Along**

## The pain
Every launch, the same update gets rewritten for five audiences - executives, ECPs, internal ops, patients, sales. It's slow, repetitive, and tone drifts. Worse, a patient version can quietly overstate a clinical benefit.

## Your mission
Build an agent that turns one source of truth (a feature list + locked clinical facts) into audience-tailored comms - right message, tone, and detail for each - without ever altering a regulated claim.

## What "great" looks like
The locked-facts rule plus the compliance banner on the patient version. Highest-stakes agent of the day: regulated claims cannot be reworded or embellished by the model.

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
1. Open **`Launch_Comms_Agent.ipynb`** in VS Code / Cursor (Python + Jupyter extensions).
2. Run the **Setup** cell; paste your Anthropic key into the `.env` it creates (never into a cell).
3. Run top to bottom. Do the three **✏️ YOUR TURN** cells. Watch the score climb.

Want the answer key? **`Launch_Comms_Agent.py`** is the completed solution - run it straight through with
`python3 Launch_Comms_Agent.py`.
