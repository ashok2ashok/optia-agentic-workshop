# Voice-of-Customer Agent · Group 1

**Level: Intermediate · ~40 min · Build-Along**

## The pain
Feedback about Optia products and clinics arrives from everywhere - app reviews, support tickets, ECP comments, survey free-text. Nobody can read it all, so patterns (and safety signals) get missed until they become problems.

## Your mission
Build an agent that ingests a batch of raw feedback, groups it into themes, ranks them by urgency × frequency, and hands the product team a clear "here's what to pay attention to" summary - while never letting a safety comment slip through.

## What "great" looks like
The demo visibly catches a safety-related comment and flags it for human review - even when the model alone might have missed it. Defense in depth: the guardrail is a rule, not a hope.

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
1. Open **`Voice_Of_Customer_Agent.ipynb`** in VS Code / Cursor (Python + Jupyter extensions).
2. Run the **Setup** cell; paste your Anthropic key into the `.env` it creates (never into a cell).
3. Run top to bottom. Do the three **✏️ YOUR TURN** cells. Watch the score climb.

Want the answer key? **`Voice_Of_Customer_Agent.py`** is the completed solution - run it straight through with
`python3 Voice_Of_Customer_Agent.py`.
