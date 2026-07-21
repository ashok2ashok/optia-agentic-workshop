# Requirement-to-Story Agent · Group 2

**Level: Intermediate · ~40 min · Build-Along**

## The pain
Stakeholder requests arrive messy - a paragraph in an email, a hallway chat, a vague "make it easier for clinics to reorder." Turning that into clear, well-formed user stories eats hours of product time.

## Your mission
Build an agent that turns a messy request into structured, decomposed user stories - each with acceptance criteria - ready for the backlog, WITHOUT inventing scope nobody asked for.

## What "great" looks like
The 'Assumptions to confirm' output is the star. The danger isn't bad stories - it's confident invented scope. A great agent separates what was asked from what it assumed.

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
1. Open **`Requirement_To_Story_Agent.ipynb`** in VS Code / Cursor (Python + Jupyter extensions).
2. Run the **Setup** cell; paste your OpenAI or Anthropic key into the `.env` it creates (never into a cell).
3. Run top to bottom. Do the three **✏️ YOUR TURN** cells. Watch the score climb.

Want the answer key? **`Requirement_To_Story_Agent.py`** is the completed solution - run it straight through with
`python3 Requirement_To_Story_Agent.py`.
