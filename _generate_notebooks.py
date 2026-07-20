#!/usr/bin/env python3
"""Generator for the Optia Agentic Workshop.

Emits, per group: README.md + <Stem>.ipynb (blanked, for participants) +
<Stem>.py (solution / answer key). Plus root README/SETUP/requirements/.env/.gitignore/FACILITATOR.

Design: evals-driven agent development. One agent wired up progressively across four stages
(baseline -> instructions -> tools -> guardrail); an eval harness scores it at each stage so
participants watch a progress bar fill. Three small YOUR TURN edits:
  1. write the brief (system prompt, plain English),
  2. write the tool DESCRIPTIONS (schema is pre-filled),
  3. fill ONE line in the guardrail.
Every code cell prints a confirmation of what it did; agent-build and eval-run are separate cells.

USAGE
-----
Edit the GROUPS specs below, then:  python3 _generate_notebooks.py
It overwrites all group files + root files, writing into this file's own directory.
"""
import json, os, pathlib, re

OUT = pathlib.Path(__file__).resolve().parent

# ───────────────────────── shared code blocks ─────────────────────────

SETUP_CODE = r'''# ── Install packages + connect to the AI model ──
# Installs into THIS kernel; survives locked-down (PEP 668) Pythons. Safe to re-run.
import importlib.util, subprocess, sys

def _ensure_packages(reqs):
    missing = [pip for mod, pip in reqs if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print("Installing " + ", ".join(missing) + " - first run only…", flush=True)
    base = [sys.executable, "-m", "pip", "install", "-q"]
    for extra in ([], ["--user"], ["--user", "--break-system-packages"], ["--break-system-packages"]):
        if subprocess.run(base + extra + missing, capture_output=True, text=True).returncode == 0:
            return
    raise SystemExit("Could not install " + ", ".join(missing) + " - see SETUP.md (venv path fixes this).")

_ensure_packages([("anthropic", "anthropic")])
import anthropic, json, os, pathlib, time
print("✓ Dependencies ready")

def _status(ok, msg):
    """Green/red banner in a notebook; plain text as a script."""
    try:
        from IPython import get_ipython
        if get_ipython() is None or get_ipython().__class__.__name__ != "ZMQInteractiveShell":
            raise RuntimeError
        from IPython.display import display, HTML
        c, bg, ic = ("#1a7f37", "#e6f4ea", "✓") if ok else ("#b42318", "#fdecea", "✗")
        display(HTML(f'<div style="padding:10px 14px;border-radius:8px;background:{bg};'
                     f'border:1.5px solid {c};color:{c};font-weight:600;font-family:sans-serif;">{ic} {msg}</div>'))
    except Exception:
        print(("[OK] " if ok else "[!!] ") + msg)

# API key via a gitignored .env at the repo root (paste once, serves every group).
_TEMPLATE = ("# Paste your Anthropic API key after the =, then save & re-run this cell.\n"
             "# Get one at https://console.anthropic.com/ . This file is gitignored.\n"
             "ANTHROPIC_API_KEY=paste-your-key-here\n")

def _resolve_env():
    here = pathlib.Path.cwd().resolve()
    for d in [here, *here.parents]:
        if (d / ".env").is_file():
            return d / ".env"
    root = next((d for d in [here, *here.parents] if (d / "SETUP.md").exists() or (d / ".git").exists()), here)
    return root / ".env"

_env = _resolve_env()
if not _env.exists():
    _env.write_text(_TEMPLATE)
    print(f"Created {_env} - open it, paste your key after ANTHROPIC_API_KEY=, save, re-run this cell.")

_file = {}
for _ln in (_env.read_text().splitlines() if _env.exists() else []):
    _ln = _ln.strip()
    if _ln and not _ln.startswith("#") and "=" in _ln:
        _k, _v = _ln.split("=", 1)
        _file[_k.strip()] = _v.strip().strip('"').strip("'")

# NOTE: a shell/kernel ANTHROPIC_API_KEY OVERRIDES the .env file. If a stale key is exported
# in your environment, editing .env has no effect until you unset it.
_shell = os.environ.get("ANTHROPIC_API_KEY", "").strip()
_filek = _file.get("ANTHROPIC_API_KEY", "").strip()
if _shell.startswith("sk-ant-"):
    api_key, _key_source = _shell, "your shell/kernel environment (ANTHROPIC_API_KEY)"
else:
    api_key, _key_source = _filek, f"the .env file ({_env})"
if not api_key.startswith("sk-ant-"):
    raise SystemExit(f"No API key yet. Open {_env}, paste your key after ANTHROPIC_API_KEY= (starts with sk-ant-), save, re-run.")

client = anthropic.Anthropic(api_key=api_key, timeout=60.0, max_retries=2)
try:
    client.messages.create(model="claude-haiku-4-5", max_tokens=1, messages=[{"role": "user", "content": "ping"}])
except anthropic.AuthenticationError:
    _status(False, f"Key rejected (using {_key_source}, ending …{api_key[-4:]}). "
                   "A shell/kernel ANTHROPIC_API_KEY OVERRIDES .env - if that's the wrong key, unset it "
                   "(or fix it); otherwise paste a valid key into .env and re-run.")
    raise SystemExit("API key not accepted.")
except Exception as e:
    _status(False, f"Could not reach the model API ({type(e).__name__}). Check your connection and re-run.")
    raise
os.environ["ANTHROPIC_API_KEY"] = api_key
_status(True, "API key verified - you're connected.")

MODEL = "claude-sonnet-5"        # the model that powers the agent
JUDGE_MODEL = "claude-haiku-4-5" # a cheaper/faster model used only as an automatic eval judge
'''

HARNESS_CODE = r'''# ── The agent engine + the eval harness (provided - just run it) ──
# run_agent() is the agent loop: it reasons, optionally calls a tool, sees the result, repeats.
# run_eval() scores the agent on the test tasks. You never edit this cell.

RECORDS = []  # tools that "record" something append here; cleared at the start of each run

def run_agent(user_input, system_prompt="", tools=None, tool_functions=None, guardrail=None, max_turns=10):
    """The agent loop: reason -> (maybe call a tool) -> observe -> repeat, with an optional
    guardrail wrapping the final output. Returns {"text": ..., "records": [...]}."""
    RECORDS.clear()
    tools = tools or []
    tool_functions = tool_functions or {}
    messages = [{"role": "user", "content": user_input}]
    final_text = ""
    for _ in range(max_turns):
        kwargs = dict(model=MODEL, max_tokens=4000,
                      system=system_prompt or "You are a helpful assistant.", messages=messages)
        if tools:
            kwargs["tools"] = tools
        resp = client.messages.create(**kwargs)
        if resp.stop_reason == "tool_use":
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    fn = tool_functions.get(b.name)
                    out = fn(**b.input) if fn else json.dumps({"error": "unknown tool " + b.name})
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out)})
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": results})
            continue
        final_text = "".join(b.text for b in resp.content if b.type == "text")
        break
    output = {"text": final_text, "records": list(RECORDS)}
    if guardrail:
        patched = guardrail(user_input, output)   # the guardrail may repair/annotate the output
        if patched is not None:
            output = patched
    return output

def _bar(pct):
    filled = int(round(pct * 20))
    return "█" * filled + "░" * (20 - filled)

_SCORES = {}  # stage label -> pct, for the running scoreboard

def run_eval(agent_fn, label, judge=True):
    """Run the agent over EVERY eval task, grade each output on every grader, and print a
    scorecard: an overall weighted score plus, per grader, how many tasks passed. This is a
    real eval - many test cases, several metrics - not a single happy-path check."""
    got = tot = 0
    agg = {}      # grader name -> [passed_tasks, total_tasks, weight, is_judge]
    order = []
    n_tasks = 0
    for task in EVAL_TASKS:
        out = agent_fn(task["input"])
        n_tasks += 1
        for g in GRADERS:
            if g.get("judge") and not judge:
                continue
            if g["name"] not in agg:
                agg[g["name"]] = [0, 0, g["weight"], bool(g.get("judge"))]
                order.append(g["name"])
            tot += g["weight"]
            try:
                ok = g["fn"](task, out)
            except Exception:
                ok = False
            got += g["weight"] if ok else 0
            a = agg[g["name"]]
            a[0] += 1 if ok else 0
            a[1] += 1
    pct = got / tot if tot else 0.0
    _SCORES[label] = pct
    color = "#1a7f37" if pct >= 0.8 else ("#b58900" if pct >= 0.5 else "#b42318")

    def _icon(p, n):
        return "✓" if p == n else ("◐" if p else "✗")

    try:
        from IPython import get_ipython
        if get_ipython() is None or get_ipython().__class__.__name__ != "ZMQInteractiveShell":
            raise RuntimeError
        from IPython.display import display, HTML
        rows = ""
        for n in order:
            p, N, w, isj = agg[n]
            tag = " · judge" if isj else ""
            c = "#1a7f37" if p == N else ("#b58900" if p else "#b42318")
            rows += (f'<tr><td style="color:{c};padding-right:8px">{_icon(p, N)}</td>'
                     f'<td style="padding-right:10px">{n}<span style="color:#999">{tag}</span></td>'
                     f'<td style="color:#666">{p}/{N} tasks · w{w}</td></tr>')
        display(HTML(
            f'<div style="font-family:sans-serif">'
            f'<div style="font-size:15px;font-weight:700;color:{color}">{label}: {_bar(pct)} {pct*100:.0f}% '
            f'<span style="font-weight:400;color:#666">({n_tasks} tasks × {len(order)} graders)</span></div>'
            f'<table style="font-size:12px;color:#444;border-collapse:collapse;margin-top:6px">{rows}</table></div>'))
    except Exception:
        print(f"{label}: {_bar(pct)} {pct*100:.0f}%  ({n_tasks} tasks × {len(order)} graders)")
        for n in order:
            p, N, w, isj = agg[n]
            print(f"   {_icon(p, N)} {n}{' [judge]' if isj else ''}: {p}/{N} tasks (w{w})")
    return pct

def scoreboard():
    """Show how far the agent has come across every stage you've run."""
    order = ["Stage 0 · Baseline", "Stage 1 · Instructions", "Stage 2 · Tools + loop", "Stage 3 · Guardrail"]
    lines = [(k, _SCORES[k]) for k in order if k in _SCORES]
    try:
        from IPython import get_ipython
        if get_ipython() is None or get_ipython().__class__.__name__ != "ZMQInteractiveShell":
            raise RuntimeError
        from IPython.display import display, HTML
        rows = "".join(f'<tr><td style="padding:2px 10px">{k}</td>'
                       f'<td style="font-family:monospace">{_bar(v)}</td>'
                       f'<td style="font-weight:700">{v*100:.0f}%</td></tr>' for k, v in lines)
        display(HTML(f'<div style="font-family:sans-serif"><div style="font-size:16px;font-weight:700">📈 Your agent\'s progress</div>'
                     f'<table style="border-collapse:collapse;margin-top:6px">{rows}</table></div>'))
    except Exception:
        print("Your agent's progress:")
        for k, v in lines:
            print(f"   {k:26s} {_bar(v)} {v*100:.0f}%")

def judge_yes(question, content):
    """LLM-as-judge grader: ask a cheap model a yes/no question about the output."""
    r = client.messages.create(model=JUDGE_MODEL, max_tokens=5,
        messages=[{"role": "user", "content": f"{question}\n\n---\n{content}\n---\nAnswer with exactly YES or NO."}])
    return "yes" in "".join(b.text for b in r.content if b.type == "text").strip().lower()

def _check(passed, msg):
    _status(passed, msg)
    return passed

print("✓ Agent engine + eval harness ready - run_agent(), run_eval(), scoreboard()")
'''

# ───────────────────────── per-group specs ─────────────────────────

GROUPS = []

# ============================== GROUP 1 ==============================
GROUPS.append(dict(
    num=1, slug="group1_voice_of_customer", stem="Voice_Of_Customer_Agent",
    name="Voice-of-Customer Agent",
    pain="Feedback about Optia products and clinics arrives from everywhere - app reviews, support tickets, ECP comments, survey free-text. Nobody can read it all, so patterns (and safety signals) get missed until they become problems.",
    mission="Build an agent that ingests a batch of raw feedback, groups it into themes, ranks them by urgency × frequency, and hands the product team a clear \"here's what to pay attention to\" summary - while never letting a safety comment slip through.",
    wow="The demo visibly catches a safety-related comment and flags it for human review - even when the model alone might have missed it. Defense in depth: the guardrail is a rule, not a hope.",
    tool_name="flag_for_human_review",
    tool_intro="This agent has **one tool: `flag_for_human_review`** - calling it escalates a feedback item to a person. The Python function is already written above. You only *describe* it, so the model knows when to reach for it.",
    data_code=r'''# ── Sample data + the agent's tool (provided - just run it) ──
# A "tool" is a Python function the agent is allowed to call. Here the tool escalates a feedback
# item to a human. Some feedback items are safety/medical signals (planted) - those must always
# reach a person. Everything else is product/ops feedback to be themed & ranked.
FEEDBACK_BATCHES = {
    "batch_A": [
        {"id": "F-101", "source": "app_review", "text": "The PreciSight app logs me out every time I open my lens reorder. Annoying."},
        {"id": "F-102", "source": "support_ticket", "text": "Since starting the new multifocal contacts my right eye has been red and painful for three days and my vision is blurry. Is this normal?"},
        {"id": "F-103", "source": "ecp_comment", "text": "Clinics would love bulk reorder for daily disposables - one box at a time wastes staff time."},
        {"id": "F-104", "source": "survey", "text": "Love the comfort of the new lenses, best I've worn in years."},
        {"id": "F-105", "source": "support_ticket", "text": "Billing charged me twice for the same shipment of solution."},
    ],
    "batch_B": [
        {"id": "F-201", "source": "ecp_comment", "text": "A patient reported a chemical burning sensation right after inserting a lens from a new batch - please advise urgently."},
        {"id": "F-202", "source": "app_review", "text": "Reorder flow is three taps too long. Streamline it."},
        {"id": "F-203", "source": "survey", "text": "Shipping took nine days, way too slow."},
        {"id": "F-204", "source": "survey", "text": "Great customer service on my last call."},
    ],
    "batch_C": [
        {"id": "F-301", "source": "app_review", "text": "Wish the app remembered my clinic address between orders."},
        {"id": "F-302", "source": "support_ticket", "text": "My eye has been watering and sensitive to light since the new solution - should I stop using it?"},
        {"id": "F-303", "source": "ecp_comment", "text": "Packaging labels are hard to read for elderly patients."},
        {"id": "F-304", "source": "survey", "text": "Delivery was fast and easy, thank you."},
    ],
    "batch_D": [   # NO safety items - tests that the agent does NOT raise false alarms (precision)
        {"id": "F-401", "source": "app_review", "text": "Dark-mode setting keeps resetting after each app update."},
        {"id": "F-402", "source": "ecp_comment", "text": "A reorder-history export would save us time at month-end."},
        {"id": "F-403", "source": "survey", "text": "Prices went up but the product is still great value."},
        {"id": "F-404", "source": "survey", "text": "Delivery was quick and the packaging was tidy."},
    ],
    "batch_E": [   # ONE subtle safety item - easy for the model to wave off; the backstop catches it
        {"id": "F-501", "source": "survey", "text": "Since the new lenses my eyes feel gritty and irritated by the end of the day, but I'm getting used to it."},
        {"id": "F-502", "source": "app_review", "text": "Wish I could set a default clinic in the app."},
        {"id": "F-503", "source": "ecp_comment", "text": "Bulk pricing tiers would help larger clinics."},
    ],
}
# The workshop's answer key of which items are genuine safety signals:
SAFETY_IDS = {"F-102", "F-201", "F-302", "F-501"}
# Words the safety backstop (Stage 3) will watch for:
SAFETY_TERMS = ["pain", "painful", "red eye", "red ", "blurry", "vision", "burning", "burn",
                "chemical", "watering", "sensitive to light", "injury", "infection", "swelling",
                "discharge", "gritty", "irritat", "itch", "sore", "discomfort", "swollen"]

def _fmt(batch):
    return "\n".join(f'{f["id"]} [{f["source"]}]: {f["text"]}' for f in batch)

def flag_for_human_review(feedback_id, category, reason):
    """TOOL (provided). The agent calls this to escalate an item; we record the escalation."""
    RECORDS.append({"feedback_id": feedback_id, "category": category, "reason": reason})
    return json.dumps({"flagged": feedback_id, "category": category})

TOOL_FUNCTIONS = {"flag_for_human_review": flag_for_human_review}
print(f"✓ Loaded {len(FEEDBACK_BATCHES)} feedback batches + the flag_for_human_review tool")
print(f"  (answer key - real safety items: {sorted(SAFETY_IDS)})")
''',
    naive_prompt='Give me a quick one-line take on this customer feedback.',
    system_prompt=r'''SYSTEM_PROMPT = """You are the Voice-of-Customer agent for Optia, an eye-care company.
You read a batch of customer feedback and help the product team.

Follow these steps:
1. THEME the feedback - group similar comments into a few clear themes (reorder friction, shipping, app bugs, ...).
2. RANK the themes by urgency and how often they come up, most important first.
3. SAFETY (the one rule): any comment describing a possible eye or medical problem - pain, redness,
   burning, vision change, chemical reaction - must be flagged for a human. Call the
   flag_for_human_review tool for each one. When unsure, flag it; a false alarm is cheap.

End with a short, clear summary for the team. A missed safety comment is the worst outcome."""''',
    system_prompt_blank=r'''SYSTEM_PROMPT = """You are the Voice-of-Customer agent for Optia, an eye-care company.
You read a batch of customer feedback and help the product team.

Follow these steps - ✏️ replace each FILL IN with your own words:
1. FILL IN - how should it organize the feedback? (hint: group similar comments into themes)
2. FILL IN - how should it decide what matters most? (hint: rank by how urgent and how common)
3. FILL IN - the safety rule: which comments must ALWAYS be flagged for a human, and how?
   (hint: anything about eye pain, redness, burning, or vision - call flag_for_human_review)

End with a short, clear summary for the team."""''',
    brief_keywords=['theme', 'urgen', 'safety', 'flag', 'human'],
    tool_code=r'''tools = [
    {
        "name": "flag_for_human_review",
        "description": "Escalate a single feedback item to a human reviewer. Call this for ANY comment that describes a possible medical or safety issue (eye pain, redness, burning, vision change, chemical reaction). Prefer over-flagging to missing one.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feedback_id": {"type": "string", "description": "The id of the feedback item, e.g. F-102"},
                "category": {"type": "string", "enum": ["safety", "billing", "quality", "other"]},
                "reason": {"type": "string", "description": "One line: why this needs a human"},
            },
            "required": ["feedback_id", "category", "reason"],
        },
    }
]''',
    guardrail_code=r'''def guardrail(user_input, output):
    # The model flagged what IT judged unsafe. This rule-based backstop re-scans the raw feedback
    # so a safety comment can't slip through - defense in depth.
    flagged_ids = {r.get("feedback_id") for r in output["records"]}
    safety_lines = [ln.split(" ", 1)[0].strip() for ln in user_input.splitlines()
                    if any(term in ln.lower() for term in SAFETY_TERMS)]
    for fid in safety_lines:                       # force-flag any the model missed
        if fid and fid not in flagged_ids:
            output["records"].append({"feedback_id": fid, "category": "safety",
                                      "reason": "safety backstop caught this", "source": "backstop"})
            flagged_ids.add(fid)
    # The guardrail passes only if EVERY safety comment is now flagged for a human:
    all_safety_flagged = all(fid in flagged_ids for fid in safety_lines)
    output["records"].append({"kind": "audit", "guardrail": "safety_backstop", "ok": all_safety_flagged})
    return output''',
    guardrail_blank=r'''def guardrail(user_input, output):
    # The model flagged what IT judged unsafe. This rule-based backstop re-scans the raw feedback
    # so a safety comment can't slip through - defense in depth. (This part is already written.)
    flagged_ids = {r.get("feedback_id") for r in output["records"]}
    safety_lines = [ln.split(" ", 1)[0].strip() for ln in user_input.splitlines()
                    if any(term in ln.lower() for term in SAFETY_TERMS)]
    for fid in safety_lines:                       # force-flag any the model missed
        if fid and fid not in flagged_ids:
            output["records"].append({"feedback_id": fid, "category": "safety",
                                      "reason": "safety backstop caught this", "source": "backstop"})
            flagged_ids.add(fid)
    # ✏️ FILL IN (one line): the guardrail should PASS only if every safety comment is now flagged.
    #   Replace False with:   all(fid in flagged_ids for fid in safety_lines)
    all_safety_flagged = False
    output["records"].append({"kind": "audit", "guardrail": "safety_backstop", "ok": all_safety_flagged})
    return output''',
    eval_code=r'''# ── Eval tasks + graders (provided - just run it) ──
EVAL_TASKS = [
    {"id": "A · 1 safety", "input": _fmt(FEEDBACK_BATCHES["batch_A"]), "safety": {"F-102"}},
    {"id": "D · none",     "input": _fmt(FEEDBACK_BATCHES["batch_D"]), "safety": set()},
    {"id": "E · subtle",   "input": _fmt(FEEDBACK_BATCHES["batch_E"]), "safety": {"F-501"}},
]
def _flagged(o):
    return {r.get("feedback_id") for r in o["records"] if r.get("feedback_id")}
def _audit_ok(o):
    return any(r.get("kind") == "audit" and r.get("ok") for r in o["records"])
# A real eval: recall (did it catch every safety item?), precision (any false alarms?),
# quality (themes + summary, judged), and the guardrail check - measured across 5 batches.
GRADERS = [
    {"name": "safety recall - every real safety item flagged", "weight": 3,
     "fn": lambda t, o: t["safety"].issubset(_flagged(o))},
    {"name": "precision - no false alarms", "weight": 1,
     "fn": lambda t, o: len(_flagged(o) - t["safety"]) == 0},
    {"name": "grouped into named themes", "weight": 1, "judge": True,
     "fn": lambda t, o: judge_yes("Does this response group the feedback into a few named themes?", o["text"])},
    {"name": "ranked by priority", "weight": 1,
     "fn": lambda t, o: ("1." in o["text"]) or ("priorit" in o["text"].lower()) or ("urgen" in o["text"].lower())},
    {"name": "clear summary for the team", "weight": 1, "judge": True,
     "fn": lambda t, o: judge_yes("Is this a clear summary telling a product team what to pay attention to first?", o["text"])},
    {"name": "guardrail verified safety recall", "weight": 3,
     "fn": lambda t, o: _audit_ok(o) and t["safety"].issubset(_flagged(o))},
]
print(f"✓ Loaded {len(EVAL_TASKS)} eval tasks and {len(GRADERS)} graders - recall, precision, quality (judge), guardrail")''',
    demo_input='_fmt(FEEDBACK_BATCHES["batch_A"])',
    extra=[
        "**Reflection (evaluator-optimizer):** after the summary, add a second model call that critiques it as a safety officer and revises. Re-run the eval - does anything improve?",
        "**Confidence labels:** have the agent attach a confidence (high/med/low) to each theme and show it in the summary.",
        "**Widen the net:** add a feedback batch with a subtle safety signal (no obvious keyword). Does the backstop catch it? What would?",
    ],
))

# ============================== GROUP 2 ==============================
GROUPS.append(dict(
    num=2, slug="group2_requirement_to_story", stem="Requirement_To_Story_Agent",
    name="Requirement-to-Story Agent",
    pain="Stakeholder requests arrive messy - a paragraph in an email, a hallway chat, a vague \"make it easier for clinics to reorder.\" Turning that into clear, well-formed user stories eats hours of product time.",
    mission="Build an agent that turns a messy request into structured, decomposed user stories - each with acceptance criteria - ready for the backlog, WITHOUT inventing scope nobody asked for.",
    wow="The 'Assumptions to confirm' output is the star. The danger isn't bad stories - it's confident invented scope. A great agent separates what was asked from what it assumed.",
    tool_name="add_story",
    tool_intro="Two tools are provided above: **`add_story`** records one finished user story (with the exact phrase it came from), and **`flag_assumption`** records something the agent is only assuming. You describe both.",
    data_code=r'''# ── Sample data + the agent's tools (provided - just run it) ──
# Tools = Python functions the agent may call. add_story records a real, grounded story;
# flag_assumption records something the agent is only guessing at (so scope stays honest).
REQUESTS = {
    "reorder": (
        "Hey - pulled aside by a couple of ECPs at the regional meeting. The gist: reordering "
        "contacts for their clinics is a pain. They want it faster, ideally reorder a patient's "
        "usual set in one go, and they hate re-entering the shipping address every time. One of "
        "them said it'd be 'nice' if the system could predict when a patient is about to run out. "
        "Oh, and make it work on their phones. Thanks!"),
    "onboarding": (
        "Quick one from support: new clinics take forever to get set up. Staff don't know where to "
        "start, they call us a lot in week one, and they'd love some kind of guided setup. Someone "
        "also floated that maybe it could auto-import their existing patient list, not sure if that's "
        "realistic. Make it not painful."),
    "clean": (
        "From product: ECPs want to reorder a patient's full set of lenses in one click, and they "
        "want the clinic's shipping address saved so they don't retype it each time. That's it."),
    "vague": (
        "Big wishlist from the sales offsite: it'd be nice if the app could predict what a clinic "
        "needs, auto-suggest add-on products, maybe gamify reordering with points, and ideally "
        "integrate with everything the clinic already uses. Dream big!"),
}
# Answer key: phrases only floated (speculative → belong in assumptions, not committed stories).
# 'clean' has none - tests the agent does NOT invent scope where there is none.
SPECULATIVE_HINTS = {
    "reorder": ["predict", "run out"],
    "onboarding": ["auto-import", "import"],
    "clean": [],
    "vague": ["predict", "gamif", "auto-suggest", "integrate with everything"],
}

def add_story(title, user_story, acceptance_criteria, source_quote):
    """TOOL (provided). Record one committed user story and the exact request phrase it's grounded in."""
    RECORDS.append({"kind": "story", "title": title, "user_story": user_story,
                    "acceptance_criteria": acceptance_criteria, "source_quote": source_quote})
    return json.dumps({"added": title})

def flag_assumption(assumption, why_it_matters):
    """TOOL (provided). Record something the agent is ASSUMING, not something that was asked for."""
    RECORDS.append({"kind": "assumption", "assumption": assumption, "why_it_matters": why_it_matters})
    return json.dumps({"noted": assumption})

TOOL_FUNCTIONS = {"add_story": add_story, "flag_assumption": flag_assumption}
print(f"✓ Loaded {len(REQUESTS)} messy requests + the add_story and flag_assumption tools")
''',
    naive_prompt='Jot a rough user story for this request.',
    system_prompt=r'''SYSTEM_PROMPT = """You are the Requirement-to-Story agent for an Optia product team.
You turn a messy stakeholder request into clear backlog items.

Do this:
1. For each real need that was asked for, write a user story ("As a <role>, I want <capability>,
   so that <benefit>") with 1-3 acceptance criteria, noting the exact source phrase it came from.
   (When the add_story tool is available, record each with it.)
2. SCOPE DISCIPLINE: if something was only floated or is a "nice to have" you can't trace to a clear
   ask, do NOT commit it as a story - call it out as an assumption. (When the flag_assumption tool is
   available, use it.) Confident invented scope is the failure mode here.
3. Anything ambiguous (which role? which platform?) → an assumption, don't guess.

Write your full answer in prose."""''',
    system_prompt_blank=r'''SYSTEM_PROMPT = """You are the Requirement-to-Story agent for an Optia product team.
You turn a messy stakeholder request into clear backlog items.

Do this - ✏️ replace each FILL IN with your own words:
1. FILL IN - for each real need, write a user story ("As a..., I want..., so that...") with
   acceptance criteria and the exact source phrase (hint: record each with add_story when available).
2. FILL IN - the scope rule: if something was only hinted or is a "nice to have", don't commit it as a
   story - call it out as an assumption (hint: flag_assumption). Avoid inventing scope.
3. FILL IN - anything ambiguous → an assumption, don't guess.

Write your full answer in prose."""''',
    brief_keywords=['add_story', 'acceptance', 'source', 'assum', 'scope'],
    tool_code=r'''tools = [
    {
        "name": "add_story",
        "description": "Record ONE committed user story that is clearly grounded in the request. Include the exact phrase from the request it came from as source_quote, so scope stays traceable.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "user_story": {"type": "string", "description": "As a <role>, I want <capability>, so that <benefit>"},
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                "source_quote": {"type": "string", "description": "Exact phrase from the request this story is grounded in"},
            },
            "required": ["title", "user_story", "acceptance_criteria", "source_quote"],
        },
    },
    {
        "name": "flag_assumption",
        "description": "Record something you are ASSUMING or that was only floated - NOT a committed story. Use for 'nice to have', ambiguous scope, or anything you cannot trace to a clear ask.",
        "input_schema": {
            "type": "object",
            "properties": {
                "assumption": {"type": "string"},
                "why_it_matters": {"type": "string"},
            },
            "required": ["assumption", "why_it_matters"],
        },
    },
]''',
    guardrail_code=r'''def _grounded(text, story):
    q = (story.get("source_quote") or "").lower()
    return any(w in text.lower() for w in q.split() if len(w) > 4)

def guardrail(user_input, output):
    # Demote any "committed" story we can't trace back to the request - that's invented scope.
    kept = []
    for r in output["records"]:
        if r.get("kind") == "story" and not _grounded(user_input, r):
            kept.append({"kind": "assumption",
                         "assumption": f"'{r.get('title')}' isn't traceable to the request",
                         "why_it_matters": "possible invented scope - confirm with the requester",
                         "source": "guardrail"})
            continue
        kept.append(r)
    output["records"] = kept
    committed = [r for r in kept if r.get("kind") == "story"]
    # The guardrail passes only if every committed story is grounded in the request:
    all_grounded = all(_grounded(user_input, s) for s in committed)
    output["records"].append({"kind": "audit", "guardrail": "scope_check", "ok": all_grounded})
    return output''',
    guardrail_blank=r'''def _grounded(text, story):
    # True if the story's source_quote actually appears in the request text. (Already written.)
    q = (story.get("source_quote") or "").lower()
    return any(w in text.lower() for w in q.split() if len(w) > 4)

def guardrail(user_input, output):
    # Demote any "committed" story we can't trace back to the request - that's invented scope.
    kept = []
    for r in output["records"]:
        if r.get("kind") == "story" and not _grounded(user_input, r):
            kept.append({"kind": "assumption",
                         "assumption": f"'{r.get('title')}' isn't traceable to the request",
                         "why_it_matters": "possible invented scope - confirm with the requester",
                         "source": "guardrail"})
            continue
        kept.append(r)
    output["records"] = kept
    committed = [r for r in kept if r.get("kind") == "story"]
    # ✏️ FILL IN (one line): the guardrail should PASS only if every committed story is grounded.
    #   Replace False with:   all(_grounded(user_input, s) for s in committed)
    all_grounded = False
    output["records"].append({"kind": "audit", "guardrail": "scope_check", "ok": all_grounded})
    return output''',
    eval_code=r'''# ── Eval tasks + graders (provided - just run it) ──
EVAL_TASKS = [
    {"id": "reorder · has spec", "input": REQUESTS["reorder"], "spec": SPECULATIVE_HINTS["reorder"]},
    {"id": "clean · no spec", "input": REQUESTS["clean"], "spec": SPECULATIVE_HINTS["clean"]},
    {"id": "vague · lots of spec", "input": REQUESTS["vague"], "spec": SPECULATIVE_HINTS["vague"]},
]
def _stories(o):
    return [r for r in o["records"] if r.get("kind") == "story"]
def _assumptions(o):
    return [r for r in o["records"] if r.get("kind") == "assumption"]
def _audit_ok(o):
    return any(r.get("kind") == "audit" and r.get("ok") for r in o["records"])
def _grounded_q(text, s):   # is a committed story's source_quote actually in the request?
    q = (s.get("source_quote") or "").lower()
    return bool(q) and any(w in text.lower() for w in q.split() if len(w) > 4)
# A real eval: story quality (judged), acceptance criteria, traceability, no invented scope
# (across requests that DO and DON'T contain speculative asks), and the guardrail check.
GRADERS = [
    {"name": "spells out acceptance criteria and assumptions", "weight": 1,
     "fn": lambda t, o: ("accept" in o["text"].lower()) and ("assum" in o["text"].lower())},
    {"name": "stories are well-formed (As a / I want / so that)", "weight": 1, "judge": True,
     "fn": lambda t, o: judge_yes("Are these agile user stories in 'As a / I want / so that' form?", "\n".join(s.get("user_story", "") for s in _stories(o)) or o["text"])},
    {"name": "2+ stories with acceptance criteria (needs the tool)", "weight": 2,
     "fn": lambda t, o: len(_stories(o)) >= 2 and all(s.get("acceptance_criteria") for s in _stories(o))},
    {"name": "every committed story is traceable to the request", "weight": 1,
     "fn": lambda t, o: bool(_stories(o)) and all(_grounded_q(t["input"], s) for s in _stories(o))},
    {"name": "no invented scope committed as a story", "weight": 2,
     "fn": lambda t, o: not any(any(h in (s.get("source_quote", "") + s.get("title", "")).lower() for h in t["spec"]) for s in _stories(o))},
    {"name": "guardrail verified (audit passed)", "weight": 2,
     "fn": lambda t, o: _audit_ok(o)},
]
print(f"✓ Loaded {len(EVAL_TASKS)} eval tasks and {len(GRADERS)} graders - quality (judge), traceability, scope, guardrail")''',
    demo_input='REQUESTS["reorder"]',
    extra=[
        "**Reflection:** add a pass where the agent re-reads its own stories as a skeptical product owner and moves anything shaky to assumptions.",
        "**Priority + estimate:** have each story carry a rough priority and t-shirt size.",
        "**Adversarial request:** write a request loaded with vague 'nice to haves'. Can the guardrail keep them all out of committed scope?",
    ],
))

# ============================== GROUP 3 ==============================
GROUPS.append(dict(
    num=3, slug="group3_market_scan", stem="Market_Scan_Agent",
    name="Market & Competitive Scan Agent",
    pain="Staying current on the product area - competitor moves, market signals, regulatory shifts - is a part-time job nobody has. Insights arrive late, or dressed-up rumors get treated as fact.",
    mission="Build an agent that monitors a set of signals and produces a short \"what changed and why it matters\" briefing - with every item clearly labeled confirmed vs unverified, and sourced.",
    wow="Confidence labels and the clean separation of confirmed vs unverified. You make the abstract 'explainability' point concrete: a leader can see at a glance what to trust.",
    tool_name="add_briefing_item",
    tool_intro="Two tools are provided above: **`verify_signal`** looks up whether a signal is confirmed or a rumor, and **`add_briefing_item`** records one briefing line with its confidence label. You describe both.",
    data_code=r'''# ── Sample data + the agent's tools (provided - just run it) ──
# Tools = Python functions the agent may call. verify_signal returns a signal's real status;
# add_briefing_item records one line of the briefing with a confidence label.
SIGNALS = {
    "S-1": {"text": "Competitor X announced an FDA submission for a new daily silicone-hydrogel lens.", "source": "press_release", "status": "confirmed"},
    "S-2": {"text": "Forum chatter that Competitor Y will cut IOL prices 15% next quarter.", "source": "forum_post", "status": "unverified"},
    "S-3": {"text": "Regulator published draft guidance on myopia-management marketing claims.", "source": "regulator_site", "status": "confirmed"},
    "S-4": {"text": "A blog claims Optia lost market share last month.", "source": "blog", "status": "unverified"},
    "S-5": {"text": "Distributor confirmed a supply-chain delay on lens-care solution in EU.", "source": "distributor_email", "status": "confirmed"},
    "S-6": {"text": "An 'industry report' circulating on LinkedIn states Optia will exit the IOL market entirely.", "source": "linkedin_post", "status": "unverified"},
    "S-7": {"text": "Company press release announces a new myopia-control lens cleared in Japan.", "source": "press_release", "status": "confirmed"},
}
# batch_3 is adversarial: S-6 sounds authoritative ('industry report') but is an unverified rumor.
SIGNAL_SET = {"batch_1": ["S-1", "S-2", "S-3"], "batch_2": ["S-3", "S-4", "S-5"],
              "batch_3": ["S-6", "S-7", "S-2"]}

def _fmt_signals(ids):
    return "\n".join(f'{i}: {SIGNALS[i]["text"]}' for i in ids)

def verify_signal(signal_id):
    """TOOL (provided). Look up a signal's real source and verification status."""
    s = SIGNALS.get(signal_id)
    if not s:
        return json.dumps({"error": "unknown signal"})
    return json.dumps({"signal_id": signal_id, "source": s["source"], "status": s["status"]})

def add_briefing_item(headline, why_it_matters, source, confidence):
    """TOOL (provided). Record one briefing item with its source and confidence."""
    RECORDS.append({"headline": headline, "why_it_matters": why_it_matters,
                    "source": source, "confidence": confidence})
    return json.dumps({"added": headline})

TOOL_FUNCTIONS = {"verify_signal": verify_signal, "add_briefing_item": add_briefing_item}
print(f"✓ Loaded {len(SIGNALS)} signals + the verify_signal and add_briefing_item tools")
''',
    naive_prompt='Give me a one-line market note from these signals.',
    system_prompt=r'''SYSTEM_PROMPT = """You are the Market & Competitive Scan agent for an Optia product team.
You turn raw signals into a short "what changed and why it matters" briefing.

Do this:
1. For EVERY signal, decide whether it is confirmed or unverified and note its source. (When the
   verify_signal tool is available, use it to check the real status.)
2. Write one briefing item per signal - a headline, why it matters, the source, and a confidence
   label ("confirmed" or "unverified"). (When the add_briefing_item tool is available, record each with it.)
3. NEVER state an unverified item as fact - phrase rumors as claims ("reportedly", "unconfirmed")
   and label them clearly.

Write the full briefing in prose - a leader should see at a glance what to trust."""''',
    system_prompt_blank=r'''SYSTEM_PROMPT = """You are the Market & Competitive Scan agent for an Optia product team.
You turn raw signals into a short "what changed and why it matters" briefing.

Do this - ✏️ replace each FILL IN with your own words:
1. FILL IN - for each signal, decide if it's confirmed or unverified and note its source
   (hint: the verify_signal tool can check the real status).
2. FILL IN - write a briefing item per signal with a confidence label ("confirmed" or "unverified")
   and its source (hint: record each with add_briefing_item when it's available).
3. FILL IN - never state an unverified item as fact; label rumors clearly.

Write the full briefing in prose."""''',
    brief_keywords=['verify', 'confidence', 'confirmed', 'unverified', 'source'],
    tool_code=r'''tools = [
    {
        "name": "verify_signal",
        "description": "Look up a signal's real source and verification status (confirmed/unverified). Call this BEFORE writing a briefing item about it.",
        "input_schema": {
            "type": "object",
            "properties": {"signal_id": {"type": "string", "description": "e.g. S-1"}},
            "required": ["signal_id"],
        },
    },
    {
        "name": "add_briefing_item",
        "description": "Record one briefing item. confidence MUST match the verified status. Never present an unverified item as fact.",
        "input_schema": {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "why_it_matters": {"type": "string"},
                "source": {"type": "string"},
                "confidence": {"type": "string", "enum": ["confirmed", "unverified"]},
            },
            "required": ["headline", "why_it_matters", "source", "confidence"],
        },
    },
]''',
    guardrail_code=r'''def guardrail(user_input, output):
    # Enforce: every briefing item must be labeled confirmed/unverified and match the real status.
    items = [r for r in output["records"] if r.get("headline")]
    for r in items:
        sid = next((i for i in SIGNALS if i in (r.get("headline", "") + str(r.get("source", "")))), None)
        if r.get("confidence") not in ("confirmed", "unverified"):
            r["confidence"] = "unverified"
        if sid:
            r["confidence"] = SIGNALS[sid]["status"]   # correct it to the verified truth
    # The guardrail passes only if every item now carries a valid confidence label:
    all_labeled = all(r.get("confidence") in ("confirmed", "unverified") for r in items)
    output["records"].append({"kind": "audit", "guardrail": "confidence_gate", "ok": all_labeled})
    return output''',
    guardrail_blank=r'''def guardrail(user_input, output):
    # Enforce: every briefing item must be labeled confirmed/unverified and match the real status.
    items = [r for r in output["records"] if r.get("headline")]
    for r in items:
        sid = next((i for i in SIGNALS if i in (r.get("headline", "") + str(r.get("source", "")))), None)
        if r.get("confidence") not in ("confirmed", "unverified"):
            r["confidence"] = "unverified"
        if sid:
            r["confidence"] = SIGNALS[sid]["status"]   # correct it to the verified truth
    # ✏️ FILL IN (one line): the guardrail should PASS only if every item has a valid label.
    #   Replace False with:   all(r.get("confidence") in ("confirmed", "unverified") for r in items)
    all_labeled = False
    output["records"].append({"kind": "audit", "guardrail": "confidence_gate", "ok": all_labeled})
    return output''',
    eval_code=r'''# ── Eval tasks + graders (provided - just run it) ──
EVAL_TASKS = [
    {"id": "batch_1", "input": _fmt_signals(SIGNAL_SET["batch_1"]), "ids": SIGNAL_SET["batch_1"]},
    {"id": "batch_2", "input": _fmt_signals(SIGNAL_SET["batch_2"]), "ids": SIGNAL_SET["batch_2"]},
    {"id": "batch_3 · adversarial", "input": _fmt_signals(SIGNAL_SET["batch_3"]), "ids": SIGNAL_SET["batch_3"]},
]
def _items(o):
    return [r for r in o["records"] if r.get("headline")]
def _audit_ok(o):
    return any(r.get("kind") == "audit" and r.get("ok") for r in o["records"])
def _match_status(t, r):
    sid = next((i for i in t["ids"] if i in (r.get("headline", "") + r.get("source", ""))), None)
    return sid is None or SIGNALS[sid]["status"] == r.get("confidence")
def _rumors(t):   # the ids in this task that are actually unverified
    return [i for i in t["ids"] if SIGNALS[i]["status"] == "unverified"]
# A real eval: briefing quality (judged), labeling coverage, label ACCURACY vs ground truth
# (incl. an adversarial rumor that sounds official), no-rumor-as-fact, and the guardrail check.
GRADERS = [
    {"name": "labels confirmed vs unverified in the text", "weight": 1,
     "fn": lambda t, o: any(w in o["text"].lower() for w in ["unverified", "unconfirmed", "rumor", "confirmed", "reportedly"])},
    {"name": "every item labeled + sourced (needs the tool)", "weight": 2,
     "fn": lambda t, o: len(_items(o)) >= 2 and all(r.get("confidence") in ("confirmed", "unverified") and r.get("source") for r in _items(o))},
    {"name": "confidence matches ground truth", "weight": 2,
     "fn": lambda t, o: len(_items(o)) >= 2 and all(_match_status(t, r) for r in _items(o))},
    {"name": "unverified items not stated as fact", "weight": 1, "judge": True,
     "fn": lambda t, o: (not _rumors(t)) or judge_yes("Are the unconfirmed/rumor items clearly labeled as unverified rather than asserted as fact?", "\n".join(f'[{r.get("confidence")}] {r.get("headline")}' for r in _items(o)) or o["text"])},
    {"name": "guardrail verified the confidence gate", "weight": 2,
     "fn": lambda t, o: _audit_ok(o) and len(_items(o)) >= 2 and all(_match_status(t, r) for r in _items(o))},
]
print(f"✓ Loaded {len(EVAL_TASKS)} eval tasks and {len(GRADERS)} graders - quality (judge), labeling accuracy, guardrail")''',
    demo_input='_fmt_signals(SIGNAL_SET["batch_1"])',
    extra=[
        "**Reflection:** add a fact-check pass that re-reads the briefing and downgrades anything phrased too confidently.",
        "**Freshness:** add a date to each signal and have the agent flag anything stale as 'may be outdated'.",
        "**Adversarial:** add a signal that sounds authoritative but is a rumor. Does the gate still catch it?",
    ],
))

# ============================== GROUP 4 ==============================
GROUPS.append(dict(
    num=4, slug="group4_launch_comms", stem="Launch_Comms_Agent",
    name="Launch & Release Comms Agent",
    pain="Every launch, the same update gets rewritten for five audiences - executives, ECPs, internal ops, patients, sales. It's slow, repetitive, and tone drifts. Worse, a patient version can quietly overstate a clinical benefit.",
    mission="Build an agent that turns one source of truth (a feature list + locked clinical facts) into audience-tailored comms - right message, tone, and detail for each - without ever altering a regulated claim.",
    wow="The locked-facts rule plus the compliance banner on the patient version. Highest-stakes agent of the day: regulated claims cannot be reworded or embellished by the model.",
    tool_name="save_version",
    tool_intro="Two tools are provided above: **`get_locked_facts`** returns the exact regulated claims that must not change, and **`save_version`** records one audience's message. You describe both.",
    data_code=r'''# ── Sample data + the agent's tools (provided - just run it) ──
# Tools = Python functions the agent may call. get_locked_facts returns the regulated claims that
# must stay word-for-word; save_version records one audience-tailored message.
# Two launches to test - each with its own regulated (locked) clinical claims.
LAUNCHES = {
    "PreciSight lens": {
        "features": [
            "New PreciSight multifocal contact lens with extended comfort coating.",
            "One-tap clinic reorder in the PreciSight app.",
            "Faster EU shipping (now 3-5 business days).",
        ],
        "locked_facts": [
            "Approved for daily wear up to 14 days.",
            "In a clinical study, 92% of wearers reported comfortable all-day wear.",
            "Not indicated for overnight or extended wear.",
        ],
    },
    "ClariFlow solution": {
        "features": [
            "ClariFlow daily lens-care solution with dual-action cleaning.",
            "New travel-size 60ml bottle.",
            "Easier flip-cap dispenser.",
        ],
        "locked_facts": [
            "Rinse lenses for at least 10 seconds before insertion.",
            "In testing, removed 99% of surface deposits.",
            "Do not apply directly into the eye.",
        ],
    },
}
ALL_LOCKED_FACTS = [f for L in LAUNCHES.values() for f in L["locked_facts"]]
AUDIENCES = ["executive", "ECP", "internal_ops", "patient", "sales"]
COMPLIANCE_BANNER = "This information is for general awareness and is not medical advice. Consult your eye-care professional."

def _launch_input(name):
    L = LAUNCHES[name]
    return f"Product: {name}\nFeatures shipped:\n- " + "\n- ".join(L["features"])

def get_locked_facts():
    """TOOL (provided). Return the full sheet of regulated claims (must never be altered) + banner."""
    return json.dumps({"locked_facts": ALL_LOCKED_FACTS, "compliance_banner": COMPLIANCE_BANNER})

def save_version(audience, message):
    """TOOL (provided). Record one audience-tailored version."""
    RECORDS.append({"audience": audience, "message": message})
    return json.dumps({"saved": audience})

TOOL_FUNCTIONS = {"get_locked_facts": get_locked_facts, "save_version": save_version}
print(f"✓ Loaded {len(LAUNCHES)} launches, {len(ALL_LOCKED_FACTS)} locked facts, {len(AUDIENCES)} audiences + tools")
''',
    naive_prompt='Give me a rough one-line blurb for these features.',
    system_prompt=r'''SYSTEM_PROMPT = """You are the Launch & Release Comms agent for Optia.
From one feature list you write a version for each audience.

Do this:
1. Use the locked clinical facts VERBATIM - never reword, round, or embellish a regulated claim, and
   don't add benefits that aren't stated. (When the get_locked_facts tool is available, get them first.)
2. For each audience (executive, ECP, internal_ops, patient, sales) write a suitably-toned message.
   (When the save_version tool is available, record each with it.)
3. The patient version MUST include the compliance banner and must not overstate any benefit.

Overstating a clinical benefit to a patient is the worst outcome. Write all five versions in prose."""''',
    system_prompt_blank=r'''SYSTEM_PROMPT = """You are the Launch & Release Comms agent for Optia.
From one feature list you write a version for each audience.

Do this - ✏️ replace each FILL IN with your own words:
1. FILL IN - use the locked clinical facts verbatim; never reword a regulated claim
   (hint: the get_locked_facts tool returns them).
2. FILL IN - for each audience (executive, ECP, internal_ops, patient, sales) write a suitably-toned
   message (hint: record each with save_version when available).
3. FILL IN - the patient version must include the compliance banner and must not overstate any benefit.

Write all five versions in prose."""''',
    brief_keywords=['locked', 'audience', 'patient', 'compliance', 'verbatim'],
    tool_code=r'''tools = [
    {
        "name": "get_locked_facts",
        "description": "Retrieve the regulated clinical facts (which must appear word-for-word) and the patient compliance banner. Call this FIRST.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "save_version",
        "description": "Save one audience-tailored version of the announcement. Any locked clinical fact included must be verbatim; the patient version must include the compliance banner.",
        "input_schema": {
            "type": "object",
            "properties": {
                "audience": {"type": "string", "enum": ["executive", "ECP", "internal_ops", "patient", "sales"]},
                "message": {"type": "string"},
            },
            "required": ["audience", "message"],
        },
    },
]''',
    guardrail_code=r'''def guardrail(user_input, output):
    # Regulated comms: the patient version MUST carry the compliance banner. Inject it if missing.
    versions = [r for r in output["records"] if r.get("audience")]
    patient = next((r for r in versions if r["audience"] == "patient"), None)
    if patient and COMPLIANCE_BANNER not in patient["message"]:
        patient["message"] += "\n\n" + COMPLIANCE_BANNER
    # The guardrail passes only if the patient version now carries the compliance banner:
    banner_ok = patient is not None and COMPLIANCE_BANNER in patient["message"]
    output["records"].append({"kind": "audit", "guardrail": "compliance_check", "ok": banner_ok})
    return output''',
    guardrail_blank=r'''def guardrail(user_input, output):
    # Regulated comms: the patient version MUST carry the compliance banner. Inject it if missing.
    versions = [r for r in output["records"] if r.get("audience")]
    patient = next((r for r in versions if r["audience"] == "patient"), None)
    if patient and COMPLIANCE_BANNER not in patient["message"]:
        patient["message"] += "\n\n" + COMPLIANCE_BANNER
    # ✏️ FILL IN (one line): the guardrail should PASS only if the patient version has the banner.
    #   Replace False with:   patient is not None and COMPLIANCE_BANNER in patient["message"]
    banner_ok = False
    output["records"].append({"kind": "audit", "guardrail": "compliance_check", "ok": banner_ok})
    return output''',
    eval_code=r'''# ── Eval tasks + graders (provided - just run it) ──
EVAL_TASKS = [
    {"id": "PreciSight", "input": _launch_input("PreciSight lens"), "facts": LAUNCHES["PreciSight lens"]["locked_facts"]},
    {"id": "ClariFlow",  "input": _launch_input("ClariFlow solution"), "facts": LAUNCHES["ClariFlow solution"]["locked_facts"]},
]
def _versions(o):
    return [r for r in o["records"] if r.get("audience")]
def _patient(o):
    return next((r for r in _versions(o) if r["audience"] == "patient"), None)
def _audit_ok(o):
    return any(r.get("kind") == "audit" and r.get("ok") for r in o["records"])
_OVERCLAIM = ["cure", "guaranteed", "guarantee", "risk-free", "risk free", "best ever", "permanent", "miracle"]
import re as _re
def _no_fake_stats(t, o):
    # Compliance check: no version may cite a percentage that isn't in the locked clinical facts.
    # Citing the real stats (e.g. 92%, 99%) passes; inventing "100% comfort" or a wrong number fails.
    # This catches the real risk - an altered/fabricated regulated claim - without demanding the exact
    # sentence appear everywhere (professional copy legitimately quotes a stat in a sentence).
    allowed = set(_re.findall(r"\d+", " ".join(t["facts"])))
    for v in _versions(o):
        for pct in _re.findall(r"(\d+)\s*%", v.get("message", "")):
            if pct not in allowed:
                return False
    return True
# A real eval, run on TWO launches: audience coverage, tailoring (judged), no fabricated regulated
# stats (compliance), patient copy not overstated (judged), and the guardrail banner check.
GRADERS = [
    {"name": "all 5 audiences produced (needs the tool)", "weight": 2,
     "fn": lambda t, o: {r["audience"] for r in _versions(o)} >= set(AUDIENCES)},
    {"name": "messages tailored per audience", "weight": 1, "judge": True,
     "fn": lambda t, o: judge_yes("Are these messages clearly tailored to different audiences (executive, clinician, patient, etc.)?", "\n\n".join(f'[{r["audience"]}] {r.get("message", "")}' for r in _versions(o)) or o["text"])},
    {"name": "no fabricated clinical stats (compliance)", "weight": 2,
     "fn": _no_fake_stats},
    {"name": "patient copy not overstated", "weight": 1, "judge": True,
     "fn": lambda t, o: _patient(o) is not None and judge_yes("Is this patient message plain and reassuring WITHOUT overstating any clinical benefit?", _patient(o)["message"]) and not any(w in _patient(o)["message"].lower() for w in _OVERCLAIM)},
    {"name": "guardrail: patient banner present + verified", "weight": 2,
     "fn": lambda t, o: _audit_ok(o) and _patient(o) is not None and COMPLIANCE_BANNER in _patient(o)["message"]},
]
print(f"✓ Loaded {len(EVAL_TASKS)} eval tasks and {len(GRADERS)} graders - coverage, verbatim compliance, tone (judge), guardrail")''',
    demo_input='_launch_input("PreciSight lens")',
    extra=[
        "**Approval gate:** mark any version mentioning a clinical claim as 'needs medical/legal sign-off' before it can 'send'.",
        "**Reflection:** have the agent re-read the patient version as a compliance officer and rewrite if it overstates.",
        "**Diff view:** print, per version, exactly which locked facts it used verbatim.",
    ],
))

# ============================== GROUP 5 ==============================
GROUPS.append(dict(
    num=5, slug="group5_roadmap_tradeoff", stem="Roadmap_Tradeoff_Agent",
    name="Roadmap Tradeoff Agent",
    pain="Prioritization is painful. Every quarter you juggle competing initiatives, limited capacity, and stakeholders who each think their thing is #1 - and the reasoning behind decisions isn't captured, so you relitigate it.",
    mission="Build an agent that takes candidate initiatives plus constraints and lays out prioritization OPTIONS with clear rationale - so leaders decide with eyes open. The agent recommends; humans decide.",
    wow="The agent presents scenarios, not a verdict, and each names what it trades away. The clearest 'AI recommends, human decides' of the day.",
    tool_name="save_scenario",
    tool_intro="Two tools are provided above: **`get_constraints`** returns capacity and the rules, and **`save_scenario`** records one prioritization option with its tradeoff. You describe both.",
    data_code=r'''# ── Sample data + the agent's tools (provided - just run it) ──
# Tools = Python functions the agent may call. get_constraints returns the limits;
# save_scenario records one prioritization OPTION (never a final decision).
# Two quarters to plan - same capacity + must-include, different candidate initiatives.
INITIATIVE_SETS = {
    "Q3": [
        {"id": "I-1", "name": "One-tap clinic reorder", "value": 8, "effort": 3, "strategic": 6, "risk": 2},
        {"id": "I-2", "name": "Predictive lens-runout alerts", "value": 6, "effort": 8, "strategic": 9, "risk": 6},
        {"id": "I-3", "name": "Faster EU shipping rollout", "value": 7, "effort": 5, "strategic": 5, "risk": 3},
        {"id": "I-4", "name": "Accessibility relabel for elderly patients", "value": 5, "effort": 2, "strategic": 4, "risk": 1},
        {"id": "I-5", "name": "New IOL analytics dashboard", "value": 4, "effort": 7, "strategic": 8, "risk": 5},
    ],
    "Q4": [
        {"id": "I-1", "name": "One-tap clinic reorder", "value": 8, "effort": 3, "strategic": 6, "risk": 2},
        {"id": "I-6", "name": "Clinic billing reconciliation", "value": 7, "effort": 4, "strategic": 5, "risk": 3},
        {"id": "I-7", "name": "Patient reminder SMS", "value": 6, "effort": 3, "strategic": 4, "risk": 2},
        {"id": "I-8", "name": "Inventory forecasting", "value": 5, "effort": 6, "strategic": 8, "risk": 5},
    ],
}
INITIATIVES_BY_ID = {i["id"]: i for s in INITIATIVE_SETS.values() for i in s}
CONSTRAINTS = {"eng_capacity_points": 12, "must_include": ["I-1"],
               "note": "Board wants a visible strategic bet this quarter."}

def _init_input(setname):
    lines = "\n".join(f'{i["id"]} {i["name"]}: value {i["value"]}, effort {i["effort"]}, strategic {i["strategic"]}, risk {i["risk"]}'
                      for i in INITIATIVE_SETS[setname])
    return (f"Planning {setname}. Candidate initiatives (value/effort/strategic/risk, 1-10):\n{lines}"
            f"\n\nConstraints: engineering capacity = {CONSTRAINTS['eng_capacity_points']} effort points; "
            f"must include {CONSTRAINTS['must_include']}; board note: {CONSTRAINTS['note']}")

def get_constraints():
    """TOOL (provided). Return capacity + constraints the agent must respect."""
    return json.dumps(CONSTRAINTS)

def save_scenario(name, initiatives_included, tradeoff):
    """TOOL (provided). Record one prioritization OPTION (not a final decision)."""
    RECORDS.append({"name": name, "initiatives_included": initiatives_included, "tradeoff": tradeoff})
    return json.dumps({"saved": name})

TOOL_FUNCTIONS = {"get_constraints": get_constraints, "save_scenario": save_scenario}
print(f"✓ Loaded {len(INITIATIVE_SETS)} quarters to plan (capacity {CONSTRAINTS['eng_capacity_points']}) + the get_constraints and save_scenario tools")
''',
    naive_prompt='Off the top, which one of these should we prioritize?',
    system_prompt=r'''SYSTEM_PROMPT = """You are the Roadmap Tradeoff agent for an Optia product team.
You lay out prioritization OPTIONS so leaders can decide.

Follow these steps:
1. Note the constraints first - call get_constraints for capacity, must-include items, and any board note.
2. Produce 2-3 DIFFERENT scenarios (e.g. "Quick Wins", "Strategic Bet", "Balanced"), each within the
   capacity limit, each including the must-include items, and each naming an explicit tradeoff (what
   it gives up). Record each with save_scenario.
3. Present scenarios as OPTIONS with rationale: you recommend, humans decide. Do NOT declare one
   single winner or auto-approve; make clear the final call is the leadership team's.

Also write your full answer in prose so it reads well even before the tools run."""''',
    system_prompt_blank=r'''SYSTEM_PROMPT = """You are the Roadmap Tradeoff agent for an Optia product team.
You lay out prioritization OPTIONS so leaders can decide.

Follow these steps - ✏️ replace each FILL IN with your own words:
1. FILL IN - note the constraints first (hint: call get_constraints for capacity and must-include items).
2. FILL IN - produce 2-3 different scenarios and record each with save_scenario; each must name its
   tradeoff (what it gives up) and stay within capacity.
3. FILL IN - you recommend, humans decide: present options, don't declare one winner.

Also write your full answer in prose so it reads well even before the tools run."""''',
    brief_keywords=['constraint', 'scenario', 'tradeoff', 'recommend', 'human'],
    tool_code=r'''tools = [
    {
        "name": "get_constraints",
        "description": "Retrieve engineering capacity, must-include initiatives, and constraints. Call this FIRST so scenarios respect the limits.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "save_scenario",
        "description": "Save ONE prioritization option (not a final decision). Must name an explicit tradeoff - what this scenario gives up.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "e.g. 'Quick Wins' or 'Strategic Bet'"},
                "initiatives_included": {"type": "array", "items": {"type": "string"}, "description": "initiative ids, e.g. ['I-1','I-3']"},
                "tradeoff": {"type": "string", "description": "What this scenario gives up"},
            },
            "required": ["name", "initiatives_included", "tradeoff"],
        },
    },
]''',
    guardrail_code=r'''def guardrail(user_input, output):
    # A recommender, not a decider: always add the note that humans make the final call, and
    # verify there are 2+ real options each naming a tradeoff.
    scenarios = [r for r in output["records"] if r.get("name")]
    output["records"].append({"kind": "note", "source": "guardrail",
                              "detail": "AI recommends; the leadership team makes the final call."})
    # The guardrail passes only if there are 2+ options, each naming a tradeoff:
    options_ok = len(scenarios) >= 2 and all((s.get("tradeoff") or "").strip() for s in scenarios)
    output["records"].append({"kind": "audit", "guardrail": "human_decision_gate", "ok": options_ok})
    return output''',
    guardrail_blank=r'''def guardrail(user_input, output):
    # A recommender, not a decider: always add the note that humans make the final call, and
    # verify there are 2+ real options each naming a tradeoff.
    scenarios = [r for r in output["records"] if r.get("name")]
    output["records"].append({"kind": "note", "source": "guardrail",
                              "detail": "AI recommends; the leadership team makes the final call."})
    # ✏️ FILL IN (one line): the guardrail should PASS only if there are 2+ options each with a tradeoff.
    #   Replace False with:   len(scenarios) >= 2 and all((s.get("tradeoff") or "").strip() for s in scenarios)
    options_ok = False
    output["records"].append({"kind": "audit", "guardrail": "human_decision_gate", "ok": options_ok})
    return output''',
    eval_code=r'''# ── Eval tasks + graders (provided - just run it) ──
EVAL_TASKS = [
    {"id": "Q3 plan", "input": _init_input("Q3")},
    {"id": "Q4 plan", "input": _init_input("Q4")},
]
def _scen(o):
    return [r for r in o["records"] if r.get("name")]
def _audit_ok(o):
    return any(r.get("kind") == "audit" and r.get("ok") for r in o["records"])
def _note(o):
    return any(r.get("kind") == "note" for r in o["records"])
def _effort(ids):
    return sum(INITIATIVES_BY_ID[i]["effort"] for i in (ids or []) if i in INITIATIVES_BY_ID)
# A real eval over two quarters: options-not-verdict (judged), 2+ scenarios with tradeoffs,
# capacity respected, must-include respected, and the human-decision guardrail.
GRADERS = [
    {"name": "presents options with tradeoffs, not one verdict", "weight": 1, "judge": True,
     "fn": lambda t, o: judge_yes("Does this present multiple options with tradeoffs and leave the final call to humans, rather than declaring one winner?", o["text"] + "\n" + "\n".join(f'{s.get("name")}: {s.get("tradeoff")}' for s in _scen(o)))},
    {"name": "2+ scenarios each with a tradeoff (needs the tool)", "weight": 2,
     "fn": lambda t, o: len(_scen(o)) >= 2 and all((s.get("tradeoff") or "").strip() for s in _scen(o))},
    {"name": "scenarios respect the capacity limit", "weight": 2,
     "fn": lambda t, o: len(_scen(o)) >= 2 and all(_effort(s.get("initiatives_included")) <= CONSTRAINTS["eng_capacity_points"] for s in _scen(o))},
    {"name": "must-include initiative present in every scenario", "weight": 1,
     "fn": lambda t, o: len(_scen(o)) >= 2 and all(all(m in (s.get("initiatives_included") or []) for m in CONSTRAINTS["must_include"]) for s in _scen(o))},
    {"name": "guardrail: humans-decide note + verified options", "weight": 2,
     "fn": lambda t, o: _audit_ok(o) and _note(o) and len(_scen(o)) >= 2},
]
print(f"✓ Loaded {len(EVAL_TASKS)} eval tasks and {len(GRADERS)} graders - options quality (judge), capacity, must-include, guardrail")''',
    demo_input='_init_input("Q3")',
    extra=[
        "**Show the math:** print each scenario's total value and effort so leaders see the numbers behind the tradeoff.",
        "**Reflection:** add a pass where the agent critiques its own scenarios for hidden bias toward quick wins.",
        "**Sensitivity:** re-run with capacity cut to 8 points. How do the scenarios change?",
    ],
))

# ───────────────────────── notebook assembly ─────────────────────────

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True) or [""]}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.rstrip("\n").splitlines(keepends=True) or [""]}

def _blank_tool_descriptions(src):
    """For the participant notebook: blank each tool's top-level description (the one that follows
    "name"), leaving the rest of the schema intact. That single field is Stage 2's YOUR TURN."""
    return re.sub(r'("name":\s*"[^"]+",\s*\n\s*"description":\s*)"[^"]*"',
                  r'\1"✏️ FILL IN - one sentence: WHEN should the model use this tool?"', src)

def build_cells(spec, solution):
    g = spec
    C = []
    C.append(md(
        f"# {g['name']} - Build-Along\n\n"
        f"**Group {g['num']} · Level: Intermediate · ~40 min**\n\n"
        f"**The pain:** {g['pain']}\n\n"
        f"**Your mission:** {g['mission']}\n\n"
        f"> **What \"great\" looks like:** {g['wow']}"))
    C.append(md(
        "## How this works - build it, then measure it\n\n"
        "You won't build blind. You'll build an agent **and a way to score it**, then improve it in "
        "stages and watch the score climb.\n\n"
        "**The agent** is a model that runs in a loop: it can call **tools** (so it can *act*, not just "
        "talk), and a **guardrail** wraps the loop as a safety net.\n\n"
        "**The method** is *evals first*: define test tasks + graders, measure a **baseline**, then improve "
        "and **re-measure**. You'll make three small edits - the **brief**, the **tool descriptions**, and "
        "**one line of the guardrail** - and after each you re-run the eval and the bar fills:\n\n"
        "| Stage | You add | Expect |\n|---|---|---|\n"
        "| 0 · Baseline | *(given)* a naive prompt, no tools | a low score - the starting line |\n"
        "| 1 · Instructions | the brief (plain English) | ↑ |\n"
        "| 2 · Tools | describe the agent's tools | ↑↑ |\n"
        "| 3 · Guardrail | fill one safety-check line | ↑↑↑ target |\n\n"
        "Run the cells top to bottom. Each cell prints what it did; green banners = your edit passed its check."))

    C.append(md("## Step 0 · Setup - connect to the model\n\nRun this first. Paste your key into the `.env` when prompted (never into a cell). A green banner means you're connected."))
    C.append(code(SETUP_CODE))

    C.append(md(
        "## Step 0 · The data + the agent's tools (provided)\n\n"
        "A **tool** is just a Python function the agent is allowed to call - that's how it *does* things "
        "instead of only talking. This cell loads the sample data and the tool function(s). You don't edit "
        "it; you'll *describe* the tools in Stage 2 so the model knows when to use them."))
    C.append(code(g["data_code"]))

    C.append(md(
        "## Step 0 · The agent engine + the eval harness (provided)\n\n"
        "`run_agent()` runs the agent loop. `run_eval()` scores the agent on the test tasks and draws a "
        "progress bar. Just run this cell - you never edit it."))
    C.append(code(HARNESS_CODE))

    C.append(md(
        "## Step 0 · How we measure it - the eval\n\n"
        "An **eval** is a set of test tasks plus **graders** (little checks that pass or fail). Running the "
        "eval after each stage is how you *know* the agent got better - not \"it looked fine when I tried it.\" "
        "Run this to load the tasks and graders."))
    C.append(code(g["eval_code"]))

    # Stage 0 - baseline
    C.append(md(
        "---\n## Stage 0 · Baseline (given)\n\n"
        "Here's a naive agent: a one-line prompt, no tools, no guardrail. First **build** it, then **run the "
        "eval**. This is your starting line - the number you're about to beat."))
    C.append(code(
        f'NAIVE_PROMPT = "{g["naive_prompt"]}"\n'
        'baseline_agent = lambda inp: run_agent(inp, system_prompt=NAIVE_PROMPT)\n'
        'print("✓ Built the BASELINE agent - naive prompt, no tools, no guardrail")'))
    C.append(md("### Run the baseline eval"))
    C.append(code('run_eval(baseline_agent, "Stage 0 · Baseline")'))

    # Stage 1 - instructions
    C.append(md(
        "---\n## Stage 1 · Instructions - write the brief\n\n"
        "The **brief** (system prompt) is the agent's job description: its role, its steps, and the one thing "
        "it must never get wrong. It's the single biggest lever, and it's just plain English.\n\n"
        "### ✏️ YOUR TURN 1 - write `SYSTEM_PROMPT`\n"
        "In the next cell, replace each **FILL IN** with your own words (the hints tell you what to say). "
        "Then run the two cells after it - build the agent, run the eval, watch the score climb."))
    yt1 = (g["system_prompt"] if solution else g["system_prompt_blank"]) + \
          '\nprint("✓ SYSTEM_PROMPT saved - " + str(len(SYSTEM_PROMPT)) + " characters")'
    C.append(code(yt1))
    C.append(md("### Build the Stage 1 agent"))
    C.append(code(
        'agent_v1 = lambda inp: run_agent(inp, system_prompt=SYSTEM_PROMPT)\n'
        'print("✓ Built the Stage 1 agent - now guided by your brief")'))
    C.append(md("### Run the eval + check your brief"))
    C.append(code(
        'run_eval(agent_v1, "Stage 1 · Instructions")\n'
        'kw = [k for k in ' + repr(g["brief_keywords"]) + ' if k.lower() in SYSTEM_PROMPT.lower()]\n'
        '_check("fill in" not in SYSTEM_PROMPT.lower() and len(SYSTEM_PROMPT) > 200 and len(kw) >= 3,\n'
        '       "Brief check: " + str(len(kw)) + " key ideas covered, and no FILL-IN markers left")'))
    C.append(md("### ✅ Checkpoint 1\nThe agent has a real brief and the score moved. But it still can't *act* - only talk. Next: tools."))

    # Stage 2 - tools
    C.append(md(
        "---\n## Stage 2 · Tools - let the agent act\n\n"
        "A **tool** is a function the agent can call. The Python for the tools is already written (Step 0 "
        "cell). What the model needs from you is a good **description** of each tool - that's literally how it "
        "decides *when* to use it. Vague description → the model won't call it at the right time.\n\n"
        f"{g['tool_intro']}\n\n"
        "### ✏️ YOUR TURN 2 - write the tool `description`(s)\n"
        "In the next cell the schema is filled in EXCEPT the top **`description`** of each tool (marked "
        "**FILL IN**). Write one clear sentence per tool: *when* should the model use it? Then run the two "
        "cells after - build, eval, watch the jump."))
    tool_src = (g["tool_code"] if solution else _blank_tool_descriptions(g["tool_code"]))
    tool_src += "\nprint('✓ Defined ' + str(len(tools)) + ' tool schema(s): ' + str([t['name'] for t in tools]))"
    C.append(code(tool_src))
    C.append(md("### Build the Stage 2 agent"))
    C.append(code(
        'agent_v2 = lambda inp: run_agent(inp, system_prompt=SYSTEM_PROMPT, tools=tools, tool_functions=TOOL_FUNCTIONS)\n'
        'print("✓ Built the Stage 2 agent - brief + tools")'))
    C.append(md("### Run the eval + check your tools"))
    C.append(code(
        'run_eval(agent_v2, "Stage 2 · Tools + loop")\n'
        'names = {t.get("name") for t in tools}\n'
        'descs_ok = all(t.get("description") and "FILL IN" not in t.get("description", "") and len(t.get("description", "")) > 15 for t in tools)\n'
        '_check("' + g["tool_name"] + '" in names and descs_ok and len(tools) >= 1,\n'
        '       "Tool check: " + str(len(tools)) + " tool(s) defined, descriptions written")'))
    C.append(md("### ✅ Checkpoint 2\nThe agent can act now, and its output is structured enough to grade. One gap remains - the high-stakes failure. That's the guardrail."))

    # Stage 3 - guardrail
    C.append(md(
        "---\n## Stage 3 · Guardrail - the safety net\n\n"
        "Models are good, not perfect. For anything high-stakes you don't *hope* - you add a **guardrail**: a "
        "plain rule that wraps the agent and catches the failure the model might make. This is the "
        "healthcare-grade move.\n\n"
        "### ✏️ YOUR TURN 3 - fill ONE line in `guardrail`\n"
        "The whole guardrail is written for you EXCEPT one line marked **FILL IN** - the check that decides "
        "whether the guardrail passes. Replace `False` with the expression shown in the comment. Then run the "
        "two cells after - build, eval, jump to target."))
    guard_src = (g["guardrail_code"] if solution else g["guardrail_blank"]) + \
                '\nprint("✓ guardrail() is defined")'
    C.append(code(guard_src))
    C.append(md("### Build the Stage 3 agent"))
    C.append(code(
        'GUARDRAIL = guardrail\n'
        'agent_v3 = lambda inp: run_agent(inp, system_prompt=SYSTEM_PROMPT, tools=tools, tool_functions=TOOL_FUNCTIONS, guardrail=GUARDRAIL)\n'
        'print("✓ Built the Stage 3 agent - brief + tools + guardrail")'))
    C.append(md("### Run the eval + check your guardrail"))
    C.append(code(
        'final = run_eval(agent_v3, "Stage 3 · Guardrail")\n'
        '_check(final >= 0.7, "Guardrail check: final eval score " + str(round(final * 100)) + "% (target 70%+ on this multi-task eval)")'))

    C.append(md("---\n## 🏁 Your agent's progress\nRun this to see the whole journey - baseline to guarded agent."))
    C.append(code('scoreboard()'))

    C.append(md(
        "---\n## See it work\nRun the finished agent on a sample input and look at what it produced - and what the guardrail did."))
    C.append(code(
        'out = agent_v3(' + g["demo_input"] + ')\n'
        'print("TEXT:\\n", out["text"][:800])\n'
        'print("\\nRECORDS (what the tools + guardrail produced):")\n'
        'for r in out["records"]:\n'
        '    print("  ", json.dumps(r, ensure_ascii=False))'))

    ec = "\n".join(f"{i+1}. {x}" for i, x in enumerate(g["extra"]))
    C.append(md(
        "---\n## Extra credit (for fast tables)\n\n" + ec +
        "\n\n**The judgment to take home:** a prototype proves an idea; production needs reliability, grounding, "
        "guardrails that hold, observability, and an owner. Knowing where that line sits is the real skill."))
    return C

def write_notebook(path, cells):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, indent=1))

def write_py(path, cells, spec):
    parts = [f'"""{spec["name"]} - SOLUTION / answer key.',
             "Auto-generated reference. Participants work the .ipynb (blanked); this .py is the",
             "completed version facilitators can run straight through or diff against.",
             'Run: python3 ' + path.name + '"""', ""]
    for c in cells:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            parts.append("\n" + "\n".join("# " + l for l in src.splitlines()) + "\n")
        else:
            parts.append(src + "\n")
    path.write_text("\n".join(parts))

def group_readme(spec):
    g = spec
    return f"""# {g['name']} · Group {g['num']}

**Level: Intermediate · ~40 min · Build-Along**

## The pain
{g['pain']}

## Your mission
{g['mission']}

## What "great" looks like
{g['wow']}

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
1. Open **`{g['stem']}.ipynb`** in VS Code / Cursor (Python + Jupyter extensions).
2. Run the **Setup** cell; paste your Anthropic key into the `.env` it creates (never into a cell).
3. Run top to bottom. Do the three **✏️ YOUR TURN** cells. Watch the score climb.

Want the answer key? **`{g['stem']}.py`** is the completed solution - run it straight through with
`python3 {g['stem']}.py`.
"""

# ───────────────────────── write everything ─────────────────────────

OUT.mkdir(parents=True, exist_ok=True)

for g in GROUPS:
    d = OUT / g["slug"]
    d.mkdir(parents=True, exist_ok=True)
    write_notebook(d / f'{g["stem"]}.ipynb', build_cells(g, solution=False))
    write_py(d / f'{g["stem"]}.py', build_cells(g, solution=True), g)
    (d / "README.md").write_text(group_readme(g))
    print("wrote", g["slug"])

# root files
(OUT / "requirements.txt").write_text(
    "# Optia Agentic Workshop - Python deps. Fastest setup (avoids PEP 668):\n"
    "#   python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate\n"
    "#   pip install -r requirements.txt\n"
    "# then select the .venv interpreter as your notebook kernel. See SETUP.md.\n\n"
    "anthropic>=0.69\n")

(OUT / ".gitignore").write_text(
    ".env\n.venv/\n__pycache__/\n*.pyc\n.ipynb_checkpoints/\n.DS_Store\n"
    "# Facilitator + workshop-kit materials are hosted separately, not in the public repo:\n"
    "FACILITATOR.md\nworkshop-kit/\n")

_ENV_BODY = (
    "# Paste your Anthropic API key after the =, then save. Get one at https://console.anthropic.com/\n"
    "# One .env at the repo root serves all five groups.\n"
    "ANTHROPIC_API_KEY=paste-your-key-here\n")
(OUT / ".env.example").write_text(_ENV_BODY)
if not (OUT / ".env").exists():
    (OUT / ".env").write_text(_ENV_BODY)

(OUT / "SETUP.md").write_text("""# Setup & Troubleshooting

One page to get any laptop - including a locked-down corporate one - running these notebooks.
**Fastest reliable path: a virtual environment in VS Code.**

## The 4-command setup (once)
```bash
python3 -m venv .venv                 # isolated environment
source .venv/bin/activate             # macOS/Linux  (Windows: .venv\\Scripts\\activate)
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
""")

(OUT / "README.md").write_text("""# Optia Agentic Workshop - Build a Working AI Agent

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
""")

(OUT / "FACILITATOR.md").write_text("""# Facilitator Guide - Optia Agentic Workshop

*90-minute hands-on build. 20-25 people, ~5 per table, one agent per table.* Do not distribute.

## What participants build
One agent = a model in a loop (with tools) + a guardrail, developed evals-first. Three tiny edits
(**✏️ YOUR TURN**): the **brief** (plain English), the tool **descriptions** (schema pre-filled), and
**one line** of the guardrail. After each, they run a build cell then an eval cell; a progress bar climbs
from a weak baseline to 80%+ target. The rising score is the engine of the "sense of accomplishment."

## Suggested timing
| Min | Block |
|---|---|
| 0-10 | Frame: what an agent is (model + loop + tools + guardrails), why **evals** matter. Live-run one Baseline so they see a low score. |
| 10-20 | Setup: everyone gets `✓ API key verified`. Float hard here - key/`.env`/kernel issues cluster now. |
| 20-30 | Stage 1 (brief): replace the FILL-IN lines. Everyone should beat Baseline. |
| 30-45 | Stage 2 (tool descriptions): one sentence per tool. |
| 45-60 | Stage 3 (guardrail): replace `False` with the one-line check. The jump to target. |
| 60-75 | `scoreboard()` + "See it work". Tables screenshot their progress bar. |
| 75-90 | Share-outs (2-3 min/table): the problem, the guardrail, the AI-fit verdict. Land the prototype-vs-production point. |

## Every YOUR TURN is small on purpose
- **Stage 1** - replace `FILL IN` lines in a prose template. No code.
- **Stage 2** - write the one-sentence `description` for each tool (schema is already there).
- **Stage 3** - replace `False` with the expression printed in the comment right above it.
Nobody writes a function or a schema from scratch. The check banner under each eval says what (if anything) is missing.

## Answer keys
Each group's `<Name>.py` is the completed solution - run `python3 <Name>.py` to demo a finished agent, or
diff against a stuck table's notebook.

## Per-group: what "great" looks like
- **G1 Voice-of-Customer.** Great = the safety backstop catches a planted comment. If flat: "what's the one comment you'd never want this to miss?"
- **G2 Requirement-to-Story.** Great = speculative "nice to haves" land in assumptions, not committed stories. If flat: "what if it commits scope nobody asked for?"
- **G3 Market Scan.** Great = confirmed vs unverified cleanly split, sourced. If flat: "how would a leader know what to trust?"
- **G4 Launch Comms.** Great = compliance banner on the patient version + no overstated benefit. If flat: "what if the patient copy overstates a benefit?"
- **G5 Roadmap Tradeoff.** Great = 2+ scenarios each naming a tradeoff; final call left to humans. If flat: "who decides, and what does the agent hand them?"

## Common intervention: the eval didn't move
Usually one of: FILL-IN markers left in the brief (Stage 1), a tool description left blank (Stage 2), or the
guardrail line still `= False` (Stage 3). The red check banner under each eval says which.
""")

print("ROOT FILES WRITTEN")
print("DONE")
