"""Launch & Release Comms Agent - SOLUTION / answer key.
Auto-generated reference. Participants work the .ipynb (blanked); this .py is the
completed version facilitators can run straight through or diff against.
Run: python3 Launch_Comms_Agent.py"""


# # Launch & Release Comms Agent - Build-Along
# 
# **Group 4 · Level: Intermediate · ~40 min**
# 
# **The pain:** Every launch, the same update gets rewritten for five audiences - executives, ECPs, internal ops, patients, sales. It's slow, repetitive, and tone drifts. Worse, a patient version can quietly overstate a clinical benefit.
# 
# **Your mission:** Build an agent that turns one source of truth (a feature list + locked clinical facts) into audience-tailored comms - right message, tone, and detail for each - without ever altering a regulated claim.
# 
# > **What "great" looks like:** The locked-facts rule plus the compliance banner on the patient version. Highest-stakes agent of the day: regulated claims cannot be reworded or embellished by the model.


# ## How this works - build it, then measure it
# 
# You won't build blind. You'll build an agent **and a way to score it**, then improve it in stages and watch the score climb.
# 
# **The agent** is a model that runs in a loop: it can call **tools** (so it can *act*, not just talk), and a **guardrail** wraps the loop as a safety net.
# 
# **The method** is *evals first*: define test tasks + graders, measure a **baseline**, then improve and **re-measure**. You'll make three small edits - the **brief**, the **tool descriptions**, and **one line of the guardrail** - and after each you re-run the eval and the bar fills:
# 
# | Stage | You add | Expect |
# |---|---|---|
# | 0 · Baseline | *(given)* a naive prompt, no tools | a low score - the starting line |
# | 1 · Instructions | the brief (plain English) | ↑ |
# | 2 · Tools | describe the agent's tools | ↑↑ |
# | 3 · Guardrail | fill one safety-check line | ↑↑↑ target |
# 
# Run the cells top to bottom. Each cell prints what it did; green banners = your edit passed its check.


# ## Step 0 · Setup - connect to the model
# 
# Run this first. Paste your key into the `.env` when prompted (never into a cell). A green banner means you're connected.

# ── Install packages + connect to the AI model ──
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

_ensure_packages([("litellm", "litellm")])
import litellm, json, os, pathlib, time
import logging; logging.getLogger("LiteLLM").setLevel(logging.ERROR); litellm.suppress_debug_info = True
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

# LiteLLM talks to many providers with one API. Put every key from .env into the environment
# so it can pick the right one; a matching shell/kernel key wins.
for _k, _v in _file.items():
    if _v and not os.environ.get(_k):
        os.environ[_k] = _v
_has_openai = os.environ.get("OPENAI_API_KEY", "").startswith("sk-")
_has_anthropic = os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-")
if not (_has_openai or _has_anthropic):
    raise SystemExit(f"No API key yet. Open {_env} and paste OPENAI_API_KEY=sk-... or ANTHROPIC_API_KEY=sk-ant-..., save, re-run.")

# Pick the model. Set WORKSHOP_MODEL in .env to any LiteLLM model id to override
# (e.g. gpt-4o, gpt-4o-mini, anthropic/claude-sonnet-5, gemini/gemini-1.5-pro).
MODEL = os.environ.get("WORKSHOP_MODEL", "").strip() or ("gpt-4o-mini" if _has_openai else "anthropic/claude-sonnet-5")
JUDGE_MODEL = os.environ.get("WORKSHOP_JUDGE", "").strip() or ("gpt-4o-mini" if MODEL.startswith(("gpt", "openai")) else "anthropic/claude-haiku-4-5")
litellm.drop_params = True   # silently ignore params a given provider does not support
try:
    litellm.completion(model=MODEL, max_tokens=4, messages=[{"role": "user", "content": "ping"}])
except Exception as e:
    _status(False, f"Could not verify model {MODEL} ({type(e).__name__}). Check the key for that provider and the model id, then re-run.")
    raise SystemExit("Model not reachable.")
_status(True, f"Connected via LiteLLM - model: {MODEL}")


# ## Step 0 · The data + the agent's tools (provided)
# 
# A **tool** is just a Python function the agent is allowed to call - that's how it *does* things instead of only talking. This cell loads the sample data and the tool function(s). You don't edit it; you'll *describe* the tools in Stage 2 so the model knows when to use them.

# ── Sample data + the agent's tools (provided - just run it) ──
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


# ## Step 0 · The agent engine + the eval harness (provided)
# 
# `run_agent()` runs the agent loop. `run_eval()` scores the agent on the test tasks and draws a progress bar. Just run this cell - you never edit it.

# ── The agent engine + the eval harness (provided - just run it) ──
# run_agent() is the agent loop: it reasons, optionally calls a tool, sees the result, repeats.
# run_eval() scores the agent on the test tasks. You never edit this cell.

RECORDS = []  # tools that "record" something append here; cleared at the start of each run

def run_agent(user_input, system_prompt="", tools=None, tool_functions=None, guardrail=None, max_turns=10):
    """The agent loop: reason -> (maybe call a tool) -> observe -> repeat, with an optional
    guardrail wrapping the final output. Returns {"text": ..., "records": [...]}."""
    RECORDS.clear()
    tools = tools or []
    tool_functions = tool_functions or {}
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})
    final_text = ""
    for _ in range(max_turns):
        kwargs = dict(model=MODEL, max_tokens=4000, messages=messages)
        if tools:
            kwargs["tools"] = tools
        resp = litellm.completion(**kwargs)
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if calls:
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            for c in calls:
                fn = tool_functions.get(c.function.name)
                try:
                    args = json.loads(c.function.arguments or "{}")
                except Exception:
                    args = {}
                out = fn(**args) if fn else json.dumps({"error": "unknown tool " + c.function.name})
                messages.append({"role": "tool", "tool_call_id": c.id, "content": str(out)})
            continue
        final_text = msg.content or ""
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
    r = litellm.completion(model=JUDGE_MODEL, max_tokens=5,
        messages=[{"role": "user", "content": f"{question}\n\n---\n{content}\n---\nAnswer with exactly YES or NO."}])
    return "yes" in (r.choices[0].message.content or "").strip().lower()

def _check(passed, msg):
    _status(passed, msg)
    return passed

print("✓ Agent engine + eval harness ready - run_agent(), run_eval(), scoreboard()")


# ## Step 0 · How we measure it - the eval
# 
# An **eval** is a set of test tasks plus **graders** (little checks that pass or fail). Running the eval after each stage is how you *know* the agent got better - not "it looked fine when I tried it." Run this to load the tasks and graders.

# ── Eval tasks + graders (provided - just run it) ──
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
_OVERCLAIM = ["cure", "cures", "guaranteed", "guarantee", "risk-free", "risk free", "best ever", "permanent", "miracle", "100%"]
# A real eval, run on TWO launches: audience coverage, tailoring (judged), patient copy clean of
# overstated wording (rule + judge), and the guardrail banner check. Deterministic graders
# (coverage + overclaim + guardrail) form the Stage-3 floor; the judges add on top.
GRADERS = [
    {"name": "all 5 audiences produced (needs the tool)", "weight": 2,
     "fn": lambda t, o: {r["audience"] for r in _versions(o)} >= set(AUDIENCES)},
    {"name": "messages tailored per audience", "weight": 1, "judge": True,
     "fn": lambda t, o: judge_yes("Are these messages clearly tailored to different audiences (executive, clinician, patient, etc.)?", "\n\n".join(f'[{r["audience"]}] {r.get("message", "")}' for r in _versions(o)) or o["text"])},
    {"name": "patient copy avoids overstated wording", "weight": 1,
     "fn": lambda t, o: _patient(o) is not None and not any(w in _patient(o)["message"].lower() for w in _OVERCLAIM)},
    {"name": "patient copy reads plain and reassuring", "weight": 1, "judge": True,
     "fn": lambda t, o: _patient(o) is not None and judge_yes("Is this patient message plain and reassuring WITHOUT overstating any clinical benefit?", _patient(o)["message"])},
    {"name": "guardrail: patient banner present + verified", "weight": 3,
     "fn": lambda t, o: _audit_ok(o) and _patient(o) is not None and COMPLIANCE_BANNER in _patient(o)["message"]},
]
print(f"✓ Loaded {len(EVAL_TASKS)} eval tasks and {len(GRADERS)} graders - coverage, tone (judge), overstatement, guardrail")


# ---
# ## Stage 0 · Baseline (given)
# 
# Here's a naive agent: a one-line prompt, no tools, no guardrail. First **build** it, then **run the eval**. This is your starting line - the number you're about to beat.

NAIVE_PROMPT = "Give me a rough one-line blurb for these features."
baseline_agent = lambda inp: run_agent(inp, system_prompt=NAIVE_PROMPT)
print("✓ Built the BASELINE agent - naive prompt, no tools, no guardrail")


# ### Run the baseline eval

run_eval(baseline_agent, "Stage 0 · Baseline")


# ---
# ## Stage 1 · Instructions - write the brief
# 
# The **brief** (system prompt) is the agent's job description: its role, its steps, and the one thing it must never get wrong. It's the single biggest lever, and it's just plain English.
# 
# ### ✏️ YOUR TURN 1 - write `SYSTEM_PROMPT`
# In the next cell, replace each **FILL IN** with your own words (the hints tell you what to say). Then run the two cells after it - build the agent, run the eval, watch the score climb.

SYSTEM_PROMPT = """You are the Launch & Release Comms agent for Optia.
From one feature list you write a version for each audience.

Do this:
1. Use the locked clinical facts VERBATIM - never reword, round, or embellish a regulated claim, and
   don't add benefits that aren't stated. (When the get_locked_facts tool is available, get them first.)
2. For each audience (executive, ECP, internal_ops, patient, sales) write a suitably-toned message.
   (When the save_version tool is available, record each with it.)
3. The patient version MUST include the compliance banner and must not overstate any benefit.

Overstating a clinical benefit to a patient is the worst outcome. Write all five versions in prose."""
print("✓ SYSTEM_PROMPT saved - " + str(len(SYSTEM_PROMPT)) + " characters")


# ### Build the Stage 1 agent

agent_v1 = lambda inp: run_agent(inp, system_prompt=SYSTEM_PROMPT)
print("✓ Built the Stage 1 agent - now guided by your brief")


# ### Run the eval + check your brief

run_eval(agent_v1, "Stage 1 · Instructions")
kw = [k for k in ['locked', 'audience', 'patient', 'compliance', 'verbatim'] if k.lower() in SYSTEM_PROMPT.lower()]
_check("fill in" not in SYSTEM_PROMPT.lower() and len(SYSTEM_PROMPT) > 200 and len(kw) >= 3,
       "Brief check: " + str(len(kw)) + " key ideas covered, and no FILL-IN markers left")


# ### ✅ Checkpoint 1
# The agent has a real brief and the score moved. But it still can't *act* - only talk. Next: tools.


# ---
# ## Stage 2 · Tools - let the agent act
# 
# A **tool** is a function the agent can call. The Python for the tools is already written (Step 0 cell). What the model needs from you is a good **description** of each tool - that's literally how it decides *when* to use it. Vague description → the model won't call it at the right time.
# 
# Two tools are provided above: **`get_locked_facts`** returns the exact regulated claims that must not change, and **`save_version`** records one audience's message. You describe both.
# 
# ### ✏️ YOUR TURN 2 - write the tool `description`(s)
# In the next cell the schema is filled in EXCEPT the top **`description`** of each tool (marked **FILL IN**). Write one clear sentence per tool: *when* should the model use it? Then run the two cells after - build, eval, watch the jump.

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_locked_facts",
            "description": "Retrieve the regulated clinical facts (which must appear word-for-word) and the patient compliance banner. Call this FIRST.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_version",
            "description": "Save one audience-tailored version of the announcement. Any locked clinical fact included must be verbatim; the patient version must include the compliance banner.",
            "parameters": {
                "type": "object",
                "properties": {
                    "audience": {
                        "type": "string",
                        "enum": [
                            "executive",
                            "ECP",
                            "internal_ops",
                            "patient",
                            "sales"
                        ]
                    },
                    "message": {
                        "type": "string"
                    }
                },
                "required": [
                    "audience",
                    "message"
                ]
            }
        }
    }
]
print('✓ Defined ' + str(len(tools)) + ' tool schema(s): ' + str([(t.get('function') or t)['name'] for t in tools]))


# ### Build the Stage 2 agent

agent_v2 = lambda inp: run_agent(inp, system_prompt=SYSTEM_PROMPT, tools=tools, tool_functions=TOOL_FUNCTIONS)
print("✓ Built the Stage 2 agent - brief + tools")


# ### Run the eval + check your tools

run_eval(agent_v2, "Stage 2 · Tools + loop")
names = {(t.get("function") or t).get("name") for t in tools}
descs_ok = all((lambda d: d and "FILL IN" not in d and len(d) > 15)((t.get("function") or t).get("description", "")) for t in tools)
_check("save_version" in names and descs_ok and len(tools) >= 1,
       "Tool check: " + str(len(tools)) + " tool(s) defined, descriptions written")


# ### ✅ Checkpoint 2
# The agent can act now, and its output is structured enough to grade. One gap remains - the high-stakes failure. That's the guardrail.


# ---
# ## Stage 3 · Guardrail - the safety net
# 
# Models are good, not perfect. For anything high-stakes you don't *hope* - you add a **guardrail**: a plain rule that wraps the agent and catches the failure the model might make. This is the healthcare-grade move.
# 
# ### ✏️ YOUR TURN 3 - fill ONE line in `guardrail`
# The whole guardrail is written for you EXCEPT one line marked **FILL IN** - the check that decides whether the guardrail passes. Replace `False` with the expression shown in the comment. Then run the two cells after - build, eval, jump to target.

def guardrail(user_input, output):
    # Regulated comms: the patient version MUST carry the compliance banner. Inject it if missing.
    versions = [r for r in output["records"] if r.get("audience")]
    patient = next((r for r in versions if r["audience"] == "patient"), None)
    if patient and COMPLIANCE_BANNER not in patient["message"]:
        patient["message"] += "\n\n" + COMPLIANCE_BANNER
    # The guardrail passes only if the patient version now carries the compliance banner:
    banner_ok = patient is not None and COMPLIANCE_BANNER in patient["message"]
    output["records"].append({"kind": "audit", "guardrail": "compliance_check", "ok": banner_ok})
    return output
print("✓ guardrail() is defined")


# ### Build the Stage 3 agent

GUARDRAIL = guardrail
agent_v3 = lambda inp: run_agent(inp, system_prompt=SYSTEM_PROMPT, tools=tools, tool_functions=TOOL_FUNCTIONS, guardrail=GUARDRAIL)
print("✓ Built the Stage 3 agent - brief + tools + guardrail")


# ### Run the eval + check your guardrail

final = run_eval(agent_v3, "Stage 3 · Guardrail")
_check(final >= 0.7, "Guardrail check: final eval score " + str(round(final * 100)) + "% (target 70%+ on this multi-task eval)")


# ---
# ## 🏁 Your agent's progress
# Run this to see the whole journey - baseline to guarded agent.

scoreboard()


# ---
# ## See it work
# Run the finished agent on a sample input and look at what it produced - and what the guardrail did.

out = agent_v3(_launch_input("PreciSight lens"))
print("TEXT:\n", out["text"][:800])
print("\nRECORDS (what the tools + guardrail produced):")
for r in out["records"]:
    print("  ", json.dumps(r, ensure_ascii=False))


# ---
# ## Extra credit (for fast tables)
# 
# 1. **Approval gate:** mark any version mentioning a clinical claim as 'needs medical/legal sign-off' before it can 'send'.
# 2. **Reflection:** have the agent re-read the patient version as a compliance officer and rewrite if it overstates.
# 3. **Diff view:** print, per version, exactly which locked facts it used verbatim.
# 
# **The judgment to take home:** a prototype proves an idea; production needs reliability, grounding, guardrails that hold, observability, and an owner. Knowing where that line sits is the real skill.
