"""Market & Competitive Scan Agent - SOLUTION / answer key.
Auto-generated reference. Participants work the .ipynb (blanked); this .py is the
completed version facilitators can run straight through or diff against.
Run: python3 Market_Scan_Agent.py"""


# # Market & Competitive Scan Agent - Build-Along
# 
# **Group 3 · Level: Intermediate · ~40 min**
# 
# **The pain:** Staying current on the product area - competitor moves, market signals, regulatory shifts - is a part-time job nobody has. Insights arrive late, or dressed-up rumors get treated as fact.
# 
# **Your mission:** Build an agent that monitors a set of signals and produces a short "what changed and why it matters" briefing - with every item clearly labeled confirmed vs unverified, and sourced.
# 
# > **What "great" looks like:** Confidence labels and the clean separation of confirmed vs unverified. You make the abstract 'explainability' point concrete: a leader can see at a glance what to trust.


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
MODEL = os.environ.get("WORKSHOP_MODEL", "").strip() or ("gpt-4o" if _has_openai else "anthropic/claude-sonnet-5")
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
print(f"✓ Loaded {len(EVAL_TASKS)} eval tasks and {len(GRADERS)} graders - quality (judge), labeling accuracy, guardrail")


# ---
# ## Stage 0 · Baseline (given)
# 
# Here's a naive agent: a one-line prompt, no tools, no guardrail. First **build** it, then **run the eval**. This is your starting line - the number you're about to beat.

NAIVE_PROMPT = "Give me a one-line market note from these signals."
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

SYSTEM_PROMPT = """You are the Market & Competitive Scan agent for an Optia product team.
You turn raw signals into a short "what changed and why it matters" briefing.

Do this:
1. For EVERY signal, decide whether it is confirmed or unverified and note its source. (When the
   verify_signal tool is available, use it to check the real status.)
2. Write one briefing item per signal - a headline, why it matters, the source, and a confidence
   label ("confirmed" or "unverified"). (When the add_briefing_item tool is available, record each with it.)
3. NEVER state an unverified item as fact - phrase rumors as claims ("reportedly", "unconfirmed")
   and label them clearly.

Write the full briefing in prose - a leader should see at a glance what to trust."""
print("✓ SYSTEM_PROMPT saved - " + str(len(SYSTEM_PROMPT)) + " characters")


# ### Build the Stage 1 agent

agent_v1 = lambda inp: run_agent(inp, system_prompt=SYSTEM_PROMPT)
print("✓ Built the Stage 1 agent - now guided by your brief")


# ### Run the eval + check your brief

run_eval(agent_v1, "Stage 1 · Instructions")
kw = [k for k in ['verify', 'confidence', 'confirmed', 'unverified', 'source'] if k.lower() in SYSTEM_PROMPT.lower()]
_check("fill in" not in SYSTEM_PROMPT.lower() and len(SYSTEM_PROMPT) > 200 and len(kw) >= 3,
       "Brief check: " + str(len(kw)) + " key ideas covered, and no FILL-IN markers left")


# ### ✅ Checkpoint 1
# The agent has a real brief and the score moved. But it still can't *act* - only talk. Next: tools.


# ---
# ## Stage 2 · Tools - let the agent act
# 
# A **tool** is a function the agent can call. The Python for the tools is already written (Step 0 cell). What the model needs from you is a good **description** of each tool - that's literally how it decides *when* to use it. Vague description → the model won't call it at the right time.
# 
# Two tools are provided above: **`verify_signal`** looks up whether a signal is confirmed or a rumor, and **`add_briefing_item`** records one briefing line with its confidence label. You describe both.
# 
# ### ✏️ YOUR TURN 2 - write the tool `description`(s)
# In the next cell the schema is filled in EXCEPT the top **`description`** of each tool (marked **FILL IN**). Write one clear sentence per tool: *when* should the model use it? Then run the two cells after - build, eval, watch the jump.

tools = [
    {
        "type": "function",
        "function": {
            "name": "verify_signal",
            "description": "Look up a signal's real source and verification status (confirmed/unverified). Call this BEFORE writing a briefing item about it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_id": {
                        "type": "string",
                        "description": "e.g. S-1"
                    }
                },
                "required": [
                    "signal_id"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_briefing_item",
            "description": "Record one briefing item. confidence MUST match the verified status. Never present an unverified item as fact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "headline": {
                        "type": "string"
                    },
                    "why_it_matters": {
                        "type": "string"
                    },
                    "source": {
                        "type": "string"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": [
                            "confirmed",
                            "unverified"
                        ]
                    }
                },
                "required": [
                    "headline",
                    "why_it_matters",
                    "source",
                    "confidence"
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
_check("add_briefing_item" in names and descs_ok and len(tools) >= 1,
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

out = agent_v3(_fmt_signals(SIGNAL_SET["batch_1"]))
print("TEXT:\n", out["text"][:800])
print("\nRECORDS (what the tools + guardrail produced):")
for r in out["records"]:
    print("  ", json.dumps(r, ensure_ascii=False))


# ---
# ## Extra credit (for fast tables)
# 
# 1. **Reflection:** add a fact-check pass that re-reads the briefing and downgrades anything phrased too confidently.
# 2. **Freshness:** add a date to each signal and have the agent flag anything stale as 'may be outdated'.
# 3. **Adversarial:** add a signal that sounds authoritative but is a rumor. Does the gate still catch it?
# 
# **The judgment to take home:** a prototype proves an idea; production needs reliability, grounding, guardrails that hold, observability, and an owner. Knowing where that line sits is the real skill.
