"""Requirement-to-Story Agent - SOLUTION / answer key.
Auto-generated reference. Participants work the .ipynb (blanked); this .py is the
completed version facilitators can run straight through or diff against.
Run: python3 Requirement_To_Story_Agent.py"""


# # Requirement-to-Story Agent - Build-Along
# 
# **Group 2 · Level: Intermediate · ~40 min**
# 
# **The pain:** Stakeholder requests arrive messy - a paragraph in an email, a hallway chat, a vague "make it easier for clinics to reorder." Turning that into clear, well-formed user stories eats hours of product time.
# 
# **Your mission:** Build an agent that turns a messy request into structured, decomposed user stories - each with acceptance criteria - ready for the backlog, WITHOUT inventing scope nobody asked for.
# 
# > **What "great" looks like:** The 'Assumptions to confirm' output is the star. The danger isn't bad stories - it's confident invented scope. A great agent separates what was asked from what it assumed.


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

# No third-party packages needed - the model is called over plain HTTPS with the standard library.
import json, os, pathlib, time, urllib.request, urllib.error

# --- tiny zero-dependency model client (OpenAI + Anthropic) ---
class _Fn:
    def __init__(self, name, arguments): self.name = name; self.arguments = arguments
class _ToolCall:
    def __init__(self, id, name, arguments): self.id = id; self.function = _Fn(name, arguments)
class _Reply:
    def __init__(self, text, tool_calls): self.text = text; self.tool_calls = tool_calls

def _provider(model):
    m = model.lower()
    if m.startswith("anthropic/") or "claude" in m: return "anthropic"
    if m.startswith("openai/") or m.startswith(("gpt", "o1", "o3", "o4")): return "openai"
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-") else "openai"

def _post(url, headers, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={**headers, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:400]}")

def _anthropic_tool(t):
    f = t.get("function", t)
    return {"name": f["name"], "description": f.get("description", ""),
            "input_schema": f.get("parameters", {"type": "object", "properties": {}})}

def _to_anthropic(messages):
    out = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant":
            content = []
            if m.get("content"): content.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls", []):
                content.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                                "input": json.loads(tc["function"]["arguments"] or "{}")})
            out.append({"role": "assistant", "content": content or "..."})
        else:
            out.append({"role": "user", "content": m["content"]})
    return out

def model_call(model, messages, tools=None, max_tokens=4000):
    """One call to the model, provider chosen from the model id / which key is set.
    Returns an object with .text and .tool_calls (each .id, .function.name, .function.arguments)."""
    prov = _provider(model)
    mid = model.split("/", 1)[1] if "/" in model else model
    if prov == "openai":
        hdr = {"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "")}
        payload = {"model": mid, "messages": messages, "max_tokens": max_tokens}
        if tools: payload["tools"] = tools
        try:
            data = _post("https://api.openai.com/v1/chat/completions", hdr, payload)
        except RuntimeError as e:
            if "max_tokens" in str(e):   # some newer models want max_completion_tokens
                payload.pop("max_tokens"); payload["max_completion_tokens"] = max_tokens
                data = _post("https://api.openai.com/v1/chat/completions", hdr, payload)
            else:
                raise
        msg = data["choices"][0]["message"]
        tcs = [_ToolCall(tc["id"], tc["function"]["name"], tc["function"]["arguments"]) for tc in (msg.get("tool_calls") or [])]
        return _Reply(msg.get("content") or "", tcs)
    else:
        hdr = {"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""), "anthropic-version": "2023-06-01"}
        sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
        conv = [m for m in messages if m["role"] != "system"]
        payload = {"model": mid, "max_tokens": max_tokens, "messages": _to_anthropic(conv)}
        if sys_txt: payload["system"] = sys_txt
        if tools: payload["tools"] = [_anthropic_tool(t) for t in tools]
        data = _post("https://api.anthropic.com/v1/messages", hdr, payload)
        text = "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
        tcs = [_ToolCall(b["id"], b["name"], json.dumps(b["input"])) for b in data["content"] if b["type"] == "tool_use"]
        return _Reply(text, tcs)
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

# Put every key from .env into the environment so the right provider is used; a matching
# shell/kernel key wins. No packages needed - the model is called over plain HTTPS.
for _k, _v in _file.items():
    if _v and not os.environ.get(_k):
        os.environ[_k] = _v
_has_openai = os.environ.get("OPENAI_API_KEY", "").startswith("sk-")
_has_anthropic = os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-")
if not (_has_openai or _has_anthropic):
    raise SystemExit(f"No API key yet. Open {_env} and paste OPENAI_API_KEY=sk-... or ANTHROPIC_API_KEY=sk-ant-..., save, re-run.")

# Pick the model. Set WORKSHOP_MODEL in .env to any OpenAI or Anthropic model id to override
# (e.g. gpt-4o, gpt-4o-mini, anthropic/claude-sonnet-5).
MODEL = os.environ.get("WORKSHOP_MODEL", "").strip() or ("gpt-4o" if _has_openai else "anthropic/claude-sonnet-5")
JUDGE_MODEL = os.environ.get("WORKSHOP_JUDGE", "").strip() or ("gpt-4o-mini" if MODEL.startswith(("gpt", "openai")) else "anthropic/claude-haiku-4-5")
try:
    model_call(MODEL, [{"role": "user", "content": "ping"}], max_tokens=4)
except Exception as e:
    _status(False, f"Could not verify model {MODEL} ({type(e).__name__}: {str(e)[:120]}). Check the key for that provider and the model id, then re-run.")
    raise SystemExit("Model not reachable.")
_status(True, f"Connected - model: {MODEL}")


# ## Step 0 · The data + the agent's tools (provided)
# 
# A **tool** is just a Python function the agent is allowed to call - that's how it *does* things instead of only talking. This cell loads the sample data and the tool function(s). You don't edit it; you'll *describe* the tools in Stage 2 so the model knows when to use them.

# ── Sample data + the agent's tools (provided - just run it) ──
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
        resp = model_call(MODEL, messages, tools=tools or None, max_tokens=4000)
        if resp.tool_calls:
            messages.append({"role": "assistant", "content": resp.text or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in resp.tool_calls]})
            for c in resp.tool_calls:
                fn = tool_functions.get(c.function.name)
                try:
                    args = json.loads(c.function.arguments or "{}")
                except Exception:
                    args = {}
                out = fn(**args) if fn else json.dumps({"error": "unknown tool " + c.function.name})
                messages.append({"role": "tool", "tool_call_id": c.id, "content": str(out)})
            continue
        final_text = resp.text or ""
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
    r = model_call(JUDGE_MODEL, [{"role": "user", "content": f"{question}\n\n---\n{content}\n---\nAnswer with exactly YES or NO."}], max_tokens=5)
    return "yes" in (r.text or "").strip().lower()

def _check(passed, msg):
    _status(passed, msg)
    return passed

print("✓ Agent engine + eval harness ready - run_agent(), run_eval(), scoreboard()")


# ## Step 0 · How we measure it - the eval
# 
# An **eval** is a set of test tasks plus **graders** (little checks that pass or fail). Running the eval after each stage is how you *know* the agent got better - not "it looked fine when I tried it." Run this to load the tasks and graders.

# ── Eval tasks + graders (provided - just run it) ──
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
print(f"✓ Loaded {len(EVAL_TASKS)} eval tasks and {len(GRADERS)} graders - quality (judge), traceability, scope, guardrail")


# ---
# ## Stage 0 · Baseline (given)
# 
# Here's a naive agent: a one-line prompt, no tools, no guardrail. First **build** it, then **run the eval**. This is your starting line - the number you're about to beat.

NAIVE_PROMPT = "Jot a rough user story for this request."
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

SYSTEM_PROMPT = """You are the Requirement-to-Story agent for an Optia product team.
You turn a messy stakeholder request into clear backlog items.

Do this:
1. For each real need that was asked for, write a user story ("As a <role>, I want <capability>,
   so that <benefit>") with 1-3 acceptance criteria, noting the exact source phrase it came from.
   (When the add_story tool is available, record each with it.)
2. SCOPE DISCIPLINE: if something was only floated or is a "nice to have" you can't trace to a clear
   ask, do NOT commit it as a story - call it out as an assumption. (When the flag_assumption tool is
   available, use it.) Confident invented scope is the failure mode here.
3. Anything ambiguous (which role? which platform?) → an assumption, don't guess.

Write your full answer in prose."""
print("✓ SYSTEM_PROMPT saved - " + str(len(SYSTEM_PROMPT)) + " characters")


# ### Build the Stage 1 agent

agent_v1 = lambda inp: run_agent(inp, system_prompt=SYSTEM_PROMPT)
print("✓ Built the Stage 1 agent - now guided by your brief")


# ### Run the eval + check your brief

run_eval(agent_v1, "Stage 1 · Instructions")
kw = [k for k in ['add_story', 'acceptance', 'source', 'assum', 'scope'] if k.lower() in SYSTEM_PROMPT.lower()]
_check("fill in" not in SYSTEM_PROMPT.lower() and len(SYSTEM_PROMPT) > 200 and len(kw) >= 3,
       "Brief check: " + str(len(kw)) + " key ideas covered, and no FILL-IN markers left")


# ### ✅ Checkpoint 1
# The agent has a real brief and the score moved. But it still can't *act* - only talk. Next: tools.


# ---
# ## Stage 2 · Tools - let the agent act
# 
# A **tool** is a function the agent can call. The Python for the tools is already written (Step 0 cell). What the model needs from you is a good **description** of each tool - that's literally how it decides *when* to use it. Vague description → the model won't call it at the right time.
# 
# Two tools are provided above: **`add_story`** records one finished user story (with the exact phrase it came from), and **`flag_assumption`** records something the agent is only assuming. You describe both.
# 
# ### ✏️ YOUR TURN 2 - write the tool `description`(s)
# In the next cell the schema is filled in EXCEPT the top **`description`** of each tool (marked **FILL IN**). Write one clear sentence per tool: *when* should the model use it? Then run the two cells after - build, eval, watch the jump.

tools = [
    {
        "type": "function",
        "function": {
            "name": "add_story",
            "description": "Record ONE committed user story that is clearly grounded in the request. Include the exact phrase from the request it came from as source_quote, so scope stays traceable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string"
                    },
                    "user_story": {
                        "type": "string",
                        "description": "As a <role>, I want <capability>, so that <benefit>"
                    },
                    "acceptance_criteria": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "source_quote": {
                        "type": "string",
                        "description": "Exact phrase from the request this story is grounded in"
                    }
                },
                "required": [
                    "title",
                    "user_story",
                    "acceptance_criteria",
                    "source_quote"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flag_assumption",
            "description": "Record something you are ASSUMING or that was only floated - NOT a committed story. Use for 'nice to have', ambiguous scope, or anything you cannot trace to a clear ask.",
            "parameters": {
                "type": "object",
                "properties": {
                    "assumption": {
                        "type": "string"
                    },
                    "why_it_matters": {
                        "type": "string"
                    }
                },
                "required": [
                    "assumption",
                    "why_it_matters"
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
_check("add_story" in names and descs_ok and len(tools) >= 1,
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

def _grounded(text, story):
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

out = agent_v3(REQUESTS["reorder"])
print("TEXT:\n", out["text"][:800])
print("\nRECORDS (what the tools + guardrail produced):")
for r in out["records"]:
    print("  ", json.dumps(r, ensure_ascii=False))


# ---
# ## Extra credit (for fast tables)
# 
# 1. **Reflection:** add a pass where the agent re-reads its own stories as a skeptical product owner and moves anything shaky to assumptions.
# 2. **Priority + estimate:** have each story carry a rough priority and t-shirt size.
# 3. **Adversarial request:** write a request loaded with vague 'nice to haves'. Can the guardrail keep them all out of committed scope?
# 
# **The judgment to take home:** a prototype proves an idea; production needs reliability, grounding, guardrails that hold, observability, and an owner. Knowing where that line sits is the real skill.
