"""
╔══════════════════════════════════════════════════════════════════════════╗
║  DEEP AGENTS EXPLORER  —  Complete Edition                              ║
║  Uses the REAL  deepagents  library (pip install deepagents)            ║
║  langchain-ai/deepagents  ·  MIT  ·  built on LangGraph                ║
╚══════════════════════════════════════════════════════════════════════════╝

Core API used:
    from deepagents import create_deep_agent
    agent = create_deep_agent(
        model   = ChatOpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key="nvapi-…", model="nvidia/llama-3.3-70b-instruct"),
        tools   = [...],                  # your custom tools
        instructions = "...",            # main-agent system prompt
        subagents    = [                 # ← MULTIPLE sub-agents
            {"name": "analyst",  "description": "...", "system_prompt": "..."},
            {"name": "critic",   "description": "...", "system_prompt": "..."},
            {"name": "writer",   "description": "...", "system_prompt": "..."},
        ],
    )
    # Returns a LangGraph CompiledStateGraph
    for chunk in agent.stream({"messages": [HumanMessage(content=task)]}):
        ...

Built-in tools auto-injected by deepagents (no setup needed):
    write_todos  read_file  write_file  edit_file
    ls  glob  grep  task  compact_conversation
"""

import os, re, json, traceback, html as _html
from datetime import datetime
from io import StringIO

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Agents Explorer",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]         { font-family: 'Sora', sans-serif; }
.stApp                             { background: #05080f; color: #dde3f0; }
section[data-testid="stSidebar"]   { background: #070b16 !important;
                                     border-right: 1px solid #111d33; }
/* ── hero ── */
.hero {
    background: linear-gradient(140deg, #08111f 0%, #0c1a34 55%, #091424 100%);
    border: 1px solid #132032; border-radius: 18px;
    padding: 2.2rem 2.8rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: '⬡'; position: absolute; right: 2rem; top: 50%;
    transform: translateY(-50%); font-size: 10rem;
    color: rgba(56,189,248,.03); line-height: 1; pointer-events: none;
}
.hero-title {
    font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8 48%, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 .45rem;
}
.hero-sub { color: #3d5170; font-size: .88rem; margin: 0 0 1rem; }
.hbadge {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: .63rem; font-weight: 700; padding: 3px 10px;
    border-radius: 20px; margin: 2px 3px 2px 0;
}
.hb1{background:rgba(56,189,248,.1);color:#38bdf8;border:1px solid #0ea5e9;}
.hb2{background:rgba(129,140,248,.1);color:#818cf8;border:1px solid #6366f1;}
.hb3{background:rgba(244,114,182,.1);color:#f472b6;border:1px solid #ec4899;}
.hb4{background:rgba(52,211,153,.1);color:#34d399;border:1px solid #10b981;}
.hb5{background:rgba(251,191,36,.1);color:#fbbf24;border:1px solid #f59e0b;}
.hb6{background:rgba(248,113,113,.1);color:#f87171;border:1px solid #ef4444;}

/* ── pipeline cards ── */
.pc {
    background: #0b1524; border: 1px solid #111d33; border-radius: 14px;
    padding: 1.3rem 1.5rem; margin-bottom: .8rem; transition: border-color .18s;
    cursor: pointer;
}
.pc:hover  { border-color: #1e4976; }
.pc.active { border-color: #818cf8; background: #0f1c36; }
.pc-title  { font-family: 'JetBrains Mono', monospace; font-size: .84rem;
             font-weight: 700; color: #e2e8f0; margin: 0 0 .3rem; }
.pc-desc   { font-size: .76rem; color: #334766; line-height: 1.5; margin: 0; }
.pc-flow   { font-family: 'JetBrains Mono', monospace; font-size: .64rem;
             color: #1a3050; margin-top: .55rem; }

/* ── agent flow bar ── */
.agent-flow { display: flex; align-items: center; flex-wrap: wrap;
              gap: .3rem; padding: .75rem 0 .35rem; }
.an {
    font-family: 'JetBrains Mono', monospace; font-size: .69rem; font-weight: 700;
    padding: .38rem .8rem; border-radius: 8px; white-space: nowrap; transition: .15s;
}
.an-main   { background: #0b1e3e; color: #38bdf8; border: 1px solid #1a4776; }
.an-sub    { background: #18093a; color: #818cf8; border: 1px solid #3a2875; }
.an-active { box-shadow: 0 0 18px rgba(56,189,248,.4) !important; transform: scale(1.07); }
.an-done   { opacity: .4; }
.an-arr    { color: #1a3050; font-size: .85rem; }

/* ── log pane ── */
.logpane {
    background: #020509; border: 1px solid #111d33; border-radius: 12px;
    padding: .9rem 1.2rem; font-family: 'JetBrains Mono', monospace;
    font-size: .71rem; line-height: 1.9; min-height: 200px; max-height: 460px;
    overflow-y: auto; color: #1e3050;
}
.lh { color: #475569; font-weight: 700; }
.lm { color: #38bdf8; }
.ls0{ color: #818cf8; }
.ls1{ color: #f472b6; }
.ls2{ color: #34d399; }
.ls3{ color: #fb923c; }
.lt { color: #fbbf24; }
.lr { color: #64748b; }
.lf { color: #4ade80; font-weight: 700; }
.lw { color: #f87171; }

/* ── result box ── */
.res-wrap {
    background: #070f1e; border: 1px solid #132032; border-radius: 14px;
    padding: 1.6rem 1.9rem; margin-top: 1.2rem;
}
.res-hdr {
    font-family: 'JetBrains Mono', monospace; font-size: .71rem; color: #38bdf8;
    margin-bottom: .9rem; letter-spacing: .06em; text-transform: uppercase;
}
.res-body {
    font-size: .86rem; line-height: 1.85; color: #c8d5e8;
    white-space: pre-wrap; word-wrap: break-word;
}

/* ── sidebar ── */
.ss {
    font-family: 'JetBrains Mono', monospace; font-size: .67rem; color: #38bdf8;
    text-transform: uppercase; letter-spacing: .12em; margin: 1rem 0 .4rem;
}
.sp {
    background: #09111f; border: 1px solid #111d33; border-radius: 7px;
    padding: .4rem .68rem; margin-bottom: .3rem; font-size: .72rem; color: #2a4060;
}
.sp b { color: #3d5885; }
.gok  { color: #4ade80; font-family: 'JetBrains Mono', monospace; font-size: .73rem; }
.gof  { color: #f87171; font-family: 'JetBrains Mono', monospace; font-size: .73rem; }

/* ── custom builder ── */
.builder-card {
    background: #0a1424; border: 1px solid #111d33; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-bottom: .7rem;
}
.builder-card-title {
    font-family: 'JetBrains Mono', monospace; font-size: .78rem;
    font-weight: 700; color: #818cf8; margin: 0 0 .5rem;
}

/* ── tabs ── */
div[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important; font-size: .76rem !important;
}

/* ── buttons ── */
button[kind="primary"] {
    background: linear-gradient(90deg, #4338ca, #0ea5e9) !important;
    border: none !important; color: white !important;
    font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important;
}
button[kind="secondary"] {
    background: #0b1524 !important; border: 1px solid #1a3050 !important;
    color: #64748b !important; font-family: 'JetBrains Mono', monospace !important;
}

/* code blocks */
code { background: #0d1828 !important; color: #38bdf8 !important; border-radius: 4px; }
pre  { background: #060d18 !important; border: 1px solid #111d33 !important;
       border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECK  (show install instructions before crashing)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _check_deps():
    missing = []
    for pkg in ["deepagents", "openai", "langchain_openai", "langchain_core", "langchain"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("_", "-"))
    return missing

MISSING = _check_deps()
if MISSING:
    st.markdown("""
<div class="hero">
  <p class="hero-title">⬡ Deep Agents Explorer</p>
  <p class="hero-sub">Setup required — install dependencies below</p>
</div>
""", unsafe_allow_html=True)
    st.error(f"Missing packages: **{', '.join(MISSING)}**")
    st.code("pip install deepagents openai langchain-openai langchain", language="bash")
    st.info("After installing, refresh this page.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# REAL DEEPAGENTS IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
from deepagents import create_deep_agent          # ← THE REAL THING
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool as lc_tool

# NVIDIA NIM base URL
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD API KEY — Streamlit secrets take priority, sidebar input is fallback
# In Streamlit Cloud: Settings → Secrets → add:  NVIDIA_API_KEY = "nvapi-..."
# ─────────────────────────────────────────────────────────────────────────────
_secret_key = ""
try:
    _secret_key = st.secrets["NVIDIA_API_KEY"]
except Exception:
    pass  # no secrets file — user will enter key manually in sidebar

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defaults = dict(
    groq_key=_secret_key, groq_ok=bool(_secret_key),
    logs=[], run_history=[],
    active_pipeline=None,
    last_output="",
    custom_subagents=[],   # for custom builder tab
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN TOOLS  — pure Python, zero external APIs
# ─────────────────────────────────────────────────────────────────────────────
def make_domain_tools(api_key: str, model_str: str):
    """
    Custom tools passed to create_deep_agent(tools=[...]).
    deepagents will also auto-inject its built-in tools on top of these.
    """
    logs = st.session_state.logs

    @lc_tool
    def analyze_code(code: str) -> str:
        """Static-analyse Python/JS code: security, bugs, complexity. Returns issues + score."""
        logs.append(("lt", "🔧 analyze_code"))
        issues = []
        if re.search(r'f["\'].*?(SELECT|INSERT|UPDATE|DELETE)', code, re.I):
            issues.append("🔴 CRITICAL — SQL Injection: f-string inside SQL. Use parameterized queries.")
        if re.search(r'(password|secret|token|api_key)\s*=\s*["\'][^"\']{3,}', code, re.I):
            issues.append("🔴 CRITICAL — Hardcoded credential detected.")
        if re.search(r'\beval\s*\(|\bexec\s*\(', code):
            issues.append("🔴 HIGH — eval()/exec() allows arbitrary code execution.")
        if "md5" in code.lower() or "sha1(" in code.lower():
            issues.append("🟠 HIGH — Weak hash (MD5/SHA1). Use SHA-256 or bcrypt.")
        if re.search(r'except\s*:', code):
            issues.append("🟡 MEDIUM — Bare `except:` masks all errors including KeyboardInterrupt.")
        if re.search(r'http://', code):
            issues.append("🟡 MEDIUM — Plain HTTP URL found. Use HTTPS.")
        if not re.search(r'("""|\'\'\').*?("""|\'\'\')|\#\s\w', code, re.S):
            issues.append("🔵 LOW — No docstrings or inline comments found.")
        lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
        loc = len(lines)
        cc  = len(re.findall(r'\b(if|elif|for|while|try|and\s|or\s|case\s)\b', code))
        score = max(1, 10 - sum(3 if "🔴" in i else 2 if "🟠" in i else 1 for i in issues))
        body  = "\n".join(issues) if issues else "  ✅ No critical issues found."
        result = f"LOC: {loc}   Complexity: ~{cc}   Score: {score}/10\n\nIssues:\n{body}"
        logs.append(("lr", f"   ↳ score={score}/10, issues={len(issues)}"))
        return result

    @lc_tool
    def compute_stats(data_text: str) -> str:
        """Extract numbers from text and compute descriptive stats + trend analysis."""
        logs.append(("lt", "🔧 compute_stats"))
        nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',
                                              data_text.replace(",", "").replace("_", ""))]
        if not nums:
            return "No numeric values found in input."
        n   = len(nums); s = sum(nums); avg = s / n
        mn  = min(nums);  mx = max(nums)
        std = (sum((x - avg) ** 2 for x in nums) / n) ** .5
        xbar = (n - 1) / 2
        denom = sum((i - xbar) ** 2 for i in range(n)) or 1
        slope = sum((i - xbar) * (y - avg) for i, y in enumerate(nums)) / denom
        pct   = (nums[-1] - nums[0]) / abs(nums[0]) * 100 if nums[0] else 0
        median = sorted(nums)[n // 2]
        result = (
            f"N={n}  Sum={s:,.2f}  Mean={avg:,.2f}  Median={median:,.2f}\n"
            f"Min={mn:,.2f}  Max={mx:,.2f}  StdDev={std:,.2f}\n"
            f"Linear slope: {slope:+.3f}/period\n"
            f"Total change: {pct:+.1f}%  ({nums[0]:,.0f} → {nums[-1]:,.0f})\n"
            f"Trend: {'📈 Upward' if slope > 0.5 else '📉 Downward' if slope < -0.5 else '➡️ Flat'}"
        )
        logs.append(("lr", f"   ↳ mean={avg:.1f}, slope={slope:+.2f}, Δ={pct:+.1f}%"))
        return result

    @lc_tool
    def classify_error(error_text: str) -> str:
        """Classify Python/JS exception: type, severity, root cause, fix direction."""
        logs.append(("lt", "🔧 classify_error"))
        catalog = {
            "AttributeError":     ("Medium",   "None/wrong-type attribute access",       "Add None-check before calling .method()"),
            "TypeError":          ("Medium",   "Wrong arg type or missing required arg",  "Verify argument types; check function signature"),
            "ValueError":         ("Medium",   "Correct type, semantically invalid value","Validate value range/format before use"),
            "KeyError":           ("Low",      "Dictionary key does not exist",           "Use dict.get(key) or check key in dict first"),
            "IndexError":         ("Low",      "Sequence index out of range",             "Check len() before indexing"),
            "ImportError":        ("High",     "Module not installed or wrong path",      "pip install <package> or fix import path"),
            "ModuleNotFoundError":("High",     "Package not installed",                   "pip install <package>"),
            "FileNotFoundError":  ("Medium",   "File path does not exist",               "Use os.path.exists() before opening"),
            "PermissionError":    ("High",     "OS-level permission denied",              "Check file/directory permissions"),
            "ConnectionError":    ("High",     "Network unreachable or refused",          "Check host, port, firewall; add retry logic"),
            "TimeoutError":       ("Medium",   "Operation exceeded time limit",           "Increase timeout or add retry with backoff"),
            "RecursionError":     ("High",     "Max recursion depth exceeded",            "Add/fix base case; consider iterative approach"),
            "MemoryError":        ("Critical", "Process out of memory",                  "Reduce batch size; use generators/streaming"),
            "ZeroDivisionError":  ("Low",      "Division by zero",                       "Guard: `if divisor != 0:`"),
            "NameError":          ("Medium",   "Variable used before assignment",         "Ensure variable is assigned before first use"),
            "OverflowError":      ("Medium",   "Numeric value too large for type",        "Use Python's arbitrary precision int or decimal"),
            "UnicodeDecodeError": ("Medium",   "Bytes not decodable with given encoding", "Specify encoding='utf-8' or use errors='ignore'"),
        }
        for etype, (sev, cause, fix) in catalog.items():
            if etype in error_text:
                icon = {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🔵"}[sev]
                result = (f"Type:     {etype}\n"
                          f"Severity: {icon} {sev}\n"
                          f"Cause:    {cause}\n"
                          f"Fix:      {fix}")
                logs.append(("lr", f"   ↳ {etype} ({sev})"))
                return result
        return ("Unknown error type — not in catalog.\n"
                "Tip: share the full traceback including file name and line number.")

    @lc_tool
    def estimate_effort(task_description: str) -> str:
        """Estimate Fibonacci story points, dev hours, sprint fit, and risk level for a task."""
        logs.append(("lt", f"🔧 estimate_effort: {task_description[:55]}"))
        d   = task_description.lower()
        hi  = ["authentication","oauth","saml","security","database","migration",
               "redis","cache","kubernetes","k8s","encryption","ci/cd","pipeline",
               "ml","machine learning","ai","distributed","microservice","real-time"]
        md  = ["api","rest","graphql","endpoint","crud","docker","deploy","webhook",
               "integration","search","notification","email","payment","stripe","export"]
        lo  = ["readme","docs","documentation","lint","rename","refactor","config",
               "env","comment","typo","cleanup","logging","monitoring"]
        pts = (13 if any(k in d for k in hi) else
               5  if any(k in d for k in md) else 2)
        risk = "🔴 High" if pts >= 13 else ("🟡 Medium" if pts >= 5 else "🟢 Low")
        result = (f"Story points:  {pts} (Fibonacci)\n"
                  f"Estimate:      ~{pts * 3}–{pts * 5} hours\n"
                  f"Risk:          {risk}\n"
                  f"Sprint fit:    {'needs breakdown (>1 sprint)' if pts > 13 else 'fits in one sprint'}")
        logs.append(("lr", f"   ↳ {pts} pts, {risk}"))
        return result

    @lc_tool
    def knowledge_lookup(query: str) -> str:
        """Answer a factual/technical question using the LLM as an in-context knowledge base."""
        logs.append(("lt", f"🔧 knowledge_lookup: {query[:65]}"))
        try:
            from langchain_openai import ChatOpenAI
            m = ChatOpenAI(
                base_url=NVIDIA_BASE_URL,
                api_key=api_key,
                model=model_str,
                temperature=0.1,
                max_tokens=400,
            )
            r = m.invoke([HumanMessage(content=(
                f"Answer this technical question with concrete, accurate facts. "
                f"Be specific (include versions, numbers, comparisons where relevant). "
                f"4-7 sentences max.\n\nQuestion: {query}"
            ))])
            ans = r.content
            logs.append(("lr", f"   ↳ {ans[:85]}…"))
            return ans
        except Exception as exc:
            return f"knowledge_lookup error: {exc}"

    @lc_tool
    def summarize_text(text: str, style: str = "bullets") -> str:
        """Summarise long text. style: 'bullets' | 'paragraph' | 'table'."""
        logs.append(("lt", f"🔧 summarize_text ({len(text)} chars, style={style})"))
        style_inst = {
            "bullets":   "Return 6-8 concise bullet points (• prefix).",
            "paragraph": "Return a 3-4 sentence executive-summary paragraph.",
            "table":     "Return a markdown table with columns: Topic | Key Point.",
        }.get(style, "Return 6 bullet points.")
        try:
            from langchain_openai import ChatOpenAI
            m = ChatOpenAI(
                base_url=NVIDIA_BASE_URL,
                api_key=api_key,
                model=model_str,
                temperature=0.05,
                max_tokens=350,
            )
            r = m.invoke([HumanMessage(content=f"{style_inst}\n\nContent:\n{text[:3500]}")])
            logs.append(("lr", "   ↳ summary done"))
            return r.content
        except Exception as exc:
            return f"summarize_text error: {exc}"

    @lc_tool
    def generate_report(title: str, sections: str) -> str:
        """
        Format a professional markdown report.
        sections: JSON string mapping section_title → content,
                  e.g. '{"Executive Summary": "...", "Findings": "..."}'
        """
        logs.append(("lt", f"🔧 generate_report: {title}"))
        ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        try:
            sec_dict = json.loads(sections) if sections.strip().startswith("{") else {"Content": sections}
        except Exception:
            sec_dict = {"Content": sections}
        lines = [f"# {title}", f"*Generated: {ts}*", "---", ""]
        for heading, body in sec_dict.items():
            lines += [f"## {heading}", "", str(body), ""]
        result = "\n".join(lines)
        logs.append(("lr", f"   ↳ report built ({len(result)} chars)"))
        return result

    @lc_tool
    def compare_options(option_a: str, option_b: str, criteria: str) -> str:
        """
        Compare two technology/approach options across given criteria.
        criteria: comma-separated list, e.g. "performance,cost,scalability,ease of use"
        """
        logs.append(("lt", f"🔧 compare_options: {option_a} vs {option_b}"))
        try:
            from langchain_openai import ChatOpenAI
            m = ChatOpenAI(
                base_url=NVIDIA_BASE_URL,
                api_key=api_key,
                model=model_str,
                temperature=0.2,
                max_tokens=450,
            )
            r = m.invoke([HumanMessage(content=(
                f"Compare '{option_a}' vs '{option_b}' across these criteria: {criteria}.\n"
                f"Format as a markdown table with columns: Criterion | {option_a} | {option_b} | Winner.\n"
                f"End with a 2-sentence recommendation."
            ))])
            logs.append(("lr", "   ↳ comparison done"))
            return r.content
        except Exception as exc:
            return f"compare_options error: {exc}"

    @lc_tool
    def extract_action_items(text: str) -> str:
        """Extract all action items, tasks, TODOs, and next steps from a body of text."""
        logs.append(("lt", f"🔧 extract_action_items ({len(text)} chars)"))
        try:
            from langchain_openai import ChatOpenAI
            m = ChatOpenAI(
                base_url=NVIDIA_BASE_URL,
                api_key=api_key,
                model=model_str,
                temperature=0.05,
                max_tokens=300,
            )
            r = m.invoke([HumanMessage(content=(
                f"Extract all action items, tasks, TODOs, and next steps from the following text. "
                f"Format each as: [ ] Owner (if known): Action description\n\nText:\n{text[:3000]}"
            ))])
            logs.append(("lr", "   ↳ action items extracted"))
            return r.content
        except Exception as exc:
            return f"extract_action_items error: {exc}"

    return [
        analyze_code, compute_stats, classify_error, estimate_effort,
        knowledge_lookup, summarize_text, generate_report,
        compare_options, extract_action_items,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE REGISTRY  (prompts kept short → stay within Groq free 12k TPM limit)
# ─────────────────────────────────────────────────────────────────────────────
PIPELINES = {

    "deep_research": {
        "icon":  "🔬",
        "title": "Deep Research Pipeline",
        "desc":  "Main agent plans with write_todos then delegates to data-analyst, critic, and writer sub-agents.",
        "use_case": "Research any topic, compare technologies, explain concepts",
        "example": "Compare LangGraph vs traditional LangChain AgentExecutor: key differences in architecture, memory, and streaming.",
        "system_prompt": (
            "You are the Lead Research Agent.\n"
            "1. write_todos with 4 steps.\n"
            "2. task(agent_name='data-analyst', task=<specific question>).\n"
            "3. write_file('notes.md', <findings>).\n"
            "4. task(agent_name='writer-agent', task=<notes>) for final report.\n"
            "Be concise. Output the writer-agent report as your final answer."
        ),
        "subagents": [
            {
                "name": "data-analyst",
                "description": "Gathers facts, benchmarks, and data points",
                "system_prompt": "You are a fact-gatherer. Use knowledge_lookup to get concrete facts and numbers. Output bullet points.",
            },
            {
                "name": "critic-agent",
                "description": "Finds weaknesses and counterarguments in findings",
                "system_prompt": "You are a critic. Find 3-5 key limitations or tradeoffs in what was presented. Be concise.",
            },
            {
                "name": "writer-agent",
                "description": "Writes the final polished report from all findings",
                "system_prompt": "You are a technical writer. Write a concise structured report: Summary, Findings, Tradeoffs, Recommendation. Use markdown.",
            },
        ],
        "flow": "main → write_todos → task(data-analyst) → task(critic-agent) → task(writer-agent)",
    },

    "code_audit": {
        "icon":  "🛡️",
        "title": "Multi-Agent Code Audit",
        "desc":  "Static-analyzer finds vulnerabilities, fix-writer produces secure replacement code.",
        "use_case": "Security audit, bug detection, code quality review",
        "example": (
            "Audit this login function:\n\n"
            "def login(username, password, db):\n"
            "    query = f\"SELECT * FROM users WHERE user='{username}' AND pass='{password}'\"\n"
            "    user = db.execute(query).fetchone()\n"
            "    if user: session['auth'] = str(user[0])\n"
            "    return user is not None"
        ),
        "system_prompt": (
            "You are the Lead Security Auditor.\n"
            "1. write_todos: [check injection, check auth, check crypto, write report].\n"
            "2. task(agent_name='static-analyzer', task=<code>).\n"
            "3. write_file('findings.md', <results>).\n"
            "4. task(agent_name='fix-writer', task=<findings>).\n"
            "5. Output final audit report."
        ),
        "subagents": [
            {
                "name": "static-analyzer",
                "description": "Finds security vulnerabilities, bugs, and anti-patterns in code",
                "system_prompt": "You are a security analyzer. Use analyze_code tool. List each issue: Severity | Location | Description. Be brief.",
            },
            {
                "name": "fix-writer",
                "description": "Writes secure replacement code with explanations",
                "system_prompt": "You are a security engineer. For each finding write: the fixed code snippet and a one-line explanation. Use markdown code blocks.",
            },
        ],
        "flow": "main → write_todos → task(static-analyzer) → write_file → task(fix-writer) → report",
    },

    "data_intelligence": {
        "icon":  "📊",
        "title": "Data Intelligence Pipeline",
        "desc":  "Stats agent crunches numbers, insights agent interprets them, report-writer delivers the final report.",
        "use_case": "Business analytics, KPI analysis, trend detection",
        "example": "Analyse SaaS metrics: Revenue($): Jan=8200,Feb=9100,Mar=11500,Apr=10200,May=14800,Jun=19400. Churn(%): 8.2,7.9,7.1,8.8,6.4,5.2",
        "system_prompt": (
            "You are the Lead Data Analyst.\n"
            "1. write_todos: [compute stats, find insights, write report].\n"
            "2. task(agent_name='stats-agent', task=<raw data>).\n"
            "3. write_file('stats.md', <results>).\n"
            "4. task(agent_name='report-writer', task=<stats>).\n"
            "Output the final report."
        ),
        "subagents": [
            {
                "name": "stats-agent",
                "description": "Computes statistics, trends, and anomalies from numerical data",
                "system_prompt": "You are a quant analyst. Use compute_stats on the data. Report: mean, trend slope, anomalies, % change. Use a table.",
            },
            {
                "name": "insights-agent",
                "description": "Translates statistics into business insights and recommendations",
                "system_prompt": "You are a BI analyst. Given stats, explain what they mean for the business. Give 3 concrete recommendations.",
            },
            {
                "name": "report-writer",
                "description": "Writes the final executive data intelligence report",
                "system_prompt": "You are a data storyteller. Write a report: Executive Summary, Key Metrics table, Top 3 Insights, Recommendations. Use markdown.",
            },
        ],
        "flow": "main → write_todos → task(stats-agent) → write_file → task(report-writer)",
    },

    "project_architect": {
        "icon":  "🗺️",
        "title": "Project Architecture Planner",
        "desc":  "Tech-researcher picks the stack, effort-estimator builds the backlog, risk-analyst flags blockers.",
        "use_case": "System design, sprint planning, architecture decisions",
        "example": "Design architecture for a real-time chat app: WebSocket messaging, user auth, message history, and a React frontend.",
        "system_prompt": (
            "You are the Lead Architect.\n"
            "1. write_todos: [research stack, estimate effort, assess risks, write plan].\n"
            "2. task(agent_name='tech-researcher', task=<requirements>).\n"
            "3. task(agent_name='effort-estimator', task=<components>).\n"
            "4. generate_report(title='Architecture Plan', sections=<json>).\n"
            "Output the final architecture plan."
        ),
        "subagents": [
            {
                "name": "tech-researcher",
                "description": "Recommends best-fit technologies for each component",
                "system_prompt": "You are a senior architect. Use knowledge_lookup to recommend technologies. For each component: Chosen Tech | Reason | Risk. Be brief.",
            },
            {
                "name": "effort-estimator",
                "description": "Estimates story points and sprint timeline for each component",
                "system_prompt": "You are an engineering manager. Use estimate_effort per component. Output a backlog table: Task | Points | Priority.",
            },
            {
                "name": "risk-analyst",
                "description": "Identifies technical risks and mitigation strategies",
                "system_prompt": "You are a risk analyst. List the top 5 risks: Risk | Probability | Impact | Mitigation. Use a markdown table.",
            },
        ],
        "flow": "main → write_todos → task(tech-researcher) → task(effort-estimator) → generate_report",
    },

    "debug_pipeline": {
        "icon":  "🐛",
        "title": "Deep Debug & Fix Pipeline",
        "desc":  "Root-cause analyst diagnoses the bug, fix-writer produces the corrected code with tests.",
        "use_case": "Error diagnosis, bug fixing, regression prevention",
        "example": (
            "Debug this error (occurs only on 1st of each month):\n"
            "ZeroDivisionError in billing/utils.py line 67:\n"
            "  daily_rate = plan_price / days_in_month(start_date.month)\n"
            "days_in_month() uses a dict — December key was removed in last refactor."
        ),
        "system_prompt": (
            "You are the Lead Debug Agent.\n"
            "1. write_todos: [classify, root-cause, fix, test].\n"
            "2. Use classify_error on the traceback.\n"
            "3. task(agent_name='root-cause-analyst', task=<error+context>).\n"
            "4. write_file('root_cause.md', <analysis>).\n"
            "5. task(agent_name='fix-writer', task=<root cause>).\n"
            "Output: Root Cause + Fix + Tests."
        ),
        "subagents": [
            {
                "name": "root-cause-analyst",
                "description": "Finds the exact root cause and trigger conditions of the error",
                "system_prompt": "You are a debug specialist. Use classify_error. Identify: root cause (1 sentence), trigger conditions, affected code path. Be precise.",
            },
            {
                "name": "fix-writer",
                "description": "Writes the minimal correct fix and unit tests",
                "system_prompt": "You are a senior engineer. Write: (1) minimal fix with comments, (2) 3 pytest tests: happy path, edge case, regression. Use code blocks.",
            },
        ],
        "flow": "main → write_todos → classify_error → task(root-cause-analyst) → task(fix-writer)",
    },

    "doc_generator": {
        "icon":  "📚",
        "title": "Technical Doc Generator",
        "desc":  "Context-gatherer researches the topic, doc-writer drafts the docs, quality-reviewer polishes the final output.",
        "use_case": "README, API docs, onboarding guides, runbooks",
        "example": "Write a README for 'FlowGraph' — a Python library for building LangGraph agent pipelines with a declarative DSL, built-in retry, and streaming support.",
        "system_prompt": (
            "You are the Lead Documentation Agent.\n"
            "1. write_todos: [gather context, draft docs, review].\n"
            "2. task(agent_name='context-gatherer', task=<project description>).\n"
            "3. write_file('context.md', <findings>).\n"
            "4. task(agent_name='doc-writer', task=<context>).\n"
            "5. Output the final documentation."
        ),
        "subagents": [
            {
                "name": "context-gatherer",
                "description": "Researches similar projects and prepares context and examples",
                "system_prompt": "You are a researcher. Use knowledge_lookup for similar libraries. Produce: feature list, 2 code examples, 3 FAQ answers.",
            },
            {
                "name": "doc-writer",
                "description": "Writes complete technical documentation with code examples",
                "system_prompt": "You are a technical writer. Write docs with sections: Intro, Install, Quick Start, API Reference, Examples. Use markdown and code blocks.",
            },
            {
                "name": "quality-reviewer",
                "description": "Reviews docs for completeness and clarity, outputs final improved version",
                "system_prompt": "You are a DX reviewer. Fix any gaps, unclear wording, or missing examples in the draft. Output the complete improved final version.",
            },
        ],
        "flow": "main → write_todos → task(context-gatherer) → task(doc-writer) → task(quality-reviewer)",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# BUILD AGENT  (real deepagents.create_deep_agent)
# ─────────────────────────────────────────────────────────────────────────────
def build_agent(api_key: str, model_str: str, system_prompt: str, subagents_cfg: list):
    """
    Creates a real deepagents agent using the correct API:

    create_deep_agent(
        model        = ChatGroq(...),        # LangChain ChatModel object
        tools        = [...],                # custom @tool functions
        system_prompt = "...",              # ← CORRECT param (NOT 'instructions')
        subagents    = [                     # list of dicts
            {"name": "...", "description": "...", "system_prompt": "...",
             "model": "groq:...", "tools": [...]}  # model as string works too
        ],
    )
    Returns a LangGraph CompiledStateGraph.
    """
    # Set env var so deepagents' internal init_chat_model calls pick it up
    os.environ["GROQ_API_KEY"] = api_key

    # Token budget: 8b models = generous output, 70b = tight (free tier 12k TPM)
    is_large = any(x in model_str for x in ["70b", "mixtral"])
    main_max  = 500 if is_large else 800
    sub_max   = 400 if is_large else 600

    # Use ChatOpenAI pointed at NVIDIA NIM — OpenAI-compatible, no TPM cap
    from langchain_openai import ChatOpenAI
    main_model = ChatOpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
        model=model_str,
        temperature=0.45,
        max_tokens=main_max,
    )
    sub_model = ChatOpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
        model=model_str,
        temperature=0.35,
        max_tokens=sub_max,
    )

    domain_tools = make_domain_tools(api_key, model_str)

    # Build sub-agent dicts in the real deepagents format
    # (same keys as official docs: name, description, system_prompt, model, tools)
    subagents = []
    for sa in subagents_cfg:
        subagents.append({
            "name":          sa["name"],
            "description":   sa["description"],
            "system_prompt": sa["system_prompt"],   # ← correct key
            "model":         sub_model,              # ChatGroq object
            "tools":         domain_tools,
        })

    agent = create_deep_agent(
        model         = main_model,
        tools         = domain_tools,
        system_prompt = system_prompt,    # ← FIXED: was 'instructions=', now 'system_prompt='
        subagents     = subagents,
    )
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING RUNNER  (handles deepagents' LangGraph stream output)
# ─────────────────────────────────────────────────────────────────────────────
def run_agent_stream(agent, task_text: str, pipeline_cfg: dict):
    """Stream the agent and update st.session_state.logs live."""
    logs = st.session_state.logs
    log_ph = st.empty()
    res_ph = st.empty()

    sa_names = [sa["name"] for sa in pipeline_cfg.get("subagents", [])]

    def _render():
        css_map = {
            "lh":"lh","lm":"lm","lt":"lt","lr":"lr","lf":"lf","lw":"lw",
            "ls0":"ls0","ls1":"ls1","ls2":"ls2","ls3":"ls3",
        }
        lines = []
        for kind, text in logs[-40:]:
            c = css_map.get(kind, "lr")
            escaped = _html.escape(str(text))
            lines.append(f'<span class="{c}">{escaped}</span>')
        log_ph.markdown(
            f'<div class="logpane">{"<br>".join(lines)}</div>',
            unsafe_allow_html=True,
        )

    final_output = ""

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=task_text)]},
        stream_mode="updates",
    ):
        for node_name, node_data in chunk.items():
            if not isinstance(node_data, dict):
                continue
            msgs = node_data.get("messages", [])
            for msg in msgs:
                # Tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tname  = tc.get("name", "?")
                        targs  = tc.get("args", {})
                        if tname == "task":
                            # Sub-agent delegation via deepagents built-in `task` tool
                            sa_name = (targs.get("agent_name") or
                                       targs.get("name") or "?")
                            idx  = next((i for i, s in enumerate(sa_names)
                                         if s == sa_name), 0)
                            ckey = f"ls{idx % 4}"
                            logs.append((ckey, f"🤖 task('{sa_name}') ← spawning sub-agent"))
                        elif tname == "write_todos":
                            tasks = targs.get("todos", targs.get("tasks", []))
                            logs.append(("lm", f"📋 write_todos → {len(tasks)} steps planned"))
                        elif tname in ("write_file", "read_file", "edit_file"):
                            fname = targs.get("filename", targs.get("path", "?"))
                            logs.append(("lm", f"💾 {tname}('{fname}')"))
                        else:
                            arg_preview = str(targs)[:55].replace("\n", " ")
                            logs.append(("lt", f"🔧 {tname}({arg_preview})"))

                # Tool results
                elif isinstance(msg, ToolMessage):
                    content_preview = str(msg.content)[:100].replace("\n", " ")
                    logs.append(("lr", f"   ↳ {content_preview}"))

                # Agent text output
                elif isinstance(msg, AIMessage) and msg.content:
                    if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        content = str(msg.content).strip()
                        if content:
                            final_output = content
                            preview = content[:80].replace("\n", " ")
                            logs.append(("lm", f"💬 agent: {preview}…"))

        _render()

    return final_output, res_ph


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:JetBrains Mono,monospace;font-size:1rem;font-weight:700;'
        'background:linear-gradient(90deg,#38bdf8,#818cf8,#f472b6);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'margin-bottom:1.5rem;letter-spacing:.04em;">⬡ DEEP AGENTS</div>',
        unsafe_allow_html=True,
    )

    # ── API Key ──
    st.markdown('<div class="ss">🔑 NVIDIA API Key</div>', unsafe_allow_html=True)
    groq_input = st.text_input(
        "nvidia_key_input", type="password", placeholder="nvapi-…",
        label_visibility="collapsed",
        value=st.session_state.groq_key,
        key="sidebar_groq_key",
    )
    if groq_input != st.session_state.groq_key:
        st.session_state.groq_key = groq_input
        st.session_state.groq_ok  = False

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✓ Verify", use_container_width=True, key="verify_btn"):
            if st.session_state.groq_key:
                with st.spinner("Verifying…"):
                    try:
                        from openai import OpenAI
                        _client = OpenAI(
                            base_url=NVIDIA_BASE_URL,
                            api_key=st.session_state.groq_key,
                        )
                        _client.chat.completions.create(
                            model="nvidia/usdcode-llama-3.1-70b-instruct",
                            messages=[{"role": "user", "content": "hi"}],
                            max_tokens=5,
                        )
                        st.session_state.groq_ok = True
                        st.rerun()
                    except Exception as e:
                        st.session_state.groq_ok = False
                        st.error(f"Auth failed: {str(e)[:120]}")
    with c2:
        if st.button("✕ Clear", use_container_width=True, key="clear_key_btn"):
            st.session_state.groq_key = ""
            st.session_state.groq_ok  = False
            st.rerun()

    if st.session_state.groq_ok:
        st.markdown('<span class="gok">✅ Connected to NVIDIA NIM</span>', unsafe_allow_html=True)
    elif st.session_state.groq_key:
        st.markdown('<span class="gof">⚠️ Key entered — click Verify</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="gof">⬡ Enter key above</span>', unsafe_allow_html=True)

    st.caption("Get your free key at build.nvidia.com — no credit card needed")

    st.divider()

    # ── Model ──
    st.markdown('<div class="ss">🤖 Model</div>', unsafe_allow_html=True)
    MODEL = st.selectbox(
        "model_select", label_visibility="collapsed",
        options=[
            "nvidia/usdcode-llama-3.1-70b-instruct",   # your primary model
            "nvidia/llama-3.1-8b-instruct",
            "nvidia/llama-3.3-70b-instruct",
            "meta/llama-3.1-70b-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1",
        ],
        index=0,   # usdcode-70b matches your API key setup
        key="model_selector",
    )
    model_notes = {
        "nvidia/usdcode-llama-3.1-70b-instruct":  "✅ Your primary model — code + reasoning optimised",
        "nvidia/llama-3.1-8b-instruct":           "✅ Fast & lightweight — use if hitting quota",
        "nvidia/llama-3.3-70b-instruct":           "✅ 70b — latest Llama 3.3 on NVIDIA NIM",
        "meta/llama-3.1-70b-instruct":             "✅ 70b — official Meta model",
        "mistralai/mixtral-8x7b-instruct-v0.1":    "✅ Mixtral MoE — fast & capable",
    }
    st.caption(model_notes.get(MODEL, ""))

    st.divider()

    # ── Run history ──
    st.markdown('<div class="ss">📜 Run History</div>', unsafe_allow_html=True)
    if st.session_state.run_history:
        for h in reversed(st.session_state.run_history[-6:]):
            st.markdown(
                f'<div class="sp"><b>{h["icon"]} {h["title"]}</b><br>'
                f'{h["ts"]} · {h["model"]}</div>',
                unsafe_allow_html=True,
            )
        if st.button("🗑 Clear history", use_container_width=True, key="clear_hist"):
            st.session_state.run_history = []
            st.rerun()
    else:
        st.markdown('<div class="sp" style="color:#111d33;">No runs yet</div>',
                    unsafe_allow_html=True)

    st.divider()

    # ── Info box ──
    st.markdown('<div class="ss">ℹ️ deepagents API</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="sp" style="line-height:1.7;">
<b>from deepagents import</b><br>
&nbsp;&nbsp;create_deep_agent<br><br>
<b>create_deep_agent(</b><br>
&nbsp;&nbsp;model=...,<br>
&nbsp;&nbsp;tools=[...],<br>
&nbsp;&nbsp;instructions="...",<br>
&nbsp;&nbsp;subagents=[...]<br>
<b>)</b><br><br>
<b>Built-in tools:</b><br>
&nbsp;&nbsp;write_todos<br>
&nbsp;&nbsp;read/write/edit_file<br>
&nbsp;&nbsp;ls · glob · grep<br>
&nbsp;&nbsp;<b>task</b> → sub-agents<br>
&nbsp;&nbsp;compact_conversation
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">⬡ Deep Agents Explorer</p>
  <p class="hero-sub">
    The real <code>deepagents</code> library by LangChain · <code>create_deep_agent(subagents=[...])</code> ·
    Built-in write_todos / filesystem / task tool · Powered by NVIDIA NIM (no TPM limits!)
  </p>
  <div>
    <span class="hbadge hb1">create_deep_agent</span>
    <span class="hbadge hb2">subagents=[ ]</span>
    <span class="hbadge hb3">write_todos</span>
    <span class="hbadge hb4">task tool</span>
    <span class="hbadge hb5">virtual filesystem</span>
    <span class="hbadge hb6">LangGraph streaming</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_pipelines, tab_custom, tab_freeform, tab_docs = st.tabs([
    "🧩 Pipelines",
    "🛠️ Custom Builder",
    "💬 Free-Form",
    "📖 Framework Docs",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINES
# ════════════════════════════════════════════════════════════════════════════
with tab_pipelines:
    st.markdown("### Pre-built Multi-Agent Pipelines")
    st.caption("Each pipeline uses `create_deep_agent(subagents=[...])` with 2-3 specialist sub-agents.")

    # Card grid
    cols = st.columns(3)
    for idx, (pk, pv) in enumerate(PIPELINES.items()):
        with cols[idx % 3]:
            active_cls = "active" if st.session_state.active_pipeline == pk else ""
            sa_str = " · ".join(
                f'<span style="color:#818cf8">{sa["name"]}</span>'
                for sa in pv["subagents"]
            )
            st.markdown(f"""
<div class="pc {active_cls}">
  <p class="pc-title">{pv['icon']} {pv['title']}</p>
  <p class="pc-desc">{pv['desc']}</p>
  <p class="pc-flow">sub-agents: {sa_str}</p>
</div>""", unsafe_allow_html=True)
            if st.button("Select", key=f"sel_{pk}", use_container_width=True):
                st.session_state.active_pipeline = pk
                st.session_state.logs = []
                st.session_state.last_output = ""
                st.rerun()

    # ── Run panel ──
    if st.session_state.active_pipeline:
        pk = st.session_state.active_pipeline
        pv = PIPELINES[pk]

        st.markdown("---")
        st.markdown(f"#### {pv['icon']} {pv['title']}")
        st.caption(f"💡 {pv['use_case']}")

        # Agent flow visual
        sa_nodes = " ".join(
            f'<span class="an an-sub">🤖 {sa["name"]}</span>'
            f'<span class="an-arr">→</span>'
            for sa in pv["subagents"]
        )
        st.markdown(f"""
<div class="agent-flow">
  <span class="an an-main">⬡ main-agent</span>
  <span class="an-arr">→</span>
  {sa_nodes}
  <span class="an" style="background:#0a1424;color:#1a3050;border:1px solid #111d33;">END</span>
</div>
<p style="font-family:JetBrains Mono,monospace;font-size:.63rem;color:#1a3050;margin:.1rem 0 .9rem;">
  {pv['flow']}
</p>
""", unsafe_allow_html=True)

        task_input = st.text_area(
            "Task input",
            value=pv["example"],
            height=155,
            key=f"task_{pk}",
            label_visibility="visible",
        )

        col_run, col_rst, col_dl = st.columns([5, 1, 1])
        with col_run:
            run_btn = st.button(
                "🚀  Run Deep Agents Pipeline",
                type="primary",
                use_container_width=True,
                key=f"run_{pk}",
            )
        with col_rst:
            if st.button("↺", use_container_width=True, key=f"rst_{pk}", help="Reset logs"):
                st.session_state.logs = []
                st.session_state.last_output = ""
                st.rerun()
        with col_dl:
            if st.session_state.last_output:
                st.download_button(
                    "⬇",
                    data=st.session_state.last_output,
                    file_name=f"deep_agent_output_{pk}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"dl_{pk}",
                    help="Download output as .md",
                )

        # ── EXECUTION ──
        if run_btn:
            if not st.session_state.groq_key:
                st.error("🔑 Enter your NVIDIA API key in the sidebar.")
                st.stop()
            st.session_state.logs = []
            st.session_state.last_output = ""

            logs = st.session_state.logs
            logs.append(("lh", "─" * 54))
            logs.append(("lh", f"  {pv['icon']} {pv['title']}"))
            logs.append(("lh", f"  model:      nvidia:{MODEL}"))
            logs.append(("lh", f"  sub-agents: {', '.join(sa['name'] for sa in pv['subagents'])}"))
            logs.append(("lh", f"  library:    deepagents.create_deep_agent"))
            logs.append(("lh", "─" * 54))

            with st.spinner("Building agent graph…"):
                try:
                    agent = build_agent(
                        st.session_state.groq_key, MODEL,
                        pv["system_prompt"], pv["subagents"],
                    )
                except Exception as e:
                    st.error(f"Build error: {e}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())
                    st.stop()

            logs.append(("lm", "✅ create_deep_agent compiled → LangGraph CompiledStateGraph"))
            logs.append(("lm", f"⬡ invoking main-agent: {task_input[:75]}…"))

            try:
                final_output, res_ph = run_agent_stream(agent, task_input, pv)
                logs.append(("lf", ""))
                logs.append(("lf", f"✅ Pipeline complete · {len(pv['subagents'])} sub-agents · deepagents built-in tools used"))

                # Re-render logs final state
                css_map = {"lh":"lh","lm":"lm","lt":"lt","lr":"lr","lf":"lf","lw":"lw",
                           "ls0":"ls0","ls1":"ls1","ls2":"ls2","ls3":"ls3"}
                lines = [f'<span class="{css_map.get(k,"lr")}">{_html.escape(str(t))}</span>'
                         for k, t in logs[-40:]]
                st.markdown(
                    f'<div class="logpane">{"<br>".join(lines)}</div>',
                    unsafe_allow_html=True,
                )

                if final_output:
                    st.session_state.last_output = final_output
                    res_ph.markdown(f"""
<div class="res-wrap">
  <div class="res-hdr">✍️ final output — {pv['title']}</div>
  <div class="res-body">{_html.escape(final_output)}</div>
</div>""", unsafe_allow_html=True)
                    st.download_button(
                        "⬇️ Download output as Markdown",
                        data=final_output,
                        file_name=f"deep_agent_{pk}_{datetime.now().strftime('%H%M%S')}.md",
                        mime="text/markdown",
                        key=f"dl_bottom_{pk}",
                    )

                st.session_state.run_history.append({
                    "icon":  pv["icon"],
                    "title": pv["title"],
                    "ts":    datetime.now().strftime("%H:%M:%S"),
                    "model": MODEL,
                })

                with st.expander("📋 Run details", expanded=False):
                    sa_lines = "\n".join(
                        f'        {{"name": "{sa["name"]}", "description": "{sa["description"][:55]}..."}},'
                        for sa in pv["subagents"]
                    )
                    code_snippet = (
                        "from deepagents import create_deep_agent\n"
                        "from langchain_openai import ChatOpenAI\n\n"
                        f'model = ChatOpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key, model=MODEL)\n'
                        "agent = create_deep_agent(\n"
                        "    model        = model,\n"
                        "    tools        = [analyze_code, compute_stats, ...],\n"
                        f'    instructions = """... {pv["title"]} instructions ...""",\n'
                        "    subagents    = [\n"
                        f"{sa_lines}\n"
                        "    ],\n"
                        ")\n"
                        'for chunk in agent.stream({"messages": [HumanMessage(content=task)]}):\n'
                        '    ...  # LangGraph stream_mode="updates"'
                    )
                    st.code(code_snippet, language="python")

            except Exception as e:
                err_str = str(e)
                if "413" in err_str or "Request too large" in err_str or "tokens per minute" in err_str.lower():
                    st.error("🔴 **Token limit exceeded (Error 413)**")
                    st.warning(
                        "**Fix:** Switch to **nvidia/llama-3.1-8b-instruct** in the sidebar — "
                        "the 8b model uses fewer tokens per call.\n\n"
                        "70b models send ~13k tokens with multi-agent prompts. "
                        "Check your NVIDIA NIM quota at build.nvidia.com."
                    )
                else:
                    st.error(f"❌ Runtime error: {e}")
                with st.expander("Full traceback"):
                    st.code(traceback.format_exc())

        elif st.session_state.logs:
            # Show previous logs when re-rendering
            css_map = {"lh":"lh","lm":"lm","lt":"lt","lr":"lr","lf":"lf","lw":"lw",
                       "ls0":"ls0","ls1":"ls1","ls2":"ls2","ls3":"ls3"}
            lines = [f'<span class="{css_map.get(k,"lr")}">{_html.escape(str(t))}</span>'
                     for k, t in st.session_state.logs]
            st.markdown(
                f'<div class="logpane">{"<br>".join(lines)}</div>',
                unsafe_allow_html=True,
            )
            if st.session_state.last_output:
                st.markdown(f"""
<div class="res-wrap">
  <div class="res-hdr">✍️ last output</div>
  <div class="res-body">{_html.escape(st.session_state.last_output)}</div>
</div>""", unsafe_allow_html=True)
    else:
        st.info("← Select a pipeline above.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOM BUILDER
# ════════════════════════════════════════════════════════════════════════════
with tab_custom:
    st.markdown("### 🛠️ Custom Agent Builder")
    st.caption(
        "Define your own main-agent instructions and up to 4 custom sub-agents. "
        "All will be wired through `create_deep_agent(subagents=[...])`."
    )

    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("#### Main Agent")
        custom_instructions = st.text_area(
            "Main agent instructions",
            height=160,
            placeholder=(
                "You are the Lead Agent.\n\n"
                "WORKFLOW:\n"
                "1. write_todos: plan 4-6 steps.\n"
                "2. task(agent_name='researcher', task=...) to gather info.\n"
                "3. write_file('notes.md', ...) to persist context.\n"
                "4. task(agent_name='writer', task=...) for final output."
            ),
            key="custom_main_instructions",
        )
        custom_task = st.text_area(
            "Task to run",
            height=100,
            placeholder="Describe the task you want the multi-agent pipeline to complete…",
            key="custom_task_input",
        )

    with c_right:
        st.markdown("#### Sub-Agents")
        num_subagents = st.slider("Number of sub-agents", 1, 4, 2, key="num_sub")

        custom_subs = []
        for i in range(num_subagents):
            with st.expander(f"Sub-Agent {i+1}", expanded=(i == 0)):
                name = st.text_input(f"Name", key=f"sub_name_{i}",
                                     placeholder=f"researcher" if i == 0 else f"writer",
                                     value=["researcher","writer","analyst","critic"][i] if i < 4 else "")
                desc = st.text_input(f"Description (shown to main agent)", key=f"sub_desc_{i}",
                                     placeholder="What this sub-agent specialises in")
                prompt = st.text_area(f"System prompt", key=f"sub_prompt_{i}", height=100,
                                      placeholder="You are a specialist in… Use knowledge_lookup to…")
                if name and desc and prompt:
                    custom_subs.append({
                        "name": name, "description": desc, "system_prompt": prompt,
                    })

    st.markdown("---")
    col_run_c, col_rst_c = st.columns([5, 1])
    with col_run_c:
        run_custom = st.button(
            "🚀  Run Custom Pipeline",
            type="primary",
            use_container_width=True,
            key="run_custom_btn",
        )
    with col_rst_c:
        if st.button("↺", use_container_width=True, key="rst_custom", help="Reset"):
            st.session_state.logs = []
            st.session_state.last_output = ""
            st.rerun()

    if run_custom:
        if not st.session_state.groq_key:
            st.error("🔑 Enter your NVIDIA API key in the sidebar.")
            st.stop()
        if not custom_instructions.strip():
            st.warning("Enter main agent instructions.")
            st.stop()
        if not custom_task.strip():
            st.warning("Enter a task to run.")
            st.stop()
        if not custom_subs:
            st.warning("Define at least one sub-agent (Name + Description + System Prompt).")
            st.stop()

        st.session_state.logs = []
        logs = st.session_state.logs
        logs.append(("lh", "─" * 54))
        logs.append(("lh", "  CUSTOM PIPELINE"))
        logs.append(("lh", f"  sub-agents: {', '.join(s['name'] for s in custom_subs)}"))
        logs.append(("lh", "─" * 54))

        custom_pipeline_cfg = {
            "icon": "🛠️", "title": "Custom Pipeline", "subagents": custom_subs,
            "flow": " → ".join(["main"] + [s["name"] for s in custom_subs]),
        }

        with st.spinner("Building custom agent…"):
            try:
                agent = build_agent(
                    st.session_state.groq_key, MODEL,
                    custom_instructions, custom_subs,
                )
            except Exception as e:
                st.error(f"Build error: {e}")
                st.code(traceback.format_exc())
                st.stop()

        logs.append(("lm", f"✅ create_deep_agent compiled with {len(custom_subs)} sub-agents"))

        try:
            final_output, res_ph = run_agent_stream(agent, custom_task, custom_pipeline_cfg)
            logs.append(("lf", f"✅ Custom pipeline complete"))

            css_map = {"lh":"lh","lm":"lm","lt":"lt","lr":"lr","lf":"lf","lw":"lw",
                       "ls0":"ls0","ls1":"ls1","ls2":"ls2","ls3":"ls3"}
            lines = [f'<span class="{css_map.get(k,"lr")}">{_html.escape(str(t))}</span>'
                     for k, t in logs[-40:]]
            st.markdown(f'<div class="logpane">{"<br>".join(lines)}</div>', unsafe_allow_html=True)

            if final_output:
                st.session_state.last_output = final_output
                res_ph.markdown(f"""
<div class="res-wrap">
  <div class="res-hdr">✍️ custom pipeline output</div>
  <div class="res-body">{_html.escape(final_output)}</div>
</div>""", unsafe_allow_html=True)
                st.download_button(
                    "⬇️ Download output",
                    data=final_output,
                    file_name=f"custom_agent_{datetime.now().strftime('%H%M%S')}.md",
                    mime="text/markdown",
                    key="dl_custom",
                )
            st.session_state.run_history.append({
                "icon": "🛠️", "title": "Custom Pipeline",
                "ts": datetime.now().strftime("%H:%M:%S"), "model": MODEL,
            })

        except Exception as e:
            err_str = str(e)
            if "413" in err_str or "Request too large" in err_str or "tokens per minute" in err_str.lower():
                st.error("🔴 **Token limit exceeded (Error 413)** — switch to **nvidia/llama-3.1-8b-instruct** in the sidebar.")
            else:
                st.error(f"❌ {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — FREE-FORM  (single deep agent, no pre-defined sub-agents)
# ════════════════════════════════════════════════════════════════════════════
with tab_freeform:
    st.markdown("### 💬 Free-Form Deep Agent")
    st.caption(
        "One `create_deep_agent` with all domain tools and deepagents built-ins "
        "(write_todos, filesystem, compact_conversation). No pre-defined sub-agents — "
        "the agent decides its own approach."
    )

    ff_system = st.text_area(
        "Agent instructions (optional — leave blank for default)",
        height=90,
        placeholder=(
            "You are a helpful deep agent. Use write_todos to plan, "
            "write_file to persist context, and all available tools."
        ),
        key="ff_system",
    )
    ff_task = st.text_area(
        "Your task",
        height=130,
        placeholder=(
            "Ask anything — analyse code, research a topic, plan a project, "
            "debug an error, or write documentation…"
        ),
        key="ff_task",
    )

    col_ff_run, col_ff_rst = st.columns([5, 1])
    with col_ff_run:
        ff_run = st.button("🚀  Run Free-Form Agent", type="primary",
                           use_container_width=True, key="ff_run_btn")
    with col_ff_rst:
        if st.button("↺", use_container_width=True, key="ff_rst", help="Reset"):
            st.session_state.logs = []
            st.session_state.last_output = ""
            st.rerun()

    if ff_run:
        if not st.session_state.groq_key:
            st.error("🔑 Enter your NVIDIA API key in the sidebar.")
            st.stop()
        if not ff_task.strip():
            st.warning("Enter a task.")
            st.stop()

        instructions = ff_system.strip() or (
            "You are a powerful Deep Agent.\n"
            "Always start by calling write_todos to plan your approach.\n"
            "Use write_file to save intermediate results.\n"
            "Use all available tools as needed.\n"
            "Produce a thorough, well-structured final answer."
        )

        st.session_state.logs = []
        logs = st.session_state.logs
        logs.append(("lh", "─" * 54))
        logs.append(("lh", "  FREE-FORM DEEP AGENT"))
        logs.append(("lh", f"  model: nvidia:{MODEL}"))
        logs.append(("lh", "─" * 54))

        with st.spinner("Building agent…"):
            try:
                agent = build_agent(
                    st.session_state.groq_key, MODEL,
                    instructions, [],   # no sub-agents in free-form mode
                )
            except Exception as e:
                st.error(f"Build error: {e}"); st.stop()

        logs.append(("lm", "✅ create_deep_agent ready (no sub-agents, all built-in tools)"))
        ff_cfg = {"icon": "💬", "title": "Free-Form", "subagents": [],
                  "flow": "main-agent (autonomous)"}
        try:
            final_output, res_ph = run_agent_stream(agent, ff_task, ff_cfg)
            logs.append(("lf", "✅ Done"))

            css_map = {"lh":"lh","lm":"lm","lt":"lt","lr":"lr","lf":"lf","lw":"lw",
                       "ls0":"ls0","ls1":"ls1","ls2":"ls2","ls3":"ls3"}
            lines = [f'<span class="{css_map.get(k,"lr")}">{_html.escape(str(t))}</span>'
                     for k, t in logs[-40:]]
            st.markdown(f'<div class="logpane">{"<br>".join(lines)}</div>', unsafe_allow_html=True)

            if final_output:
                st.session_state.last_output = final_output
                res_ph.markdown(f"""
<div class="res-wrap">
  <div class="res-hdr">✍️ agent output</div>
  <div class="res-body">{_html.escape(final_output)}</div>
</div>""", unsafe_allow_html=True)
                st.download_button(
                    "⬇️ Download output",
                    data=final_output,
                    file_name=f"freeform_{datetime.now().strftime('%H%M%S')}.md",
                    mime="text/markdown",
                    key="dl_ff",
                )
            st.session_state.run_history.append({
                "icon": "💬", "title": "Free-Form",
                "ts": datetime.now().strftime("%H:%M:%S"), "model": MODEL,
            })
        except Exception as e:
            err_str = str(e)
            if "413" in err_str or "Request too large" in err_str or "tokens per minute" in err_str.lower():
                st.error("🔴 **Token limit exceeded (Error 413)** — switch to **nvidia/llama-3.1-8b-instruct** in the sidebar.")
            else:
                st.error(f"❌ {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — FRAMEWORK DOCS
# ════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("### 📖 deepagents Framework Reference")

    st.markdown("""
#### What is `deepagents`?

`deepagents` is LangChain's batteries-included agent harness, built on top of LangGraph.
It solves the "shallow agent" problem by bundling four key capabilities that power tools like
Claude Code, Deep Research, and Manus:

| Capability | deepagents mechanism |
|------------|---------------------|
| **Explicit planning** | `write_todos` built-in tool |
| **Context persistence** | Virtual filesystem (`read_file`, `write_file`, `edit_file`) |
| **Sub-agent delegation** | `task(agent_name, task)` built-in tool |
| **Context window management** | `compact_conversation` + `SummarizationMiddleware` |

---

#### Installation

```bash
pip install deepagents openai langchain-openai langchain
```

---

#### Core API

```python
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 1. Define your model — NVIDIA NIM (OpenAI-compatible, no TPM cap)
model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="nvapi-YOUR_KEY_HERE",
    model="nvidia/llama-3.1-8b-instruct",   # or any NVIDIA NIM model
)

# 2. Define sub-agents (optional but powerful)
researcher = {
    "name":          "researcher",
    "description":   "Gathers facts and data about the topic",   # shown to main agent
    "system_prompt": "You are a precise researcher. Use knowledge_lookup...",
    "model":         model,    # optional — defaults to main model
    "tools":         [...],    # optional — defaults to main agent tools
}

critic = {
    "name":          "critic",
    "description":   "Finds weaknesses and counterarguments",
    "system_prompt": "You are a rigorous critic...",
}

# 3. Create the agent
agent = create_deep_agent(
    model        = model,
    tools        = [my_tool_1, my_tool_2],   # custom @tool functions
    instructions = "You are the lead agent. Workflow: 1. write_todos...",
    subagents    = [researcher, critic],      # main agent can call task('researcher', ...)
)

# 4. agent is a LangGraph CompiledStateGraph — invoke or stream
result = agent.invoke({"messages": [HumanMessage(content="Research LangGraph")]})

# Or stream for real-time output:
for chunk in agent.stream(
    {"messages": [HumanMessage(content="Research LangGraph")]},
    stream_mode="updates",
):
    for node, data in chunk.items():
        for msg in data.get("messages", []):
            print(msg.content)
```

---

#### Built-in Tools (auto-injected — no setup needed)

| Tool | Signature | Purpose |
|------|-----------|---------|
| `write_todos` | `write_todos(todos: list[str])` | Agent plans explicitly |
| `read_file` | `read_file(filename: str)` | Read from virtual filesystem |
| `write_file` | `write_file(filename: str, content: str)` | Write to virtual filesystem |
| `edit_file` | `edit_file(filename, old_text, new_text)` | Patch a file in-place |
| `ls` | `ls(path: str = ".")` | List directory contents |
| `glob` | `glob(pattern: str)` | Find files by pattern |
| `grep` | `grep(pattern, filename)` | Search within files |
| `task` | `task(agent_name: str, task: str)` | **Spawn a named sub-agent** |
| `compact_conversation` | `compact_conversation()` | Summarise history to free context |

---

#### Sub-Agent Delegation Flow

```
Main Agent
    │
    │  calls: task(agent_name="researcher", task="Find benchmarks for X")
    │
    ▼
deepagents SubAgentMiddleware
    │
    │  1. Looks up "researcher" in subagents=[...]
    │  2. Creates isolated agent with researcher's system_prompt + tools
    │  3. Runs to completion (its own tool-calling loop)
    │  4. Returns final text as ToolMessage to main agent
    │
    ▼
Main Agent continues with researcher's output in context
```

---

#### Middleware Stack (injected automatically)

```
create_deep_agent()
    │
    └─► MemoryMiddleware          (optional cross-session memory)
        └─► SkillsMiddleware      (optional skills injection)
            └─► TodoListMiddleware (powers write_todos)
                └─► FilesystemMiddleware (powers read/write_file, ls, glob, grep)
                    └─► SubAgentMiddleware (powers task tool)
                        └─► SummarizationMiddleware (auto-compact on context overflow)
                            └─► LangGraph ReAct loop
```

---

#### Custom Tools

Your tools are plain `@tool` functions and are passed alongside deepagents' built-ins:

```python
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    \"\"\"Search the internal documentation.\"\"\"\
    return my_search_engine.query(query)

agent = create_deep_agent(
    model=model,
    tools=[search_docs],          # custom + deepagents built-ins all available
    instructions="...",
    subagents=[...],
)
```

---

#### Links

- **PyPI:** `pip install deepagents` → [pypi.org/project/deepagents](https://pypi.org/project/deepagents/)
- **GitHub:** [github.com/langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)
- **Docs:** [docs.langchain.com/oss/python/deepagents](https://docs.langchain.com/oss/python/deepagents/overview)
- **NVIDIA NIM API:** [build.nvidia.com](https://build.nvidia.com) — free credits, no TPM limit
    """)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1rem 0;border-top:1px solid #111d33;
     text-align:center;color:#111d33;font-size:.68rem;
     font-family:JetBrains Mono,monospace;letter-spacing:.05em;">
  pip install deepagents openai langchain-openai  ·  langchain-ai/deepagents  ·  MIT license  ·  NVIDIA NIM free tier
</div>
""", unsafe_allow_html=True)
