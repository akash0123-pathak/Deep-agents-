"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          LangChain Deep Agents Explorer  ·  Streamlit Edition               ║
║  Explores every Deep Agents feature: Planning, Filesystem, Subagents,       ║
║  Memory, Context Management, Middleware, Skills — powered by Groq (free)    ║
╚══════════════════════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import os, json, time, datetime
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Agents Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  code, pre { font-family: 'IBM Plex Mono', monospace !important; }
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding: 1.5rem 2rem 3rem; max-width: 1200px; }
  section[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1f2333; }

  .da-card { background:#111318; border:1px solid #1f2333; border-radius:8px; padding:1rem 1.25rem; margin-bottom:.75rem; }
  .da-card-blue   { border-left:3px solid #4f8ef7; }
  .da-card-green  { border-left:3px solid #34d399; }
  .da-card-purple { border-left:3px solid #a78bfa; }
  .da-card-yellow { border-left:3px solid #fbbf24; }
  .da-card-pink   { border-left:3px solid #f472b6; }
  .da-card-orange { border-left:3px solid #fb923c; }
  .da-card-cyan   { border-left:3px solid #22d3ee; }

  .badge { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:.65rem; padding:2px 8px;
           border-radius:3px; letter-spacing:.08em; text-transform:uppercase; margin-right:6px; }
  .badge-blue   { background:#1e3a5f; color:#4f8ef7; border:1px solid #2d5288; }
  .badge-green  { background:#0f2e1e; color:#34d399; border:1px solid #1a4a30; }
  .badge-purple { background:#2d1f4e; color:#a78bfa; border:1px solid #4a3580; }
  .badge-yellow { background:#2e2300; color:#fbbf24; border:1px solid #4a3800; }
  .badge-red    { background:#2e1010; color:#f87171; border:1px solid #4a1a1a; }

  .section-header { font-family:'IBM Plex Mono',monospace; font-size:1.3rem; font-weight:700; margin-bottom:.25rem; }
  .section-sub    { font-size:.78rem; color:#6b7280; letter-spacing:.06em; margin-bottom:1.25rem; }

  .trace-box { background:#0d1117; border:1px solid #1f2333; border-radius:6px; padding:.9rem 1rem;
               font-family:'IBM Plex Mono',monospace; font-size:.75rem; color:#94a3b8;
               white-space:pre-wrap; max-height:400px; overflow-y:auto; line-height:1.65; }

  .fs-item { display:flex; align-items:center; gap:8px; padding:5px 8px;
             border-bottom:1px solid #1a1d26; font-family:'IBM Plex Mono',monospace;
             font-size:.75rem; color:#94a3b8; }
  .mw-layer { display:flex; align-items:center; gap:10px; background:#111318;
              border:1px solid #1f2333; border-radius:6px; padding:8px 14px; margin-bottom:6px; }
  .metric-tile { background:#111318; border:1px solid #1f2333; border-radius:8px;
                 padding:.9rem; text-align:center; }
  .metric-tile .val { font-family:'IBM Plex Mono',monospace; font-size:1.6rem; font-weight:700; }
  .metric-tile .lbl { font-size:.72rem; color:#6b7280; margin-top:2px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT BACKEND  (real deepagents + Groq; graceful fallback to simulation)
# ═══════════════════════════════════════════════════════════════════════════════

def _try_import():
    try:
        from deepagents import create_deep_agent
        from langchain_groq import ChatGroq
        from langchain_core.tools import tool
        return create_deep_agent, ChatGroq, tool
    except ImportError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATION DATA
# ═══════════════════════════════════════════════════════════════════════════════

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

MIDDLEWARE_LAYERS = [
    {"icon":"📋","name":"TodoListMiddleware",    "hook":"before_agent",    "color":"#4f8ef7",
     "desc":"Adds write_todos tool; agent creates a step plan before acting."},
    {"icon":"💾","name":"MemoryMiddleware",      "hook":"before_model",    "color":"#34d399",
     "desc":"Loads AGENTS.md + cross-session memories into system prompt."},
    {"icon":"🛠","name":"SkillsMiddleware",      "hook":"before_model",    "color":"#a78bfa",
     "desc":"Injects skill definitions (markdown files) into system prompt."},
    {"icon":"📁","name":"FilesystemMiddleware",  "hook":"wrap_tool_call",  "color":"#fbbf24",
     "desc":"Provides ls/read/write/edit/glob/grep tools backed by pluggable storage."},
    {"icon":"🔀","name":"SubAgentMiddleware",    "hook":"wrap_model_call", "color":"#f472b6",
     "desc":"Injects 'task' tool; parent spawns isolated child agents with own context."},
    {"icon":"✂️","name":"SummarizationMiddleware","hook":"before_model",   "color":"#fb923c",
     "desc":"Detects context overflow → auto-compresses history to save tokens."},
    {"icon":"🧑","name":"HILMiddleware",         "hook":"wrap_tool_call",  "color":"#22d3ee",
     "desc":"Pauses before specified tools for human approval / modification."},
]

SUBAGENT_TYPES = [
    {"name":"general_purpose","model":"parent model",        "tools":"all built-in",                    "color":"#4f8ef7"},
    {"name":"researcher",     "model":"groq:llama-3.3-70b", "tools":"web_search, read_file, write_file","color":"#34d399"},
    {"name":"coder",          "model":"groq:llama-3.3-70b", "tools":"execute, write_file, edit_file",  "color":"#a78bfa"},
    {"name":"analyst",        "model":"groq:gemma2-9b-it",  "tools":"read_file, write_file, glob",     "color":"#fbbf24"},
]

def simulated_planning_trace(task):
    return [
        ("think",   f"Received task: {task}"),
        ("tool",    "write_todos", f"Breaking down: '{task}'", "✓ Created 4-step plan"),
        ("step",    "Step 1 → Research & gather information"),
        ("tool",    "read_file",   "context/research_notes.md", "File not found — starting fresh"),
        ("tool",    "write_file",  "context/notes.md",          "✓ Written 0.8 KB scratchpad"),
        ("step",    "Step 2 → Analyse and synthesise"),
        ("tool",    "write_file",  "context/analysis.md",       "✓ Written 1.2 KB"),
        ("step",    "Step 3 → Delegate sub-task to specialist subagent"),
        ("tool",    "task",        "subagent_type=researcher",  "✓ Subagent completed — results in context/"),
        ("step",    "Step 4 → Compile final report"),
        ("tool",    "write_file",  "output/report.md",          "✓ Written 3.7 KB"),
        ("final",   f"Task complete: {task}\nArtifacts → output/report.md"),
    ]

def simulated_fs_state():
    return [
        ("AGENTS.md",               "1.4 KB","system"),
        ("skills/researcher.md",    "3.2 KB","skill"),
        ("skills/coder.md",         "2.8 KB","skill"),
        ("context/research_notes.md","8.1 KB","context"),
        ("context/analysis.md",     "1.2 KB","context"),
        ("output/report.md",        "3.7 KB","output"),
        ("memory/user_prefs.json",  "0.6 KB","memory"),
        ("memory/past_tasks.jsonl", "12.3 KB","memory"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🤖 Deep Agents Explorer")
    st.markdown('<p style="font-size:.72rem;color:#4f8ef7;letter-spacing:.08em">LANGCHAIN · GROQ · STREAMLIT</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🔑 Groq API Key")
    groq_key = st.text_input("Free key from console.groq.com", type="password", placeholder="gsk_...")
    if groq_key:
        st.success("Key loaded ✓", icon="✅")
    else:
        st.info("Without a key the app runs in **simulation mode** — all features demonstrated with realistic mock traces.", icon="ℹ️")

    st.markdown("### ⚙️ Model")
    model_choice = st.selectbox("Groq model", GROQ_MODELS, index=0)

    st.markdown("### 📑 Navigation")
    page = st.radio("", [
        "🏠 Overview",
        "📋 Task Planning",
        "📁 Virtual Filesystem",
        "🔀 Subagents",
        "💾 Long-Term Memory",
        "✂️ Context Management",
        "🛠 Middleware Stack",
        "🧑 Skills System",
        "🚀 Live Agent Demo",
        "📦 Code Reference",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown('<div style="font-size:.7rem;color:#4b5563;line-height:1.6"><b style="color:#6b7280">Deep Agents v0.4</b><br>Built on LangGraph · MIT License<br>Released July 2025 · 9.9k ★</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_trace_animated(steps):
    ph = st.empty()
    lines = []
    for s in steps:
        t = s[0]
        if t == "tool":
            _, name, args, result = s
            lines.append(f"🔧 TOOL CALL  [{name}]")
            lines.append(f"   args   → {args}")
            lines.append(f"   result → {result}")
        elif t == "final":
            lines.append(f"\n✅ FINAL OUTPUT\n{s[1]}")
        elif t == "step":
            lines.append(f"▶  {s[1]}")
        else:
            lines.append(f"💭 {s[1]}")
        ph.markdown(f'<div class="trace-box">{chr(10).join(lines)}</div>', unsafe_allow_html=True)
        time.sleep(0.15)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── OVERVIEW ─────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="section-header" style="color:#4f8ef7">Deep Agents Framework</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">LANGCHAIN · OPEN-SOURCE · BATTERIES INCLUDED · V0.4</div>', unsafe_allow_html=True)
    st.markdown(' '.join([
        '<span class="badge badge-blue">MIT Licensed</span>',
        '<span class="badge badge-green">pip install deepagents</span>',
        '<span class="badge badge-purple">LangGraph Native</span>',
        '<span class="badge badge-yellow">Groq Ready</span>',
        '<span class="badge badge-red">9.9k ★ GitHub</span>',
    ]), unsafe_allow_html=True)
    st.markdown("")

    cols = st.columns(4)
    tiles = [
        ("Planning",    "Built-in write_todos for multi-step decomposition",  "#4f8ef7"),
        ("Filesystem",  "Virtual FS offloads large artifacts from context",   "#34d399"),
        ("Subagents",   "Spawn isolated child agents for parallel work",       "#a78bfa"),
        ("Memory",      "Cross-session persistence via LangGraph Store",      "#fbbf24"),
    ]
    for col, (v, l, c) in zip(cols, tiles):
        col.markdown(f'<div class="metric-tile"><div class="val" style="color:{c}">⬡</div><div style="font-size:.9rem;font-weight:600;margin-top:6px">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### What is Deep Agents?")
        st.markdown("""
Deep Agents is LangChain's **batteries-included agent harness** built on LangGraph — launched July 2025.
It packages the same architecture behind **Claude Code**, **Deep Research**, and **Manus** into an
open-source, provider-agnostic library installed in one command.

**The problem it solves:** Traditional agent loops (think → act → observe) break down on real-world tasks
that are multi-step, stateful, and artifact-heavy. Deep Agents adds five built-in systems:

- **Planning** — automatic task decomposition via `write_todos`
- **Virtual Filesystem** — offload large context to pluggable storage (S3 / local / in-memory)
- **Subagents** — spawn isolated child agents via `task` tool
- **Memory** — cross-thread persistence via LangGraph Memory Store
- **Middleware** — composable pipeline: summarisation, skills, HIL, PII redaction
""")
    with c2:
        st.code("""from deepagents import create_deep_agent
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="gsk_..."
)

agent = create_deep_agent(
    model=llm,
    system_prompt="You are a researcher.",
)

# Invoke
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Research quantum computing"
    }]
})

# Stream
for chunk in agent.stream({...}):
    print(chunk)
""", language="python")

    st.markdown("---")
    st.markdown("### Architecture")
    st.code("""create_deep_agent()
├── LangGraph State Machine          ← durable execution, streaming, checkpointing
├── Middleware Pipeline
│   ├── TodoListMiddleware           ← write_todos → structured step plan
│   ├── MemoryMiddleware             ← AGENTS.md + cross-session memories
│   ├── SkillsMiddleware             ← injects skill .md files into system prompt
│   ├── FilesystemMiddleware         ← ls / read_file / write_file / edit_file / glob / grep
│   ├── SubAgentMiddleware           ← task tool → isolated child agent per subtask
│   ├── SummarizationMiddleware      ← auto-compress history on context overflow
│   └── HILMiddleware                ← human-in-the-loop approval gates
├── Backend Protocol                 ← S3 / local disk / in-memory (pluggable)
└── Your Custom @tool functions      ← any LangChain-compatible tool
""", language="text")

    st.markdown("### Deep Agents vs alternatives")
    import pandas as pd
    st.dataframe(pd.DataFrame({
        "Feature":                ["Planning","Virtual FS","Subagents","Auto-summarise","Provider agnostic","LangGraph native"],
        "Deep Agents":            ["✅","✅","✅","✅","✅","✅"],
        "LangChain create_agent": ["❌","❌","❌","❌","✅","✅"],
        "Raw LangGraph":          ["❌","❌","❌","❌","✅","✅"],
        "Simple ReAct":           ["❌","❌","❌","❌","⚠️","❌"],
    }), use_container_width=True, hide_index=True)


# ─── TASK PLANNING ────────────────────────────────────────────────────────────
elif page == "📋 Task Planning":
    st.markdown('<div class="section-header" style="color:#a78bfa">Task Planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">WRITE_TODOS TOOL · STEP DECOMPOSITION · STRATEGIC EXECUTION</div>', unsafe_allow_html=True)
    st.markdown("""
Deep Agents automatically breaks complex tasks into ordered steps using the built-in **`write_todos` tool**.
Before execution begins the agent plans — transforming it from *reactive* to *strategic*.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.code("""# Planning is ON by default
from deepagents import create_deep_agent
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile")

agent = create_deep_agent(
    model=llm,
    system_prompt="You are a research analyst.",
)

# Agent will AUTOMATICALLY:
# 1. Call write_todos to plan
# 2. Execute each step
# 3. Update todos as it progresses

result = agent.invoke({"messages": [{
    "role": "user",
    "content": "Write a competitive analysis"
}]})
""", language="python")
    with c2:
        st.code("""# Disable planning (not recommended)
from deepagents.middleware import TodoListMiddleware

agent_no_plan = create_deep_agent(
    model=llm,
    excluded_middleware=[TodoListMiddleware],
)

# What write_todos produces:
# {
#   "todos": [
#     {"id": 1, "task": "Research market", "status": "pending"},
#     {"id": 2, "task": "Analyse competitors", "status": "pending"},
#     {"id": 3, "task": "Write summary",  "status": "pending"},
#   ]
# }
# Agent updates status → "done" as it progresses
""", language="python")

    st.markdown("---")
    st.markdown("### Simulate Planning Trace")
    task_input = st.text_input("Task", value="Write a competitive analysis for a new B2B SaaS product")

    if st.button("▶  Run Planning Trace", type="primary"):
        render_trace_animated(simulated_planning_trace(task_input))

    if groq_key:
        st.markdown("---")
        st.markdown("### 🟢 Live Execution")
        live_task = st.text_area("Live task", value=task_input, height=80)
        if st.button("🚀  Run Live Deep Agent", type="primary"):
            imports = _try_import()
            if imports:
                create_deep_agent, ChatGroq, _ = imports
                llm = ChatGroq(model=model_choice, temperature=0, api_key=groq_key)
                agent = create_deep_agent(model=llm, system_prompt="You are a strategic research analyst. Always plan before acting.")
                ph = st.empty()
                full = ""
                with st.spinner("Agent running…"):
                    for chunk in agent.stream({"messages": [{"role": "user", "content": live_task}]}):
                        for msg in chunk.get("messages", []):
                            if hasattr(msg, "content") and isinstance(msg.content, str):
                                full += msg.content
                                ph.markdown(f'<div class="trace-box">{full}</div>', unsafe_allow_html=True)
            else:
                st.warning("Run `pip install deepagents langchain-groq` first, then restart Streamlit.")


# ─── VIRTUAL FILESYSTEM ───────────────────────────────────────────────────────
elif page == "📁 Virtual Filesystem":
    st.markdown('<div class="section-header" style="color:#34d399">Virtual Filesystem</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">CONTEXT OFFLOADING · PLUGGABLE BACKENDS · ARTIFACT MANAGEMENT</div>', unsafe_allow_html=True)
    st.markdown("""
Rather than stuffing every artifact into the active prompt, Deep Agents provides a **virtual filesystem**
backed by LangGraph's state store. Large outputs get written to storage and retrieved on demand —
preventing context overflow on long tasks.
""")

    c1, c2, c3 = st.columns(3)
    for col, (b, d, c) in zip([c1,c2,c3],[
        ("In-Memory",   "Default. No config. Resets between runs.",      "#4f8ef7"),
        ("Local Disk",  "Persists across runs on the same machine.",      "#34d399"),
        ("S3 / Remote", "Production: share state across agents & deploys.","#a78bfa"),
    ]):
        col.markdown(f'<div class="da-card" style="border-left:3px solid {c}"><b style="color:{c}">{b}</b><br><span style="font-size:.8rem;color:#94a3b8">{d}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Built-in Filesystem Tools")
    for name, desc, ex in [
        ("ls",         "List directory contents",          "ls('context/')"),
        ("read_file",  "Read a file from virtual FS",      "read_file('output/report.md')"),
        ("write_file", "Write content to virtual FS",      "write_file('notes.md', content)"),
        ("edit_file",  "Apply targeted edits to a file",   "edit_file('notes.md', old_str, new_str)"),
        ("glob",       "Pattern-match file paths",         "glob('**/*.md')"),
        ("grep",       "Search file contents",             "grep('keyword', 'context/')"),
    ]:
        st.markdown(f'<div class="da-card da-card-green"><code style="color:#34d399">{name}</code>&nbsp;<span style="color:#94a3b8;font-size:.82rem">{desc}</span><br><code style="font-size:.72rem;color:#6b7280">Example: {ex}</code></div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Current FS State (simulated)")
        type_colors = {"system":"#4f8ef7","skill":"#a78bfa","context":"#fbbf24","output":"#34d399","memory":"#f472b6"}
        rows = "".join([
            f'<div class="fs-item"><span style="color:{type_colors[t]}">{t[0].upper()}</span>'
            f'<code style="color:#e2e8f0;flex:1">{p}</code>'
            f'<span style="color:#6b7280;font-size:.7rem">{s}</span></div>'
            for p, s, t in simulated_fs_state()
        ])
        st.markdown(f'<div style="background:#0d1117;border:1px solid #1f2333;border-radius:6px;padding:8px">{rows}</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("### Backend Configuration")
        st.code("""from deepagents import create_deep_agent
from deepagents.backends import (
    InMemoryBackend,          # default
    LocalFilesystemBackend,
    S3Backend,
    CompositeBackend,
)

# Local disk
agent = create_deep_agent(
    model=llm,
    backend=LocalFilesystemBackend(
        root="./agent_workspace"
    ),
)

# S3 (production)
agent = create_deep_agent(
    model=llm,
    backend=S3Backend(
        bucket="my-agent-bucket",
        prefix="runs/",
        region_name="us-east-1",
    ),
)

# Composite routing (v0.2+)
agent = create_deep_agent(
    model=llm,
    backend=CompositeBackend(
        memory=InMemoryBackend(),
        skills=LocalFilesystemBackend("./skills"),
        output=S3Backend(bucket="results"),
    ),
)
""", language="python")


# ─── SUBAGENTS ────────────────────────────────────────────────────────────────
elif page == "🔀 Subagents":
    st.markdown('<div class="section-header" style="color:#f472b6">Subagents</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">PARALLEL DELEGATION · ISOLATED CONTEXT · SPECIALIST MODELS</div>', unsafe_allow_html=True)
    st.markdown("""
Deep Agents can spawn **isolated child agents** for independent subtasks via the built-in **`task` tool**.
Each subagent gets its own context window — the parent pays zero tokens for the child's history.
You can run a cheaper model for routine subtasks and a larger one for the top-level planner.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.code("""from deepagents import create_deep_agent, SubAgent
from langchain_groq import ChatGroq

llm  = ChatGroq(model="llama-3.3-70b-versatile")
fast = ChatGroq(model="llama3-8b-8192")  # cheaper

agent = create_deep_agent(
    model=llm,
    subagents=[
        SubAgent(
            name="researcher",
            model=llm,
            tools=[web_search, read_file, write_file],
            system_prompt="You are a web researcher.",
        ),
        SubAgent(
            name="coder",
            model=llm,
            tools=[execute, write_file, edit_file],
            system_prompt="You are a software engineer.",
        ),
        SubAgent(
            name="analyst",
            model=fast,   # cheap model for analysis
            tools=[read_file, write_file, glob],
            system_prompt="You are a data analyst.",
        ),
    ],
)
""", language="python")
    with c2:
        st.code("""# Subagent execution flow
#
# Parent (llama-3.3-70b)
# │  [write_todos → 3 steps]
# │
# │  [task] subagent_type=researcher
# │   └─► Researcher (isolated context)
# │       [web_search("AI papers 2026")]
# │       [write_file("context/papers.md")]
# │       returns: "Done. Notes at papers.md"
# │
# │  [task] subagent_type=coder
# │   └─► Coder (isolated context)
# │       [read_file("context/papers.md")]
# │       [write_file("output/impl.py")]
# │       [execute("python output/impl.py")]
# │       returns: "Implemented. Tests pass."
# │
# │  [write_file("output/SUMMARY.md")]
# └─ DONE
#
# Key: each child's full history is INVISIBLE
# to the parent — only the final message passes.
""", language="text")

    st.markdown("### Subagent Types")
    cols = st.columns(len(SUBAGENT_TYPES))
    for col, sa in zip(cols, SUBAGENT_TYPES):
        col.markdown(f'<div class="da-card" style="border-left:3px solid {sa["color"]}"><b style="color:{sa["color"]}">{sa["name"]}</b><br><div style="font-size:.72rem;color:#94a3b8;margin-top:6px"><b>Model:</b> {sa["model"]}<br><b>Tools:</b> {sa["tools"]}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Simulate Subagent Delegation")
    if st.button("▶  Run Subagent Trace", type="primary"):
        trace_lines = [
            "💭 Parent → task: 'Research + implement binary search tree in Python'",
            "🔧 TOOL CALL  [write_todos]",
            "   → Step 1: Research BST algorithms",
            "   → Step 2: Implement in Python",
            "   → Step 3: Write tests",
            "",
            "🔧 TOOL CALL  [task]  subagent_type=researcher",
            "   ┌─ Researcher Subagent (isolated context) ─────────────┐",
            "   │  🔧 web_search('binary search tree Python 2026')     │",
            "   │  🔧 write_file('context/bst_notes.md', ...)          │",
            "   │  ✅ Done — 2.1 KB written to context/bst_notes.md   │",
            "   └──────────────────────────────────────────────────────┘",
            "   ↳ ToolMessage → 'Research done. Notes at context/bst_notes.md'",
            "",
            "🔧 TOOL CALL  [task]  subagent_type=coder",
            "   ┌─ Coder Subagent (isolated context) ───────────────────┐",
            "   │  🔧 read_file('context/bst_notes.md')                 │",
            "   │  🔧 write_file('output/bst.py', implementation)       │",
            "   │  🔧 execute('python -m pytest output/test_bst.py')    │",
            "   │  ✅ 3/3 tests passed                                   │",
            "   └──────────────────────────────────────────────────────  ┘",
            "   ↳ ToolMessage → 'bst.py written, all tests passing.'",
            "",
            "🔧 TOOL CALL  [write_file]  'output/SUMMARY.md'",
            "✅ COMPLETE — Artifacts: context/bst_notes.md · output/bst.py · output/SUMMARY.md",
        ]
        ph = st.empty()
        shown = []
        for line in trace_lines:
            shown.append(line)
            ph.markdown(f'<div class="trace-box">{chr(10).join(shown)}</div>', unsafe_allow_html=True)
            time.sleep(0.12)


# ─── LONG-TERM MEMORY ─────────────────────────────────────────────────────────
elif page == "💾 Long-Term Memory":
    st.markdown('<div class="section-header" style="color:#f472b6">Long-Term Memory</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">CROSS-SESSION PERSISTENCE · AGENTS.MD · LANGGRAPH MEMORY STORE</div>', unsafe_allow_html=True)
    st.markdown("""
Deep Agents persists knowledge **across sessions** via LangGraph's Memory Store.
Memory takes two forms:

- **AGENTS.md** — loaded at every startup; defines agent identity, principles, and standing preferences
- **Dynamic memories** — facts the agent saves/loads during execution (user prefs, past summaries, task history)
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### AGENTS.md — Agent Identity File")
        st.code("""# AGENTS.md  (in virtual FS root)

## Identity
You are a senior research analyst specialising
in technology trends.

## Principles
- Always create a plan with write_todos first
- Save intermediate findings to context/
- Summarise long documents before storing

## User Preferences
- Format: Markdown with headers
- Depth: detailed with citations
- Language: British English

## Standing Instructions
- Check 3+ sources when web searching
- Flag conflicting information explicitly
""", language="markdown")

    with c2:
        st.markdown("#### Dynamic Memory API")
        st.code("""from deepagents import create_deep_agent
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()   # swap for Redis/Postgres

agent = create_deep_agent(
    model=llm,
    store=store,
    # Agent can now call:
    # save_memory(key, value)
    # search_memories(query)
    # delete_memory(key)
)

# Thread isolation — one thread per user/session
config = {
    "configurable": {
        "thread_id": "user-42-session-7"
    }
}

agent.invoke(
    {"messages": [{"role":"user","content":"..."}]},
    config=config,
)

# Same store, different session → memories persist
agent.invoke(
    {"messages": [{"role":"user","content":"Continue from yesterday"}]},
    config={"configurable":{"thread_id":"user-42-session-8"}},
)
""", language="python")

    st.markdown("---")
    st.markdown("### Memory Store (simulated)")
    if "memories" not in st.session_state:
        st.session_state.memories = [
            {"key":"user/preferred_format",    "value":"Markdown with headers and bullet points","updated":"2026-05-28"},
            {"key":"user/writing_style",       "value":"Formal, cite sources, British English",  "updated":"2026-05-28"},
            {"key":"task/last_research_topic", "value":"Quantum computing hardware trends 2025", "updated":"2026-05-29"},
            {"key":"task/pending_todos",       "value":"['Review paper on photonic qubits']",     "updated":"2026-05-30"},
        ]
    for m in st.session_state.memories:
        a, b, c = st.columns([2,3,1])
        a.code(m["key"])
        b.markdown(f'<span style="font-size:.82rem;color:#94a3b8">{m["value"]}</span>', unsafe_allow_html=True)
        c.markdown(f'<span style="font-size:.72rem;color:#4b5563">{m["updated"]}</span>', unsafe_allow_html=True)
    st.markdown("#### Add a memory")
    mk = st.text_input("Key", value="user/timezone")
    mv = st.text_input("Value", value="Europe/London")
    if st.button("💾  Save Memory"):
        st.session_state.memories.append({"key":mk,"value":mv,"updated":str(datetime.date.today())})
        st.success("Memory saved ✓")
        st.rerun()


# ─── CONTEXT MANAGEMENT ───────────────────────────────────────────────────────
elif page == "✂️ Context Management":
    st.markdown('<div class="section-header" style="color:#fb923c">Context Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">AUTO-SUMMARISATION · LARGE RESULT EVICTION · PROMPT CACHING</div>', unsafe_allow_html=True)
    st.markdown("Deep Agents ships **three strategies** to keep the context window healthy on long tasks:")

    for title, desc, color in [
        ("Auto-summarisation",    "SummarizationMiddleware monitors tokens before each model call. When it exceeds a threshold, old messages are compressed into a summary and removed from the active window.", "#fb923c"),
        ("Large result eviction", "Tool results over a configurable size limit are stored in the virtual filesystem. The agent receives a pointer ('written to context/big_output.txt') instead of the raw content.", "#4f8ef7"),
        ("Subagent isolation",    "Each subagent runs in its own context window. The parent only receives the final ToolMessage, not the child's full conversation history.", "#a78bfa"),
    ]:
        st.markdown(f'<div class="da-card" style="border-left:3px solid {color}"><b style="color:{color}">{title}</b><br><span style="font-size:.82rem;color:#94a3b8">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Summarisation Config")
        st.code("""from deepagents import create_deep_agent
from langchain_groq import ChatGroq

llm       = ChatGroq(model="llama-3.3-70b-versatile")
cheap_llm = ChatGroq(model="llama3-8b-8192")

agent = create_deep_agent(
    model=llm,
    summarization_config={
        # Trigger when context > 70% of model limit
        "threshold_fraction": 0.70,

        # Keep last N messages verbatim
        "keep_recent": 6,

        # Cheaper model handles compression
        "model": cheap_llm,
    },
)

# Agent can also manually trigger:
# compact_conversation()
""", language="python")

    with c2:
        st.markdown("### Context Window Simulator")
        model_limit  = st.selectbox("Model context limit", [8192, 32768, 131072], index=2)
        total_tokens = st.slider("Tokens used", 0, model_limit, int(model_limit*0.72), step=500)
        threshold    = st.slider("Summarisation threshold (%)", 50, 95, 70)

        pct = (total_tokens / model_limit) * 100
        trigger = pct >= threshold

        a, b = st.columns(2)
        a.metric("Context used", f"{total_tokens:,}", f"{pct:.1f}%")
        b.metric("Threshold at", f"{int(threshold*model_limit/100):,}", f"{threshold}%")
        st.progress(min(pct/100, 1.0))

        if trigger:
            st.error(f"⚠️  {pct:.1f}% ≥ {threshold}% → **SummarizationMiddleware fires**\n\nCompressing messages[0:-6] → summary → context reduced to ~18%")
        elif pct > threshold * 0.85:
            st.warning(f"🟡  {pct:.1f}% — approaching threshold")
        else:
            st.success(f"🟢  {pct:.1f}% — context healthy")


# ─── MIDDLEWARE STACK ─────────────────────────────────────────────────────────
elif page == "🛠 Middleware Stack":
    st.markdown('<div class="section-header" style="color:#22d3ee">Middleware Stack</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">COMPOSABLE PIPELINE · LIFECYCLE HOOKS · CUSTOM MIDDLEWARE</div>', unsafe_allow_html=True)
    st.markdown("Deep Agents assembles a **pipeline of middleware** intercepting model calls and tool executions.")

    st.markdown("### Built-in Layers")
    for mw in MIDDLEWARE_LAYERS:
        st.markdown(f'<div class="mw-layer"><span style="font-size:1.2rem">{mw["icon"]}</span><div style="flex:1"><code style="color:{mw["color"]}">{mw["name"]}</code><span style="color:#6b7280;font-size:.72rem;margin-left:10px">hook: {mw["hook"]}</span><br><span style="color:#94a3b8;font-size:.8rem">{mw["desc"]}</span></div></div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Hooks Reference")
        st.code("""class AgentMiddleware(Protocol):
    # Called once before agent starts
    def before_agent(self, state): ...

    # Called before every model invocation
    def before_model(self, request): ...

    # Wraps the model call
    def wrap_model_call(self, request, handler): ...

    # Wraps tool execution (can block/modify)
    def wrap_tool_call(self, request, handler): ...

    # Called after every model response
    def after_model(self, response): ...

    # Called once after agent completes
    def after_agent(self, state): ...

    # Async variants also available:
    async def awrap_model_call(...): ...
    async def awrap_tool_call(...): ...
""", language="python")
    with c2:
        st.markdown("### Custom Middleware Example")
        st.code("""from langchain.agents.middleware.types import AgentMiddleware

class AuditMiddleware(AgentMiddleware):
    \"\"\"Log every tool call to a database.\"\"\"

    async def awrap_tool_call(self, request, handler):
        # Log BEFORE
        print(f"TOOL: {request.tool_name}")
        print(f"ARGS: {request.tool_input}")

        # Execute
        response = await handler(request)

        # Log AFTER
        print(f"RESULT: {response.content[:200]}")
        return response


class PIIRedactMiddleware(AgentMiddleware):
    \"\"\"Strip PII from all tool results.\"\"\"

    def wrap_tool_call(self, request, handler):
        response = handler(request)
        response.content = redact_pii(response.content)
        return response


# Apply
agent = create_deep_agent(
    model=llm,
    extra_middleware=[
        AuditMiddleware(),
        PIIRedactMiddleware(),
    ],
)
""", language="python")

    st.markdown("### Human-in-the-Loop Config")
    st.code("""agent = create_deep_agent(
    model=llm,
    hil_config={
        "pause_before": ["execute", "write_file"],  # tools requiring approval
        "timeout_seconds": 30,                       # None = wait forever
    },
)
# When agent calls `execute`:
# → HILMiddleware pauses execution
# → Waits for human: approve / modify / reject
# → Resumes with human-amended tool call
""", language="python")


# ─── SKILLS SYSTEM ────────────────────────────────────────────────────────────
elif page == "🧑 Skills System":
    st.markdown('<div class="section-header" style="color:#4f8ef7">Skills System</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">MARKDOWN SKILL FILES · PROGRESSIVE DISCLOSURE · AGENTS.MD</div>', unsafe_allow_html=True)
    st.markdown("""
Skills are **Markdown files** injected into the agent's system prompt on demand.
They encode reusable behaviours, domain knowledge, or procedures — without burning tokens by loading everything upfront.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Example: `skills/researcher.md`")
        st.code("""# Web Research Skill

## When to use
Use when asked to research a topic, gather
information, or fact-check claims.

## Process
1. Break query into 3-5 sub-questions
2. For each sub-question:
   - Search the web with a specific query
   - Read the top 2 results fully
   - Save notes to context/research_<topic>.md
3. Cross-reference sources for conflicts
4. Synthesise into a structured summary

## Output format
Save report to output/research_<topic>.md:
- Executive Summary (3 sentences)
- Key Findings (bullet list)
- Source table with URLs + reliability scores

## Quality checks
- At least 3 independent sources
- Flag any conflicting information
- Note recency of sources
""", language="markdown")

    with c2:
        st.markdown("#### Skills Configuration")
        st.code("""from deepagents import create_deep_agent
from deepagents.backends import LocalFilesystemBackend

agent = create_deep_agent(
    model=llm,
    # Skills auto-loaded from skills/ directory
    # in the backend root.
    backend=LocalFilesystemBackend(root="./agent_fs"),
)

# Progressive disclosure pattern:
# 1. AGENTS.md loaded at startup (identity)
# 2. Skill *names* listed in system prompt
# 3. Agent reads full skill file only when needed
#    → Saves tokens vs loading all skills upfront

# Skills directory:
# agent_fs/
# ├── AGENTS.md
# ├── skills/
# │   ├── researcher.md
# │   ├── coder.md
# │   ├── analyst.md
# │   └── writer.md
# ├── context/
# ├── output/
# └── memory/
""", language="python")

    st.markdown("---")
    st.markdown("### Skill Editor (simulated)")
    skill_name = st.text_input("Skill filename", value="custom_skill.md")
    skill_content = st.text_area("Skill content (Markdown)", height=200, value="""# My Custom Skill

## When to use
Describe when the agent should apply this skill.

## Process
1. Step one
2. Step two
3. Step three

## Output
Describe the expected output format.
""")
    if st.button("💾  Save Skill"):
        st.success(f"Skill saved to `skills/{skill_name}` ✓ — available to agent on next invocation")


# ─── LIVE AGENT DEMO ──────────────────────────────────────────────────────────
elif page == "🚀 Live Agent Demo":
    st.markdown('<div class="section-header" style="color:#34d399">Live Agent Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">REAL GROQ INFERENCE · ALL FEATURES ENABLED · STREAMING</div>', unsafe_allow_html=True)

    if not groq_key:
        st.warning("⚠️  Paste your **Groq API key** in the sidebar. Free key at: https://console.groq.com", icon="🔑")
        st.markdown("---")
        st.markdown("""
**When you provide a key, the app will:**

1. Build a real `create_deep_agent()` backed by Groq
2. Enable all middleware: planning, filesystem, subagents, summarisation
3. Stream the agent's execution trace token-by-token
4. Show tool calls, intermediate steps, and final output
""")

    presets = {
        "Research":  "Research the top 3 open-source LLM frameworks in 2026 and compare their strengths for building autonomous agents. Write a structured comparison.",
        "Coding":    "Write a Python implementation of a priority queue using a binary heap. Include docstrings, type hints, and 3 unit tests.",
        "Analysis":  "Analyse the trade-offs between in-context learning and fine-tuning for domain adaptation. Create a decision framework with 5 evaluation criteria.",
        "Custom":    "",
    }
    preset = st.selectbox("Quick preset", list(presets.keys()))
    user_task = st.text_area("Task (edit freely)", value=presets[preset], height=100)
    sys_prompt = st.text_area("System prompt", value="You are an expert AI assistant. Always plan before acting. Be thorough and cite your reasoning.", height=70)

    with st.expander("⚙️ Advanced options"):
        c1, c2 = st.columns(2)
        with c1:
            enable_planning  = st.checkbox("Planning (write_todos)",     value=True)
            enable_fs        = st.checkbox("Virtual filesystem",          value=True)
            enable_subagents = st.checkbox("Subagents",                   value=True)
        with c2:
            enable_summarise = st.checkbox("Auto-summarisation",          value=True)
            enable_memory    = st.checkbox("Long-term memory",            value=False)
            stream_mode      = st.checkbox("Stream output",               value=True)

    run_btn = st.button("🚀  Run Deep Agent", type="primary", disabled=not groq_key)
    output_ph = st.empty()

    if run_btn and groq_key:
        imports = _try_import()
        if not imports:
            st.error("**deepagents** and/or **langchain-groq** are not installed.\n\nRun:\n```bash\npip install deepagents langchain-groq\n```\nthen restart: `streamlit run app.py`")
        else:
            create_deep_agent, ChatGroq, tool = imports
            from langchain_core.tools import tool as lc_tool

            @lc_tool
            def calculator(expression: str) -> str:
                """Evaluate a Python math expression safely."""
                try:
                    return str(eval(expression, {"__builtins__": {}}, {}))
                except Exception as e:
                    return f"Error: {e}"

            @lc_tool
            def current_date() -> str:
                """Return today's date in ISO format."""
                return str(datetime.date.today())

            excluded = []
            if not enable_planning:
                from deepagents.middleware import TodoListMiddleware
                excluded.append(TodoListMiddleware)
            if not enable_fs:
                from deepagents.middleware import FilesystemMiddleware
                excluded.append(FilesystemMiddleware)
            if not enable_subagents:
                from deepagents.middleware import SubAgentMiddleware
                excluded.append(SubAgentMiddleware)
            if not enable_summarise:
                from deepagents.middleware import SummarizationMiddleware
                excluded.append(SummarizationMiddleware)

            llm = ChatGroq(model=model_choice, temperature=0, api_key=groq_key)
            kwargs = dict(model=llm, tools=[calculator, current_date], system_prompt=sys_prompt)
            if excluded:
                kwargs["excluded_middleware"] = excluded
            if enable_memory:
                from langgraph.store.memory import InMemoryStore
                kwargs["store"] = InMemoryStore()

            try:
                agent = create_deep_agent(**kwargs)
                full = ""
                with st.spinner("Agent running…"):
                    if stream_mode:
                        for chunk in agent.stream({"messages":[{"role":"user","content":user_task}]}):
                            for msg in chunk.get("messages", []):
                                if hasattr(msg, "content") and isinstance(msg.content, str):
                                    full += msg.content
                                    output_ph.markdown(f'<div class="trace-box">{full}</div>', unsafe_allow_html=True)
                    else:
                        result = agent.invoke({"messages":[{"role":"user","content":user_task}]})
                        full = "\n\n".join(m.content for m in result.get("messages",[]) if hasattr(m,"content") and m.content)
                        output_ph.markdown(f'<div class="trace-box">{full}</div>', unsafe_allow_html=True)
                st.success("✅ Agent completed")
            except Exception as e:
                st.error(f"Agent error: {e}")
    elif not run_btn:
        output_ph.markdown('<div class="trace-box" style="color:#4b5563">Agent output will stream here…</div>', unsafe_allow_html=True)


# ─── CODE REFERENCE ───────────────────────────────────────────────────────────
elif page == "📦 Code Reference":
    st.markdown('<div class="section-header" style="color:#4f8ef7">Code Reference</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">COPY-PASTE READY · ALL PATTERNS · GROQ COMPATIBLE</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Quickstart","🔀 Subagents","📁 Filesystem","🛠 Middleware","🏭 Production"])

    with tab1:
        st.code("""# pip install deepagents langchain-groq
import os
from deepagents import create_deep_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool

os.environ["GROQ_API_KEY"] = "gsk_..."

@tool
def calculator(expression: str) -> str:
    \"\"\"Evaluate a Python math expression.\"\"\"
    return str(eval(expression))

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

agent = create_deep_agent(
    model=llm,
    tools=[calculator],
    system_prompt="You are a helpful assistant. Always plan before acting.",
)

# Invoke
result = agent.invoke({"messages": [{"role":"user","content":"What is 42 * 1337?"}]})
print(result["messages"][-1].content)

# Stream
for chunk in agent.stream({"messages": [{"role":"user","content":"Research deep learning"}]}):
    for msg in chunk.get("messages", []):
        if hasattr(msg, "content"):
            print(msg.content, end="", flush=True)
""", language="python")

    with tab2:
        st.code("""from deepagents import create_deep_agent, SubAgent
from langchain_groq import ChatGroq

llm  = ChatGroq(model="llama-3.3-70b-versatile")
fast = ChatGroq(model="llama3-8b-8192")  # cheaper model for subtasks

agent = create_deep_agent(
    model=llm,
    subagents=[
        SubAgent(
            name="researcher",
            model=llm,
            tools=[web_search, read_file, write_file],
            system_prompt="You are a web researcher.",
        ),
        SubAgent(
            name="coder",
            model=fast,
            tools=[execute, write_file, edit_file],
            system_prompt="You are a software engineer.",
        ),
    ],
)
# Parent delegates via: task(subagent_type="researcher", prompt="...")
""", language="python")

    with tab3:
        st.code("""from deepagents import create_deep_agent
from deepagents.backends import (
    InMemoryBackend,           # default
    LocalFilesystemBackend,
    S3Backend,
    CompositeBackend,
)

# Local disk
agent = create_deep_agent(
    model=llm,
    backend=LocalFilesystemBackend(root="./workspace"),
)

# S3 (production)
agent = create_deep_agent(
    model=llm,
    backend=S3Backend(bucket="my-bucket", prefix="runs/"),
)

# Composite routing
agent = create_deep_agent(
    model=llm,
    backend=CompositeBackend(
        memory=InMemoryBackend(),
        skills=LocalFilesystemBackend("./skills"),
        output=S3Backend(bucket="results"),
    ),
)
""", language="python")

    with tab4:
        st.code("""from langchain.agents.middleware.types import AgentMiddleware

class CostTrackerMiddleware(AgentMiddleware):
    def __init__(self):
        self.total_tokens = 0

    def after_model(self, response):
        if hasattr(response, "usage_metadata"):
            self.total_tokens += response.usage_metadata.get("total_tokens", 0)
        return response

tracker = CostTrackerMiddleware()

agent = create_deep_agent(
    model=llm,
    extra_middleware=[tracker],
    excluded_middleware=[HILMiddleware],   # disable specific layers
    hil_config={
        "pause_before": ["execute", "write_file"],
        "timeout_seconds": 30,
    },
    summarization_config={
        "threshold_fraction": 0.75,
        "keep_recent": 4,
        "model": ChatGroq(model="llama3-8b-8192"),
    },
)
""", language="python")

    with tab5:
        st.code("""import os
from deepagents import create_deep_agent
from deepagents.backends import S3Backend
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.redis import RedisStore

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "ls_..."

llm = ChatGroq(model="llama-3.3-70b-versatile")

agent = create_deep_agent(
    model=llm,
    backend=S3Backend(bucket="prod-artifacts"),
    checkpointer=PostgresSaver.from_conn_string("postgresql://..."),
    store=RedisStore(url="redis://..."),
    summarization_config={"threshold_fraction": 0.70},
)

# Thread isolation per user
config = {"configurable": {"thread_id": "user-42-session-7"}}
result = agent.invoke(
    {"messages": [{"role":"user","content":"Continue from yesterday"}]},
    config=config,
)
""", language="python")
