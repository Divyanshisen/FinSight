import os
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import crag, baseline_rag
from charts import extract_chart, build_figure


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CRAG Finance",
    page_icon="💹",
    layout="wide"
)


# =========================================================
# DYNAMIC PROFESSIONAL NAVY GRADIENT THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --navy-primary: #0A192F;
    --navy-deep: #020C1B;
    --navy-mid: #112240;
    --navy-light: #233554;
    --navy-gradient: linear-gradient(135deg, #0A192F 0%, #112240 50%, #1A365D 100%);
    --btn-gradient: linear-gradient(135deg, #002D62 0%, #001A3D 50%, #0A192F 100%);
    --btn-hover-gradient: linear-gradient(135deg, #003B80 0%, #002855 50%, #112240 100%);
    --gold: #C5A16B;
    --gold-bright: #F3E5AB;
    --slate-text: #E6F1FF;
    --muted-text: #8892B0;
    --ink: #18263D;
    --white-card: rgba(255, 255, 255, 0.95);
}

.stApp {
    background: radial-gradient(circle at 88% 4%, rgba(20, 40, 75, 0.25), transparent 35%),
                linear-gradient(135deg, #F0F4F8 0%, #E2E8F0 50%, #EDF2F7 100%);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

/* Header customization */
[data-testid="stHeader"] {
    background: var(--navy-primary) !important;
    border-bottom: 1px solid rgba(255,255,255,0.12);
}
[data-testid="stHeader"] * { color: #FFFFFF !important; }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stDecoration"] { background: var(--gold); }

.block-container {
    max-width: 1420px;
    padding-top: 2rem;
    padding-bottom: 3.5rem;
}

/* Headings with Navy Accent */
h1, h2, h3, h4, h5, h6 {
    color: var(--navy-primary) !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -.025em;
}
h1 { line-height: 1.15 !important; }
h2 { margin-top: .35rem !important; }
h3 { font-size: 1.25rem !important; }

.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] {
    color: var(--ink);
}
.stMarkdown p, .stMarkdown li { line-height: 1.72; font-size: .98rem; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
    color: #4A5568 !important;
    font-size: .88rem;
}
label, .stRadio label, .stCheckbox label, .stToggle label,
[data-testid="stWidgetLabel"] * { 
    color: var(--navy-primary) !important; 
    font-weight: 700 !important;
}

/* Premium Navy Gradient Sidebar */
[data-testid="stSidebar"] {
    background: var(--navy-gradient) !important;
    border-right: 1px solid rgba(255,255,255,0.12);
}
[data-testid="stSidebar"] * { color: var(--slate-text) !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: var(--muted-text) !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: var(--slate-text) !important;
    font-weight: 650;
    padding: .45rem .6rem;
    border-radius: 10px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: .35rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(255,255,255,0.12);
    border-left: 3px solid var(--gold);
}

/* Hero Section with Dynamic Navy Overlay */
.hero {
    position: relative;
    overflow: hidden;
    isolation: isolate;
    padding: clamp(2rem, 4vw, 3.5rem);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 24px;
    background: linear-gradient(115deg, rgba(2, 12, 27, 0.95) 0%, rgba(10, 25, 47, 0.90) 50%, rgba(23, 42, 69, 0.85) 100%),
                url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=1800&q=85') center/cover;
    margin-bottom: 1.8rem;
    box-shadow: 0 20px 40px rgba(2, 12, 27, 0.25);
    animation: heroReveal .75s cubic-bezier(.2,.75,.25,1) both;
}
.hero::before {
    content: "";
    position: absolute;
    z-index: -1;
    left: 0;
    top: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, var(--gold), rgba(197, 161, 107, 0.2) 70%, transparent);
}
.eyebrow-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    margin: 0 8px 1px 0;
    border-radius: 50%;
    background: var(--gold);
    box-shadow: 0 0 0 4px rgba(197, 161, 107, 0.25);
}
.eyebrow {
    display: inline-block;
    text-transform: uppercase; letter-spacing: 2px;
    font-size: .78rem; font-weight: 800;
    color: var(--gold-bright) !important;
    margin-bottom: .45rem;
}
.hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2rem, 4.2vw, 3.2rem);
    font-weight: 800; line-height: 1.16;
    color: #FFFFFF !important;
    letter-spacing: -.035em; margin: .55rem 0 .75rem;
    max-width: 900px;
}
.hero-copy {
    color: var(--slate-text) !important;
    font-size: 1.06rem; line-height: 1.8; max-width: 820px;
}
.badge {
    display: inline-block; padding: 6px 15px; border-radius: 999px;
    font-weight: 800; font-size: .82rem; margin-bottom: .6rem;
}

/* Translucent Cards & Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--white-card) !important;
    border: 1px solid rgba(255,255,255,0.9) !important;
    border-radius: 20px !important;
    box-shadow: 0 12px 32px rgba(10, 25, 47, 0.08);
    backdrop-filter: blur(16px);
}
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E0 !important;
    border-radius: 16px !important;
    padding: 1.05rem 1.15rem !important;
    box-shadow: 0 4px 15px rgba(10, 25, 47, 0.05);
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    color: var(--muted-text) !important; font-weight: 700 !important; font-size: .84rem !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    color: var(--navy-primary) !important; font-family: 'Manrope', sans-serif !important; font-weight: 800 !important;
}

/* Text Input & Text Area Styling */
.stTextArea textarea, .stTextInput input {
    background: #FFFFFF !important;
    color: var(--navy-primary) !important;
    -webkit-text-fill-color: var(--navy-primary) !important;
    border: 1.5px solid #CBD5E0 !important;
    border-radius: 13px !important;
    font-size: 1rem !important; line-height: 1.65 !important;
    padding: .75rem .9rem !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: #A0AEC0 !important; -webkit-text-fill-color: #A0AEC0 !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border: 1.5px solid var(--navy-mid) !important;
    box-shadow: 0 0 0 3px rgba(17, 34, 64, 0.15) !important;
}

/* BaseWeb Select Styling */
[data-baseweb="select"] > div {
    background: #FFFFFF !important; border-color: #CBD5E0 !important; border-radius: 12px !important;
}
[data-baseweb="select"] * { color: var(--navy-primary) !important; font-weight: 600; }

/* PROMINENT NAVY GRADIENT BUTTON (Get Answer) */
div.stButton > button[kind="primary"] {
    background: var(--btn-gradient) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--navy-light) !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.5rem !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.5px;
    box-shadow: 0 10px 25px rgba(10, 25, 47, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div.stButton > button[kind="primary"] p {
    color: #FFFFFF !important;
}
div.stButton > button[kind="primary"]:hover {
    background: var(--btn-hover-gradient) !important;
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(10, 25, 47, 0.4) !important;
}

/* Secondary Buttons */
div.stButton > button:not([kind="primary"]), div.stDownloadButton > button {
    background: #FFFFFF !important;
    color: var(--navy-primary) !important;
    border: 1px solid #CBD5E0 !important;
    border-radius: 12px !important;
    font-weight: 650 !important;
    transition: all .2s ease;
}
div.stButton > button:not([kind="primary"]):hover, div.stDownloadButton > button:hover {
    background: #EDF2F7 !important;
    border-color: var(--navy-mid) !important;
    transform: translateY(-1px);
}

/* Expanders & Tabs */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important; border-radius: 14px !important;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
    color: var(--navy-primary) !important; font-weight: 700;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: #E2E8F0; padding: 6px;
    border-radius: 14px;
}
.stTabs [data-baseweb="tab"] { color: #4A5568 !important; font-weight: 700; border-radius: 10px; }
.stTabs [aria-selected="true"] { 
    background: var(--navy-primary) !important; 
    color: #FFFFFF !important; 
    box-shadow: 0 4px 12px rgba(10, 25, 47, 0.2); 
}
.stTabs [aria-selected="true"] * { color: #FFFFFF !important; }

.src-link {
    display: block; padding: 10px 13px; margin: 5px 0; border-radius: 10px;
    background: #FFFFFF; border: 1px solid #CBD5E0; color: #1A365D !important;
    font-size: .92rem; word-break: break-all; text-decoration: none; font-weight: 600;
}
.src-link:hover { border-color: #4A5568; background: #F7FAFC; }

@media (max-width: 768px) {
    .block-container { padding: 1rem .85rem 2rem; }
    .hero { padding: 1.5rem; border-radius: 17px; }
    .hero-title { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CONSTANTS & SAMPLE QUESTIONS (COVERING VARIOUS CONCEPTS)
# =========================================================
FENCE = "`" * 3
CHART_MARK = FENCE + "chart"

ACTION_INFO = {
    "CORRECT": ("🟢", "The retrieval evaluator found relevant evidence.", ("#0B7A3E", "#E3F6EC")),
    "INCORRECT": ("🔴", "The retrieved evidence was judged not relevant.", ("#B42318", "#FDE8E6")),
    "AMBIGUOUS": ("🟠", "The evidence or retrieval result is uncertain.", ("#9A6700", "#FFF4D6")),
    "CALCULATOR": ("🧮", "The answer was calculated by code.", ("#0F4C81", "#E4EEF8")),
    "REFUSED": ("🛡️", "The system declined to guess without sufficient evidence.", ("#475467", "#EEF1F5")),
}

# 9 Sample questions covering different finance concepts & query types
SAMPLE_QUESTIONS = [
    "What is a mutual fund?",
    "What is the current RBI repo rate?",
    "What is the difference between CRR and SLR?",
    "How is capital gains tax calculated on mutual funds in India?",
    "Compare Infosys key performance highlights in a bar chart: revenue growth, adjusted operating margin, return on equity, dividend per share growth and EPS growth.",
    "Show a pie chart of Infosys revenue distribution by geographical segments.",
    "What is a mutual fund, and what is the current RBI repo rate?",
    "Explain Infosys FY2026 revenue growth and adjusted operating margin.",
    "What is my personal bank account balance?"
]


# =========================================================
# ANSWER AND CHART FUNCTIONS
# =========================================================
def style_chart(fig):
    """Apply the shared CRAG Finance chart theme without changing chart data."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", size=13, color="#1A202C"),
        title=dict(font=dict(family="Manrope, sans-serif", size=19, color="#0A192F")),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12, color="#2D3748")
        ),
        margin=dict(l=45, r=25, t=65, b=45),
        hoverlabel=dict(
            bgcolor="#0A192F",
            bordercolor="#0A192F",
            font=dict(color="#FFFFFF", family="DM Sans, sans-serif")
        )
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#CBD5E0",
        tickfont=dict(color="#4A5568", size=12),
        title_font=dict(color="#2D3748", size=13)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(203, 213, 224, 0.4)",
        zeroline=False,
        linecolor="#CBD5E0",
        tickfont=dict(color="#4A5568", size=12),
        title_font=dict(color="#2D3748", size=13)
    )
    return fig


def show_plotly(fig, key):
    style_chart(fig)
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except TypeError:
        st.plotly_chart(fig, width="stretch", key=key)


def show_answer(answer, key_prefix="ans"):
    answer = answer or ""
    clean_text, chart_spec = extract_chart(answer)

    if clean_text:
        st.markdown(clean_text)

    if chart_spec is not None:
        try:
            fig = build_figure(chart_spec)
            show_plotly(fig, key=f"{key_prefix}_chart")

            if isinstance(chart_spec, dict) and chart_spec.get("source"):
                st.caption(f"Chart source: {chart_spec['source']}")

        except Exception as exc:
            st.warning("The chart could not be displayed.")
            with st.expander("Chart details"):
                st.write(str(exc))
                st.json(chart_spec)

    elif CHART_MARK in answer:
        st.warning("A chart was generated but its data could not be read.")


# =========================================================
# VERIFICATION AND SOURCES
# =========================================================
def show_verification(result):
    verification = result.get("verification") or {}
    status = verification.get("status", "")
    items = ", ".join(map(str, verification.get("unverified", []) or []))

    if status == "calculated":
        st.info("🧮 Answer calculated by code.")
    elif status == "verified" and verification.get("checked", 0) > 0:
        st.success(f"✅ Figure check passed: {verification['checked']} figure(s) checked.")
    elif status == "verified":
        st.success("✅ Verification returned a verified status.")
    elif status == "partial":
        st.warning(
            "⚠️ Part of the answer was verified. Lines with figures that "
            "could not be confirmed in the sources were removed."
            + (f" Removed figures: {items}" if items else "")
        )
    elif status == "check":
        st.warning("⚠️ Some details need checking. " + (f"Details: {items}" if items else ""))
    elif status == "uncited":
        st.warning("⚠️ Numbered citations were not detected. Check important claims against official sources.")
    elif status == "refused":
        st.info("🛡️ The system refused to guess.")
    else:
        st.info("ℹ️ Verification status is unavailable.")


def show_sources(result):
    st.markdown("#### Sources and evidence")

    knowledge = result.get("knowledge") or []

    if knowledge:
        for i, item in enumerate(knowledge, 1):
            if not isinstance(item, dict):
                st.write(item)
                continue

            source = str(item.get("source", "Unknown source"))

            with st.expander(f"[{i}] {source}", expanded=(i == 1)):
                if source.startswith("http"):
                    st.markdown(
                        f'<a class="src-link" href="{source}" target="_blank">🔗 Open source page</a>',
                        unsafe_allow_html=True
                    )

                if item.get("text"):
                    st.write(item["text"])

                image_path = item.get("image")
                if image_path and os.path.exists(image_path):
                    st.image(image_path, caption=f"Source [{i}] page image")

    elif result.get("sources"):
        for i, source in enumerate(result["sources"], 1):
            source = str(source)
            if source.startswith("http"):
                st.markdown(
                    f'<a class="src-link" href="{source}" target="_blank">[{i}] 🔗 {source}</a>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"**[{i}]** `{source}`")
    else:
        st.info("No sources were returned by the pipeline.")


# =========================================================
# RENDER PIPELINE RESULTS
# =========================================================
def render_crag(result, seconds=None):
    action = str(result.get("action", "UNKNOWN")).upper()
    icon, description, (color, bg) = ACTION_INFO.get(
        action, ("⚪", "Retrieval status is unavailable.", ("#475467", "#EEF1F5"))
    )

    st.markdown("### 🧠 Corrective RAG")
    st.markdown(
        f'<span class="badge" style="color:{color};background:{bg};">{icon} {action}</span>',
        unsafe_allow_html=True
    )
    st.caption(description)

    scores = result.get("scores") or []
    sources = result.get("sources") or []

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Retrieval decision", f"{icon} {action}")
    with col2:
        try:
            top_score = max(float(s) for s in scores)
            st.metric("Top relevance score", f"{top_score:.2f}")
        except (ValueError, TypeError):
            st.metric("Top relevance score", "N/A")
    with col3:
        st.metric("Response time", f"{seconds:.0f}s" if seconds is not None else "N/A")
    with col4:
        st.metric("Sources used", len(sources))

    parts = result.get("parts") or []
    sub_questions = result.get("sub_questions") or []

    if parts or len(sub_questions) > 1:
        with st.expander("Question breakdown"):
            for part in parts:
                part_scores = part.get("scores") or []
                try:
                    part_score = f"{max(float(s) for s in part_scores):.2f}"
                except (ValueError, TypeError):
                    part_score = "N/A"
                st.markdown(
                    f"- **{part.get('action', 'UNKNOWN')}** (score {part_score}): {part.get('question', '')}"
                )
            if not parts:
                st.write(sub_questions)

    st.markdown("#### Answer")
    answer_text = result.get("answer", "")
    show_answer(answer_text, key_prefix="crag")

    clean_text, _ = extract_chart(answer_text or "")
    if clean_text:
        st.download_button(
            "⬇️ Download answer (.txt)",
            data=clean_text,
            file_name="finance_answer.txt",
            mime="text/plain"
        )

    show_verification(result)
    show_sources(result)

    with st.expander("Advanced pipeline details"):
        st.markdown("**Retrieval scores**")
        st.write(scores if scores else "No scores available.")
        st.markdown("**Web search query**")
        st.code(result.get("web_query") or "No external web search query returned.")

        for title, key in [("Internal knowledge", "k_in"), ("External knowledge", "k_ex")]:
            st.markdown(f"**{title}**")
            items = result.get(key) or []
            if items:
                for item in items:
                    if isinstance(item, dict):
                        st.markdown(f"*{item.get('source', 'Source')}*")
                        st.write(item.get("text", ""))
                    else:
                        st.write(item)
            else:
                st.write("No knowledge returned.")


def render_normal(result):
    st.markdown("### 📚 Normal RAG")
    st.caption("Baseline retrieval without the corrective workflow.")

    st.markdown("#### Answer")
    show_answer(result.get("answer", ""), key_prefix="normal")

    st.markdown("#### Sources")
    sources = result.get("sources") or []
    if sources:
        for source in sources:
            st.markdown(f"- `{source}`")
    else:
        st.info("No source list was returned by the pipeline.")


# =========================================================
# INSIGHTS PAGE
# =========================================================
def _load_results(path):
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    lab = df["label"].astype(str).str.strip().str.lower()
    auto = df["auto_label"].astype(str).str.strip().str.lower()
    df["final"] = lab.where(lab.isin(["c", "w", "r"]), auto.where(auto == "r", ""))
    return df


def _metrics(d):
    n = len(d)
    c = int((d["final"] == "c").sum())
    r = int((d["final"] == "r").sum())
    ans = n - r
    return {"N": n, "Correct": c, "Refused": r,
            "Accuracy": c / n * 100 if n else 0,
            "Reliability": c / ans * 100 if ans else 0,
            "Hallucination": (ans - c) / n * 100 if n else 0}


def render_insights():
    import glob
    st.markdown("""
    <div class="hero">
        <div class="eyebrow"><span class="eyebrow-dot"></span> EVALUATION WORKSPACE</div>
        <div class="hero-title">RAG vs CRAG Insights</div>
        <div class="hero-copy">Real-time benchmark evaluation and analytics across retrieval models.</div>
    </div>
    """, unsafe_allow_html=True)

    files = sorted(glob.glob("results*.csv"))
    if not files:
        st.info("No results*.csv found. Run evaluate.py first.")
        return
    default = files.index("results_final.csv") if "results_final.csv" in files else len(files) - 1
    path = st.selectbox("Results file", files, index=default)
    df = _load_results(path)

    graded = df[df["final"] != ""]
    if len(graded) < len(df):
        st.warning(f"{len(df) - len(graded)} row(s) are not graded yet and are excluded.")
    ansr = graded[graded["answerable"] == "yes"]

    rows = []
    for system, g in ansr.groupby("system"):
        rows.append({"System": system, **_metrics(g)})
    if not rows:
        st.info("No graded answerable rows yet.")
        return
    summary = pd.DataFrame(rows)
    st.markdown("### Answerable questions (Accuracy, Reliability, Hallucination)")
    st.dataframe(summary.round(1), use_container_width=True, hide_index=True)

    fig = px.bar(summary, x="System", y="Accuracy", color="System", text=summary["Accuracy"].round(1).astype(str) + "%",
                 title="Answer accuracy")
    fig.update_layout(showlegend=False, yaxis_range=[0, 100], yaxis_title="Accuracy (%)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    style_chart(fig)
    fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

    t = []
    for (typ, system), g in ansr.groupby(["type", "system"]):
        m = _metrics(g)
        t.append({"Question type": typ, "System": system, "Correct": m["Correct"], "Total": m["N"],
                  "Accuracy": m["Accuracy"]})
    if t:
        tdf = pd.DataFrame(t)
        fig2 = px.bar(tdf, x="Question type", y="Accuracy", color="System", barmode="group",
                      text=tdf["Correct"].astype(str) + "/" + tdf["Total"].astype(str),
                      title="Accuracy by question type",
                      color_discrete_sequence=["#0A192F", "#C5A16B", "#4A5568"])
        fig2.update_layout(yaxis_range=[0, 110], yaxis_title="Accuracy (%)",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        style_chart(fig2)
        fig2.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
        st.plotly_chart(fig2, use_container_width=True)

    un = graded[graded["answerable"] == "no"]
    if len(un):
        st.markdown("### Unanswerable / safety questions (correct refusals)")
        st.dataframe(un.groupby("system")["final"].agg(
            Refused=lambda x: int((x == "r").sum()), Total="count").reset_index(),
            use_container_width=True, hide_index=True)

    crag_rows = df[df["system"] == "crag"]
    sec = pd.to_numeric(crag_rows["seconds"], errors="coerce").dropna()
    if len(sec):
        st.caption(f"CRAG average time per answer: {sec.mean():.1f} s over {len(sec)} questions.")
    ok = crag_rows["source_ok"].astype(str).str.lower()
    chk = ok[ok.isin(["y", "n"])]
    st.caption(f"Citation support: {int((chk == 'y').sum())}/{len(chk)} audited rows."
               if len(chk) else "Citation support: not audited yet.")


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
if "history" not in st.session_state:
    st.session_state["history"] = []

if "finance_question" not in st.session_state:
    st.session_state["finance_question"] = ""

with st.sidebar:
    st.markdown("## 💹 CRAG Finance")
    st.caption("Finance research & answer verification")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Ask Finance", "Insights"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### How to use")
    st.markdown("""
    1. Select a sample question or type your own.
    2. Click **🔎 Get Answer**.
    3. Review the answer and verification.
    4. Expand sources to inspect the evidence.
    """)

    if st.session_state["history"]:
        st.divider()
        st.markdown("### Recent questions")
        for i, past in enumerate(st.session_state["history"]):
            label = past if len(past) <= 42 else past[:40] + "..."
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state["finance_question"] = past

    st.caption("Educational use only — not financial advice.")


# =========================================================
# ASK FINANCE PAGE
# =========================================================
if page == "Ask Finance":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow"><span class="eyebrow-dot"></span> CRAG FINANCE</div>
        <div class="hero-title">Finance Research Assistant</div>
        <div class="hero-copy">
            Ask financial questions, compare retrieval models, and review verified sources using Corrective RAG.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("## Ask a question")

        # Compact dropdown for sample questions across concepts
        selected_sample = st.selectbox(
            "Try a sample question",
            options=["-- Select a sample question --"] + SAMPLE_QUESTIONS,
            index=0
        )

        # Sync dropdown selection with text area automatically
        if selected_sample != "-- Select a sample question --":
            st.session_state["finance_question"] = selected_sample

        question = st.text_area(
            "Your question",
            key="finance_question",
            placeholder="For example: What is the difference between CRR and SLR?",
            height=120
        )

        col_toggle, col_space = st.columns([1, 1])
        with col_toggle:
            compare = st.toggle(
                "Compare with Normal RAG",
                value=False,
                help="Also runs the baseline pipeline for comparison."
            )

        ask = st.button(
            "🔎 Get Answer",
            type="primary",
            use_container_width=True
        )

    # Execution Flow
    if ask:
        if not question.strip():
            st.warning("Please enter or select a question first.")
        else:
            try:
                start_time = time.time()

                with st.spinner("Searching documents, checking sources and verifying figures..."):
                    crag_result = crag(question.strip())

                crag_seconds = time.time() - start_time
                normal_result = None

                if compare:
                    with st.spinner("Preparing the Normal RAG answer..."):
                        normal_result = baseline_rag(question.strip())

                st.session_state["last_question"] = question.strip()
                st.session_state["last_crag_result"] = crag_result
                st.session_state["last_normal_result"] = normal_result
                st.session_state["last_compare"] = compare
                st.session_state["last_seconds"] = crag_seconds

                history = st.session_state["history"]
                q_clean = question.strip()
                if q_clean in history:
                    history.remove(q_clean)
                history.insert(0, q_clean)
                del history[8:]

            except Exception as exc:
                st.error("The question could not be processed.")
                st.caption("Check API keys, internet connection, and pipeline dependencies.")
                with st.expander("Technical error details"):
                    st.code(str(exc))

    # Display Results
    if st.session_state.get("last_crag_result") is not None:
        st.divider()

        st.markdown(f"## Results for: *\"{st.session_state.get('last_question', '')}\"*")

        result = st.session_state["last_crag_result"]
        normal_result = st.session_state.get("last_normal_result")
        seconds = st.session_state.get("last_seconds")

        if st.session_state.get("last_compare") and normal_result is not None:
            crag_tab, rag_tab = st.tabs(["🧠 Corrective RAG", "📚 Normal RAG"])

            with crag_tab:
                render_crag(result, seconds)

            with rag_tab:
                render_normal(normal_result)
        else:
            render_crag(result, seconds)

else:
    render_insights()


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
---
*CRAG Finance is an educational research tool, not a financial adviser.
Verify important information using official sources such as RBI and SEBI.*
""")