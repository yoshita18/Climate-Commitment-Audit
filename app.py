"""
Streamlit UI — Climate Commitment Auditing Pipeline
====================================================
Upload a corporate climate report (PDF or TXT) and get a full
AI-powered audit against IPCC AR6 benchmarks in seconds.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Page config (must be the very first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Climate Commitment Auditor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .metric-label { font-size: 0.78rem; color: #9aa5b1; margin-bottom: 4px; }
    .metric-value { font-size: 1.9rem; font-weight: 700; }
    .grade-A { color: #2ecc71; }
    .grade-B { color: #3498db; }
    .grade-C { color: #f1c40f; }
    .grade-D { color: #e67e22; }
    .grade-F { color: #e74c3c; }
    .pill {
        display: inline-block; padding: 2px 10px;
        border-radius: 12px; font-size: 0.78rem; font-weight: 600;
    }
    .pill-green  { background: #1a3a2a; color: #2ecc71; }
    .pill-red    { background: #3a1a1a; color: #e74c3c; }
    .pill-yellow { background: #3a3010; color: #f1c40f; }
    .pill-blue   { background: #1a2a3a; color: #3498db; }
    .section-header {
        font-size: 1.05rem; font-weight: 700;
        border-bottom: 2px solid #2d3250;
        padding-bottom: 6px; margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Session-state initialisation  (runs once per browser session)
# ═══════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "report_path":   None,   # Path to file being audited
        "source_label":  "",     # Display name for that file
        "audit_result":  None,   # BenchmarkResult from last run
        "audit_elapsed": 0.0,    # Wall-clock time for last audit
        "audit_provider":"",     # Which provider was used
        "audit_error":   None,   # Error string if last run failed
        "running":       False,  # Guard against double-submission
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _score_bar(value: float, width: int = 220) -> str:
    pct = int(value * 100)
    colour = "#2ecc71" if value >= 0.75 else "#f1c40f" if value >= 0.5 else "#e74c3c"
    return (
        f'<div style="background:#2d3250;border-radius:6px;width:{width}px;height:10px;">'
        f'<div style="background:{colour};width:{pct}%;height:10px;border-radius:6px;"></div>'
        f'</div>'
    )

def _pill(text: str, kind: str = "blue") -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'

def _save_upload(upload) -> Path:
    """Save upload to a temp file named after the original file (not a random hash)."""
    suffix = Path(upload.name).suffix
    # Use the original stem so DocumentLoader infers the company name correctly
    safe_stem = re.sub(r"[^\w\-]", "_", Path(upload.name).stem)[:60]
    tmp_dir = Path(tempfile.gettempdir())
    dest = tmp_dir / f"{safe_stem}{suffix}"
    dest.write_bytes(upload.read())
    return dest

def _clear_result():
    st.session_state["audit_result"]  = None
    st.session_state["audit_error"]   = None
    st.session_state["audit_elapsed"] = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Cached pipeline builder  (embedding model + IPCC index persist per session)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _get_benchmarker(provider: str, api_key: str, backend: str):
    """
    Build the benchmarker once per (provider, api_key, backend) combo and
    cache it for the lifetime of the Streamlit server process.
    Embedding model loading + IPCC benchmark indexing only happen once.
    """
    from src.evaluation.benchmarker import PipelineBenchmarker
    from src.rag.embeddings import EmbeddingModel
    from src.rag.vector_store import VectorStore
    from src.rag.faiss_store import FaissVectorStore
    from src.rag.retriever import Retriever
    from src.audit.consistency_checker import ConsistencyChecker
    from src.adversarial.adversarial_prompter import AdversarialPrompter
    from config import get_settings

    if provider == "demo":
        from src.llm.mock import MockLLMClient
        llm = MockLLMClient()
    else:
        from src.llm.factory import get_llm_client
        llm = get_llm_client(provider=provider, api_key=api_key or None)

    settings = get_settings()
    emb = EmbeddingModel(settings.embedding_model)

    vs = (
        FaissVectorStore(settings.faiss_persist_dir, emb)
        if backend == "faiss"
        else VectorStore(settings.chroma_persist_dir, emb)
    )

    retriever = Retriever(vs, top_k=settings.top_k_retrieval)
    bm = PipelineBenchmarker(llm_client=llm)
    bm.vector_store = vs
    bm.retriever = retriever
    bm.consistency_checker = ConsistencyChecker(llm, retriever)
    bm.adversarial_prompter = AdversarialPrompter(llm, retriever)
    bm.setup()   # index IPCC AR6 benchmarks into vector store
    return bm


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🌍 Climate Auditor")
    st.caption("RAG · FAISS · LangChain · IPCC AR6")
    st.divider()

    st.subheader("⚙️ Configuration")
    provider = st.selectbox(
        "LLM Provider",
        options=["demo", "groq", "huggingface"],
        format_func=lambda x: {
            "demo":        "🎮 Demo (no API key)",
            "groq":        "⚡ Groq — Llama 3.3-70B (FREE)",
            "huggingface": "🤗 HuggingFace — Mistral-7B (FREE)",
        }[x],
        index=0,
    )

    api_key = ""
    if provider != "demo":
        key_label = {"groq": "GROQ_API_KEY", "huggingface": "HF_API_TOKEN"}[provider]
        api_key = st.text_input(
            f"API Key (`{key_label}`)",
            type="password",
            placeholder=f"Paste your {key_label}…",
            help={
                "groq":        "Free key at https://console.groq.com — no credit card",
                "huggingface": "Free token at https://huggingface.co/settings/tokens",
            }[provider],
        )
        if not api_key:
            st.warning(f"Enter your {key_label} to run a live audit.", icon="🔑")

    backend = st.selectbox(
        "Vector Store",
        options=["faiss", "chroma"],
        format_func=lambda x: {"faiss": "🗄 FAISS", "chroma": "🎨 ChromaDB"}[x],
        index=0,
    )

    st.divider()
    st.caption("v1.0 · IPCC AR6 · Apache 2.0")


# ═══════════════════════════════════════════════════════════════════════════
# Main — header
# ═══════════════════════════════════════════════════════════════════════════

st.title("🌍 Climate Commitment Auditor")
st.markdown(
    "Upload a corporate climate report (**PDF or TXT**) and get an AI-powered audit "
    "against **IPCC AR6** benchmarks — factual accuracy, greenwashing indicators, "
    "policy alignment, adversarial robustness, and an A–F grade."
)

if provider == "demo":
    st.info(
        "**Demo mode** — results use a built-in mock LLM (instant, no API key). "
        "Switch to Groq or HuggingFace in the sidebar for live analysis.",
        icon="🎮",
    )

st.divider()


# ═══════════════════════════════════════════════════════════════════════════
# File selection  — stored in session_state so it survives the "Run" click
# ═══════════════════════════════════════════════════════════════════════════

col_up, col_sample, col_clear = st.columns([3, 1, 1])

with col_up:
    uploaded = st.file_uploader(
        "Upload a climate report",
        type=["txt", "pdf"],
        accept_multiple_files=False,
        help="PDF or plain-text corporate climate / sustainability report.",
        on_change=_clear_result,   # clear stale results when a new file is chosen
    )

with col_sample:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📄 Sample report", use_container_width=True):
        st.session_state["report_path"] = (
            Path(__file__).parent / "data" / "sample_reports"
            / "report_01_credible_greentech_corp.txt"
        )
        st.session_state["source_label"] = "report_01_credible_greentech_corp.txt"
        _clear_result()

with col_clear:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state["report_path"]  = None
        st.session_state["source_label"] = ""
        _clear_result()
        st.rerun()

# Prefer uploaded file over sample-button choice
if uploaded is not None:
    st.session_state["report_path"]  = _save_upload(uploaded)
    st.session_state["source_label"] = uploaded.name

report_path:  Optional[Path] = st.session_state["report_path"]
source_label: str            = st.session_state["source_label"]

if report_path:
    st.success(f"📂 Ready to audit: **{source_label}**")

# Gate: need both a file and (for live providers) an API key
ready = report_path is not None and (provider == "demo" or bool(api_key))

if not ready:
    if report_path is None:
        st.markdown("⬆️ Upload a report or click **Sample report** to get started.")
    else:
        st.markdown("🔑 Enter your API key in the sidebar to run a live audit.")


# ═══════════════════════════════════════════════════════════════════════════
# Run button
# ═══════════════════════════════════════════════════════════════════════════

st.divider()
run_btn = st.button("🔍 Run Audit", type="primary", disabled=not ready)

if run_btn and ready:
    _clear_result()

    # ── Step 1: build / retrieve cached pipeline ───────────────────────────
    with st.status("⚙️ Initialising pipeline…", expanded=True) as status:
        st.write("Loading embedding model and IPCC AR6 benchmark index…")
        try:
            bm = _get_benchmarker(provider, api_key, backend)
            st.write(f"✅ Pipeline ready  ({provider} · {backend})")
        except Exception as exc:
            try:
                from tenacity import RetryError
                cause_msg = (
                    f"{type(exc.last_attempt.exception()).__name__}: "
                    f"{exc.last_attempt.exception()}"
                    if isinstance(exc, RetryError)
                    else str(exc)
                )
            except Exception:
                cause_msg = str(exc)
            status.update(label="❌ Pipeline init failed", state="error")
            st.error(cause_msg)
            st.session_state["audit_error"] = cause_msg
            st.stop()

        # ── Step 2: run the audit ──────────────────────────────────────────
        st.write(f"Auditing **{source_label}**…")
        t0 = time.time()
        try:
            doc    = bm.loader.load_file(report_path)
            result = bm.audit_document(doc)
        except Exception as exc:
            # Unwrap tenacity RetryError → show the real underlying exception
            try:
                from tenacity import RetryError
                if isinstance(exc, RetryError):
                    real_exc = exc.last_attempt.exception()
                    cause_msg = f"{type(real_exc).__name__}: {real_exc}"
                else:
                    cause_msg = str(exc)
            except Exception:
                cause_msg = str(exc)

            status.update(label="❌ Audit failed", state="error")
            st.error(cause_msg)
            st.exception(exc)
            st.session_state["audit_error"] = cause_msg
            st.stop()

        elapsed = time.time() - t0

        if result.error:
            status.update(label="❌ Audit returned an error", state="error")
            st.error(result.error)
            st.session_state["audit_error"] = result.error
            st.stop()

        # Persist result so it survives future reruns
        st.session_state["audit_result"]  = result
        st.session_state["audit_elapsed"] = elapsed
        st.session_state["audit_provider"] = provider
        status.update(label=f"✅ Audit complete in {elapsed:.1f}s", state="complete", expanded=False)


# ═══════════════════════════════════════════════════════════════════════════
# Results  — rendered from session_state, so they persist after the run
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state["audit_error"]:
    st.error(f"Last audit failed: {st.session_state['audit_error']}", icon="❌")

result = st.session_state["audit_result"]

if result is not None:
    m        = result.metrics
    elapsed  = st.session_state["audit_elapsed"]
    used_pvd = st.session_state["audit_provider"]

    GRADE_COLOUR = {"A":"#2ecc71","B":"#3498db","C":"#f1c40f","D":"#e67e22","F":"#e74c3c"}
    grade_colour = GRADE_COLOUR.get(m.grade, "#e74c3c")

    # ── Grade banner ────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:#1e2130;border-radius:12px;padding:20px 28px;
                    border:1px solid #2d3250;margin-bottom:1rem;
                    display:flex;align-items:center;gap:28px;">
            <div style="text-align:center;min-width:80px;">
                <div style="font-size:0.75rem;color:#9aa5b1;">GRADE</div>
                <div style="font-size:4rem;font-weight:900;line-height:1.1;
                             color:{grade_colour};">{m.grade}</div>
            </div>
            <div style="flex:1;">
                <div style="font-size:1.3rem;font-weight:700;">{result.company_name}</div>
                <div style="color:#9aa5b1;font-size:0.85rem;margin-top:4px;">
                    {result.num_claims} claims &nbsp;·&nbsp;
                    {result.num_chunks} chunks &nbsp;·&nbsp;
                    {elapsed:.1f}s &nbsp;·&nbsp;
                    {used_pvd}
                </div>
                <div style="margin-top:12px;">{_score_bar(m.overall_audit_score, 340)}</div>
                <div style="color:#9aa5b1;font-size:0.8rem;margin-top:4px;">
                    Overall score:
                    <b style="color:#fff;">{m.overall_audit_score:.3f}</b>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Five metric cards ───────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Factual Accuracy",   m.factual_accuracy,       False),
        (c2, "Policy Alignment",   m.policy_alignment_score, False),
        (c3, "Greenwashing Index", m.greenwashing_index,     True),   # lower = better
        (c4, "Hallucination Rate", m.hallucination_rate,     True),   # lower = better
        (c5, "Robustness Score",   m.robustness_score,       False),
    ]
    for col, label, val, lower_better in cards:
        good = val <= 0.25 if lower_better else val >= 0.75
        mid  = val <= 0.5  if lower_better else val >= 0.5
        clr  = "#2ecc71" if good else "#f1c40f" if mid else "#e74c3c"
        note = " ↓ better" if lower_better else ""
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}{note}</div>'
            f'<div class="metric-value" style="color:{clr};">{val:.2f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Claims + greenwashing side-by-side ─────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-header">📋 Claims Assessed</div>',
                    unsafe_allow_html=True)
        # Re-run consistency check to get the ClaimAssessment objects for display
        try:
            from src.ingestion.document_loader import DocumentLoader
            _doc    = DocumentLoader().load_file(report_path)
            _claims = bm.consistency_checker.check_document(
                _doc.content, doc_id=_doc.doc_id
            )
        except Exception:
            _claims = []

        if _claims:
            for i, ca in enumerate(_claims[:8], 1):
                aligned   = ca.consistent_with_ipcc
                pill_kind = "green" if aligned else "red"
                pill_text = "✔ Aligned" if aligned else "✘ Inconsistent"
                snippet   = ca.claim_text[:110] + ("…" if len(ca.claim_text) > 110 else "")
                st.markdown(
                    f"**{i}.** {snippet}<br>"
                    f"{_pill(pill_text, pill_kind)}&nbsp;"
                    f"{_pill(ca.alignment_level.replace('_',' '), 'blue')}&nbsp;"
                    f"accuracy {ca.accuracy_score:.2f}",
                    unsafe_allow_html=True,
                )
                st.markdown("---")
            if result.num_claims > 8:
                st.caption(f"… and {result.num_claims - 8} more claims assessed.")
        else:
            st.info(f"{result.num_claims} claims were assessed during the audit.")

    with right:
        st.markdown('<div class="section-header">🚨 Greenwashing Analysis</div>',
                    unsafe_allow_html=True)
        gw = m.greenwashing_index
        gw_clr  = "#2ecc71" if gw < 0.3 else "#f1c40f" if gw < 0.6 else "#e74c3c"
        gw_pill = _pill(
            "Minimal" if gw < 0.3 else "Moderate" if gw < 0.6 else "Significant",
            "green"   if gw < 0.3 else "yellow"   if gw < 0.6 else "red",
        )
        st.markdown(
            f'**Greenwashing index:** '
            f'<span style="color:{gw_clr};font-size:1.1rem;font-weight:700;">{gw:.2f}</span>'
            f' &nbsp;{gw_pill}',
            unsafe_allow_html=True,
        )
        st.markdown(_score_bar(gw, 300), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        INDICATORS = [
            ("offset_dependency",      "Offset Dependency"),
            ("scope3_omission",        "Scope 3 Omission"),
            ("vague_language",         "Vague Language"),
            ("cherry_picked_baseline", "Cherry-picked Baseline"),
            ("distant_targets",        "Distant Targets"),
            ("unproven_technology",    "Unproven Technology"),
            ("misleading_framing",     "Misleading Framing"),
            ("marketing_language",     "Marketing Language"),
        ]
        # Heuristic flags when greenwashing score is elevated
        HIGH_RISK = {"scope3_omission", "marketing_language",
                     "offset_dependency", "vague_language"}
        for key, label in INDICATORS:
            detected = gw >= 0.4 and key in HIGH_RISK
            st.markdown(f"{'🔴' if detected else '🟢'} {label}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Score breakdown chart ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Score Breakdown</div>',
                unsafe_allow_html=True)
    import pandas as pd
    chart_df = pd.DataFrame({
        "Metric": ["Factual Accuracy", "Policy Alignment",
                   "Robustness", "Overall Score"],
        "Score":  [m.factual_accuracy, m.policy_alignment_score,
                   m.robustness_score, m.overall_audit_score],
    })
    st.bar_chart(chart_df.set_index("Metric"), color="#3498db", height=220)

    # ── Recommendations ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💡 Interpretation</div>',
                unsafe_allow_html=True)
    recs = []
    if m.hallucination_rate > 0.2:
        recs.append("⚠️ **High hallucination rate** — several claims could not be verified against IPCC benchmarks. Independent data validation recommended.")
    if m.greenwashing_index >= 0.5:
        recs.append("🚨 **Significant greenwashing indicators** — review offset dependency, Scope 3 coverage, and target timelines.")
    if m.policy_alignment_score < 0.6:
        recs.append("📉 **Below Paris Agreement alignment** — targets may be insufficient for 1.5 °C. Consider SBTi certification.")
    if m.factual_accuracy < 0.7:
        recs.append("🔍 **Factual accuracy concerns** — quantitative claims deviate from IPCC AR6 reference values.")
    if m.robustness_score < 0.6:
        recs.append("🛡 **Low adversarial robustness** — internal inconsistencies or weak evidence under stress-testing.")
    if not recs:
        recs.append("✅ **Report appears credible** — claims are broadly IPCC AR6-consistent, greenwashing indicators minimal, and targets science-aligned.")
    for rec in recs:
        st.markdown(rec)

    # ── Download ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="⬇️ Download full audit (JSON)",
        data=json.dumps(result.to_dict(), indent=2),
        file_name=f"audit_{result.company_name.replace(' ','_').lower()}.json",
        mime="application/json",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════

st.divider()
with st.expander("ℹ️ How it works"):
    st.markdown("""
**8-stage pipeline:**

1. **Ingest** — report is split into overlapping chunks via LangChain `RecursiveCharacterTextSplitter`
2. **Index** — chunks embedded with HuggingFace `all-MiniLM-L6-v2` → stored in FAISS / ChromaDB
3. **Extract claims** — LLM identifies quantitative and qualitative climate commitments
4. **RAG consistency check** — each claim retrieved against 16 IPCC AR6 benchmark passages and assessed for accuracy
5. **Credibility scoring** — specificity, science alignment, completeness, verifiability and ambition rated 0–1
6. **Greenwashing detection** — 8-indicator framework (offset dependency, Scope 3 omission, vague language …)
7. **Adversarial stress-test** — contradiction probing, claim perturbation, devil's advocate, misinformation injection
8. **Evaluation metrics** — hallucination rate, factual accuracy, policy alignment → composite A–F grade

**Benchmarks:** IPCC AR6 WG III (2022) — global CO₂, energy, transport, industry, buildings, land-use.
    """)

st.caption(
    "Climate Commitment Auditing Pipeline · IPCC AR6 · "
    "Groq Llama-3.3-70B / HuggingFace Mistral-7B / Demo · Apache 2.0"
)
