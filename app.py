from dotenv import load_dotenv
import os
import streamlit as st
import joblib
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────
# LOAD KEY
# ─────────────────────────────────────────
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Industrial Mech AI Assistant",
    page_icon="⚙️",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD ML MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    return joblib.load("model.pkl")

try:
    ml_model = load_ml_model()
    ml_ready = True
except:
    ml_ready = False

# ─────────────────────────────────────────
# LOAD RAG SYSTEM
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Wikipedia knowledge base...")
def build_rag_system():
    urls = [
        "https://en.wikipedia.org/wiki/Machine_design",
        "https://en.wikipedia.org/wiki/Bearing_(mechanical)",
        "https://en.wikipedia.org/wiki/Gear",
        "https://en.wikipedia.org/wiki/Predictive_maintenance",
        "https://en.wikipedia.org/wiki/Lubrication",
        "https://en.wikipedia.org/wiki/Vibration",
        "https://en.wikipedia.org/wiki/Heat_transfer",
        "https://en.wikipedia.org/wiki/Cooling_tower",
    ]
    loader      = WebBaseLoader(urls)
    docs_raw    = loader.load()
    splitter    = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    docs        = splitter.split_documents(docs_raw)
    embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = build_rag_system()

# ─────────────────────────────────────────
# QA CHAIN
# ─────────────────────────────────────────
def get_qa_chain():
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )
    prompt_template = """
You are an Industrial Maintenance AI Assistant.
Use the context to answer the question in 2-3 sentences only.
Be direct and simple. No bullet points. No lists. No headings.

Context:
{context}

Question: {question}

Short Answer:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analyze_result" not in st.session_state:
    st.session_state.analyze_result = None

# ═══════════════════════════════════════════
# LEFT SIDE — SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:

    st.markdown("""
<div style="line-height:1.2; margin-bottom:0px;">
<span style="font-size:23px; font-weight:700;">⚙️ Industrial Mechanical</span><br>
<span style="font-size:18px; font-weight:400; color:gray;">AI Assistant</span>
</div>
""", unsafe_allow_html=True)
    st.divider()

    # ── Sensor Thresholds ──
    st.markdown("**📊 Sensor Thresholds**")
    temp_limit = st.slider("Max Temperature (°C)", 60, 120, 85)
    vib_limit  = st.slider("Max Vibration (mm/s)", 1, 10, 5)
    st.divider()

    # ── Machine Sensor Input ──
    st.markdown("**⚙️ Machine Sensor Input**")
    unit        = st.selectbox("Select Unit", ["Gear Unit 1", "Gear Unit 2", "Gear Unit 3", "Cooling Tower"])
    temperature = st.number_input("🌡️ Temperature (°C)", value=30.0, step=0.5)
    vibration   = st.number_input("📊 Vibration (mm/s)", value=1.0, step=0.1)
    load        = st.slider("⚡ Machine Load (%)", 0, 100, 40)
    oil_pressure= st.slider("🛢️ Oil Pressure (PSI)", 20, 80, 45)

    st.divider()

    if st.button("🚀 Analyze Machine Status", use_container_width=True):

        # ── Rule Based Status ──
        if temperature > temp_limit and vibration > vib_limit:
            status = "🔴 CRITICAL — Immediate Action Required"
            color  = "error"
            query  = (
                f"What causes high temperature {temperature}°C and "
                f"high vibration {vibration}mm/s in industrial gear units? "
                f"What immediate actions should be taken?"
            )
        elif temperature > temp_limit:
            status = "🟠 WARNING — High Temperature"
            color  = "warning"
            query  = (
                f"What are the causes and solutions for high temperature "
                f"{temperature}°C in industrial gear units and bearings?"
            )
        elif vibration > vib_limit:
            status = "🟠 WARNING — High Vibration"
            color  = "warning"
            query  = (
                f"What causes excessive vibration {vibration}mm/s "
                f"in industrial machinery? How to diagnose and fix it?"
            )
        elif oil_pressure < 30:
            status = "🟡 CAUTION — Low Oil Pressure"
            color  = "warning"
            query  = (
                f"What are the effects of low oil pressure {oil_pressure} PSI "
                f"in industrial machines and gear systems?"
            )
        else:
            status = "🟢 NORMAL — Machine Operating Fine"
            color  = "success"
            query  = (
                "What are best practices for maintaining normal "
                "operating conditions in industrial gear units?"
            )

        # ── ML Prediction ──
        if ml_ready:
            input_data  = np.array([[temperature, vibration, load]])
            prediction  = ml_model.predict(input_data)[0]
            probability = ml_model.predict_proba(input_data)[0][1]
            ml_result   = "🔴 FAILURE PREDICTED" if prediction == 1 else "🟢 NORMAL"
        else:
            prediction  = None
            probability = 0
            ml_result   = "ML model not loaded"

        # ── Get AI Answer ──
        qa_chain = get_qa_chain()
        result   = qa_chain.invoke({"query": query})

        st.session_state.analyze_result = {
            "unit":        unit,
            "status":      status,
            "color":       color,
            "ml_result":   ml_result,
            "probability": probability,
            "prediction":  prediction,
            "temperature": temperature,
            "vibration":   vibration,
            "load":        load,
            "oil_pressure":oil_pressure,
            "ai_answer":   result["result"]
        }
        st.rerun()

# ═══════════════════════════════════════════
# RIGHT SIDE — MAIN AREA
# ═══════════════════════════════════════════

st.markdown("""
<div style="line-height:1.3; margin-bottom:4px;">
<span style="font-size:26px; font-weight:700;">💬 Ask Industrial AI Agent</span><br>
<span style="font-size:16px; color:gray;">Ask anything about machine maintenance, gears, bearings, vibration...</span>
</div>
""", unsafe_allow_html=True)

# ── Show Analyze Result if exists ──
if st.session_state.analyze_result:
    r = st.session_state.analyze_result
    st.markdown(f"### Analysis: {r['unit']}")

    c1, c2 = st.columns(2)
    with c1:
        if r["color"] == "error":
            st.error(r["status"])
        elif r["color"] == "warning":
            st.warning(r["status"])
        else:
            st.success(r["status"])
    with c2:
        if r["prediction"] == 1:
            st.error(f"{r['ml_result']}  |  Risk: {r['probability']*100:.1f}%")
        else:
            st.success(f"{r['ml_result']}  |  Risk: {r['probability']*100:.1f}%")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Temperature",  f"{r['temperature']}°C")
    m2.metric("Vibration",    f"{r['vibration']} mm/s")
    m3.metric("Load",         f"{r['load']}%")
    m4.metric("Oil Pressure", f"{r['oil_pressure']} PSI")

    st.info(r["ai_answer"])
    st.divider()

# ── Quick Questions ──
example_questions = [
    "Why does a gear unit overheat?",
    "How to reduce vibration?",
    "What is predictive maintenance?",
    "How does bearing lubrication work?",
    "What causes cooling tower failure?",
]

st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] button {
    height: 30px !important;
    font-size: 13px !important;
    padding: 0px 2px !important;
    white-space: normal !important;
    line-height: 1 !important;
}
div[data-testid="stHorizontalBlock"] button p {
    font-size: 13px !important;
    line-height: 1.1 !important;
}
</style>
""", unsafe_allow_html=True)

st.caption("Quick Questions:")
cols = st.columns(len(example_questions))
for i, eq in enumerate(example_questions):
    with cols[i]:
        if st.button(eq, key=f"eq_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": eq})
            qa_chain = get_qa_chain()
            result   = qa_chain.invoke({"query": eq})
            st.session_state.messages.append({"role": "assistant", "content": result["result"]})
            st.rerun()
# ── Chat History ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── User Input ──
user_query = st.chat_input("Type your question here...")
if user_query:
    if groq_api_key:
        st.session_state.messages.append({"role": "user", "content": user_query})
        qa_chain = get_qa_chain()
        result   = qa_chain.invoke({"query": user_query})
        st.session_state.messages.append({"role": "assistant", "content": result["result"]})
        st.rerun()
    else:
        st.error("❌ Add Groq API Key in .env file")