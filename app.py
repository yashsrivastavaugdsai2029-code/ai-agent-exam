"""
app.py — RetailMind Analytics: Streamlit UI for StyleCraft Product Intelligence.

Layout:
  • Sidebar  : Google API key input, category filter, catalog summary panel,
               clear-chat button.
  • Main area: Auto-generated daily briefing on load, then chat interface
               with full conversation history and multi-turn memory.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from agent import generate_daily_briefing, run_agent
from tools import generate_restock_alert, load_products

# ── Bootstrap ────────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="RetailMind Analytics — StyleCraft",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """Instantiate and cache the Gemini LLM for a given API key."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.3,
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_catalog_summary() -> dict:
    """Compute sidebar summary statistics (cached for 5 min)."""
    df = load_products()
    df = df.copy()
    df["margin_pct"] = (df["price"] - df["cost"]) / df["price"] * 100
    restock = generate_restock_alert(threshold_days=7)
    return {
        "total_skus": len(df),
        "critical_stock": restock["total_at_risk"],
        "avg_margin": round(float(df["margin_pct"].mean()), 1),
        "avg_rating": round(float(df["avg_rating"].mean()), 2),
    }


# ── Session-state initialisation ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "briefing_generated" not in st.session_state:
    st.session_state.briefing_generated = False
if "briefing_content" not in st.session_state:
    st.session_state.briefing_content = None


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🛍️ RetailMind")
    st.caption("AI Product Intelligence · StyleCraft")
    st.divider()

    # ── API key input ─────────────────────────────────────────────────────────
    st.subheader("🔑 Authentication")
    api_key = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        placeholder="AIza…",
        help="Get your key at https://aistudio.google.com/app/apikey",
    )

    st.divider()

    # ── Category filter ───────────────────────────────────────────────────────
    st.subheader("📂 Category Filter")
    selected_category = st.selectbox(
        "Filter by Category",
        options=["All", "Tops", "Dresses", "Bottoms", "Outerwear", "Accessories"],
        index=0,
    )

    st.divider()

    # ── Catalog Summary panel ─────────────────────────────────────────────────
    st.subheader("📊 Catalog Summary")
    try:
        summary = get_catalog_summary()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total SKUs",   summary["total_skus"])
            st.metric("Avg Rating",   f"⭐ {summary['avg_rating']}")
        with c2:
            st.metric("Critical Stock", f"🔴 {summary['critical_stock']}")
            st.metric("Avg Margin",     f"{summary['avg_margin']}%")
    except Exception as e:
        st.error(f"Catalog summary unavailable: {e}")

    st.divider()

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat & Re-brief", use_container_width=True, type="secondary"):
        st.session_state.chat_history      = []
        st.session_state.briefing_generated = False
        st.session_state.briefing_content   = None
        st.cache_resource.clear()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🛍️ RetailMind Analytics")
st.caption(
    "AI-powered product intelligence for **StyleCraft** · "
    "80+ SKUs across Tops, Dresses, Bottoms, Outerwear & Accessories"
)

# ── Gate on API key ───────────────────────────────────────────────────────────
if not api_key:
    st.warning(
        "⚠️ Please enter your **Google API Key** in the sidebar to get started.",
        icon="🔑",
    )
    st.stop()

try:
    llm = get_llm(api_key)
except Exception as err:
    st.error(f"Failed to initialise Gemini LLM: {err}")
    st.stop()


# ── Daily Briefing (auto-generated once per session / after clear) ─────────────
if not st.session_state.briefing_generated:
    with st.spinner("🔄 Generating your daily intelligence briefing…"):
        try:
            briefing = generate_daily_briefing(llm)
            st.session_state.briefing_content   = briefing
            st.session_state.briefing_generated = True
        except Exception as err:
            st.session_state.briefing_content   = f"⚠️ Could not generate briefing: {err}"
            st.session_state.briefing_generated = True

if st.session_state.briefing_content:
    with st.expander("📋 Daily Intelligence Briefing", expanded=True):
        st.markdown(st.session_state.briefing_content)

st.divider()

# ── Conversation history ──────────────────────────────────────────────────────
st.subheader("💬 Ask RetailMind")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
placeholder = (
    "Ask about inventory levels, pricing margins, customer reviews, or product search…"
)
if user_input := st.chat_input(placeholder):
    # Persist and display the user's message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display the assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = run_agent(
                    query=user_input,
                    chat_history=st.session_state.chat_history[:-1],  # exclude current turn
                    llm=llm,
                    sidebar_category=selected_category,
                )
                st.markdown(response)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as err:
                error_msg = f"❌ Error: {err}"
                st.error(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )
