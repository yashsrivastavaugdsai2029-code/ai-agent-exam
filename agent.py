"""
agent.py — RetailMind Analytics: LLM-based router + agent orchestration.

Architecture:
  1. classify_intent()   — LLM classifies user query into one of 5 intents
  2. run_agent()         — dispatches to the right tool(s) and synthesises a reply
  3. generate_daily_briefing() — auto-generated startup summary
"""

import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage

from tools import (
    generate_restock_alert,
    get_category_performance,
    get_inventory_health,
    get_pricing_analysis,
    get_review_insights,
    load_products,
    search_products,
)

# ── Constants ────────────────────────────────────────────────────────────────
VALID_INTENTS   = {"INVENTORY", "PRICING", "REVIEWS", "CATALOG", "GENERAL"}
VALID_CATEGORIES = {"TOPS", "DRESSES", "BOTTOMS", "OUTERWEAR", "ACCESSORIES", "ALL"}

SYSTEM_PERSONA = (
    "You are RetailMind, an AI product-intelligence assistant for StyleCraft — "
    "a D2C fashion brand. You help the team analyse inventory, pricing, customer "
    "reviews, and catalog performance. Be concise, professional, and actionable. "
    "Use bullet points and clear markdown formatting where helpful. "
    "Always reference specific product IDs and names when relevant."
)


# ── Step 1: LLM-based intent classification ──────────────────────────────────
def classify_intent(query: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Uses the LLM to classify the user query into one of 5 intent buckets.
    Returns a string from VALID_INTENTS; falls back to 'GENERAL' on failure.
    """
    prompt = (
        "Classify the following user query into EXACTLY one of these categories:\n"
        "- INVENTORY  : stock levels, days-to-stockout, restock, replenishment\n"
        "- PRICING    : gross margins, profitability, cost analysis, price strategy\n"
        "- REVIEWS    : customer feedback, ratings, complaints, sentiment, quality issues\n"
        "- CATALOG    : product search, category overview, top performers, SKU listings\n"
        "- GENERAL    : greetings, meta questions, general brand/assistant questions\n\n"
        f'Query: "{query}"\n\n'
        "Respond with ONLY the single category name in uppercase. No explanation."
    )
    try:
        intent = llm.invoke(prompt).content.strip().upper()
        return intent if intent in VALID_INTENTS else "GENERAL"
    except Exception:
        return "GENERAL"


# ── Step 2: Helper — extract product ID from free text ───────────────────────
def _extract_product_id(query: str, llm: ChatGoogleGenerativeAI) -> str | None:
    """
    Asks the LLM to extract a product ID (e.g. SC001) from the query.
    Returns the ID string or None.
    """
    prompt = (
        "Does the following query reference a specific product ID "
        "(format: SC001 … SC030)?\n"
        f'Query: "{query}"\n'
        "If yes, respond with just the product ID (e.g. SC007). "
        "If no specific product ID is mentioned, respond with NONE."
    )
    try:
        result = llm.invoke(prompt).content.strip().upper()
        if result != "NONE" and result.startswith("SC"):
            return result
    except Exception:
        pass
    return None


# ── Step 3: Helper — detect category from query ──────────────────────────────
def _extract_category(query: str, llm: ChatGoogleGenerativeAI) -> str | None:
    """
    Asks the LLM whether the query targets a specific category.
    Returns the category name or None.
    """
    prompt = (
        "Does the following query ask about a specific product category?\n"
        "Valid categories: Tops, Dresses, Bottoms, Outerwear, Accessories\n"
        f'Query: "{query}"\n'
        "If yes, reply with ONLY the category name (e.g. Tops). "
        "If no specific category, reply with NONE."
    )
    try:
        result = llm.invoke(prompt).content.strip()
        if result.upper() in VALID_CATEGORIES and result.upper() != "NONE":
            return result.capitalize()
    except Exception:
        pass
    return None


# ── Step 4: Router dispatch ──────────────────────────────────────────────────
def _handle_inventory(query: str, llm: ChatGoogleGenerativeAI) -> str:
    product_id = _extract_product_id(query, llm)
    if product_id:
        data = get_inventory_health(product_id)
    else:
        data = generate_restock_alert(threshold_days=14)
    return f"**Inventory Data:**\n```json\n{json.dumps(data, indent=2)}\n```"


def _handle_pricing(query: str, llm: ChatGoogleGenerativeAI) -> str:
    product_id = _extract_product_id(query, llm)
    if product_id:
        data = get_pricing_analysis(product_id)
    else:
        # Return a pricing overview — lowest-margin products
        df = load_products()
        df = df.copy()
        df["margin_pct"] = (df["price"] - df["cost"]) / df["price"] * 100
        low_margin = (
            df[df["margin_pct"] < 30]
            .sort_values("margin_pct")
            .head(6)[["product_id", "product_name", "price", "cost"]]
        )
        low_margin["margin_pct"] = low_margin.apply(
            lambda r: round((r["price"] - r["cost"]) / r["price"] * 100, 1), axis=1
        )
        data = {
            "pricing_overview": "Products with margin below 30%",
            "products": low_margin.to_dict("records"),
        }
    return f"**Pricing Data:**\n```json\n{json.dumps(data, indent=2)}\n```"


def _handle_reviews(query: str, llm: ChatGoogleGenerativeAI) -> str:
    product_id = _extract_product_id(query, llm)
    if product_id:
        data = get_review_insights(product_id, llm)
    else:
        df = load_products()
        low_rated = df.nsmallest(5, "avg_rating")[
            ["product_id", "product_name", "avg_rating", "review_count"]
        ].to_dict("records")
        data = {
            "review_overview": "Lowest-rated products across the catalog",
            "products": low_rated,
        }
    return f"**Review Data:**\n```json\n{json.dumps(data, indent=2)}\n```"


def _handle_catalog(query: str, llm: ChatGoogleGenerativeAI, sidebar_category: str = None) -> str:
    category = _extract_category(query, llm) or sidebar_category
    if category and category.lower() != "all":
        data = get_category_performance(category)
        return f"**Category Performance:**\n```json\n{json.dumps(data, indent=2)}\n```"
    else:
        results = search_products(query, category=sidebar_category)
        return f"**Product Search Results:**\n```json\n{json.dumps(results, indent=2)}\n```"


# ── Main agent entry point ────────────────────────────────────────────────────
def run_agent(
    query: str,
    chat_history: list[dict],
    llm: ChatGoogleGenerativeAI,
    sidebar_category: str = "All",
) -> str:
    """
    LLM-routed agent:
      1. Classify intent with LLM
      2. Call appropriate tool(s) based on intent
      3. Synthesise a natural-language response with the LLM
    """

    # ── 1. Classify intent (LLM-based, NOT keyword matching) ──────────────────
    intent = classify_intent(query, llm)

    # ── 2. Gather tool data ───────────────────────────────────────────────────
    tool_context = ""

    if intent == "INVENTORY":
        tool_context = _handle_inventory(query, llm)

    elif intent == "PRICING":
        tool_context = _handle_pricing(query, llm)

    elif intent == "REVIEWS":
        tool_context = _handle_reviews(query, llm)

    elif intent == "CATALOG":
        tool_context = _handle_catalog(query, llm, sidebar_category=sidebar_category)

    # GENERAL — no tool data needed
    # ── 3. Build conversation history for multi-turn context ──────────────────
    lc_history: list[HumanMessage | AIMessage] = []
    for msg in chat_history[-8:]:   # last 4 turns
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        else:
            lc_history.append(AIMessage(content=msg["content"]))

    # ── 4. Synthesise final response ──────────────────────────────────────────
    if intent == "GENERAL":
        lc_history.append(
            HumanMessage(content=f"{SYSTEM_PERSONA}\n\nUser: {query}")
        )
        return llm.invoke(lc_history).content

    synthesis_prompt = (
        f"{SYSTEM_PERSONA}\n\n"
        f"User query: {query}\n"
        f"Detected intent: {intent}\n\n"
        f"Data retrieved from tools:\n{tool_context}\n\n"
        "Using the data above, provide a helpful, concise, and actionable response. "
        "Use markdown formatting, bullet points, and ₹ for currency where appropriate. "
        "If data shows a problem (critical stock, low margin, poor reviews), "
        "explicitly call it out and suggest a next step."
    )

    lc_history.append(HumanMessage(content=synthesis_prompt))
    return llm.invoke(lc_history).content


# ── Daily Briefing generator ─────────────────────────────────────────────────
def generate_daily_briefing(llm: ChatGoogleGenerativeAI) -> str:
    """
    Auto-generates a daily intelligence briefing covering:
      - Top 3 critically low-stock products
      - Worst-rated product with a one-line unhappiness reason
      - Lowest-margin pricing flag (if margin < 25%)
    """
    # 1. Critical stock data
    restock = generate_restock_alert(threshold_days=7)
    critical_items = restock["alerts"][:3]

    # 2. Worst-rated product + quick review insight (LLM summarises)
    df = load_products()
    worst = df.nsmallest(1, "avg_rating").iloc[0]
    worst_reviews = get_review_insights(str(worst["product_id"]), llm)

    # 3. Lowest gross-margin product below 25%
    df2 = df.copy()
    df2["margin_pct"] = (df2["price"] - df2["cost"]) / df2["price"] * 100
    low_margin_df = df2[df2["margin_pct"] < 25].sort_values("margin_pct")
    pricing_flag = None
    if not low_margin_df.empty:
        row = low_margin_df.iloc[0]
        pricing_flag = {
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "price": float(row["price"]),
            "cost": float(row["cost"]),
            "gross_margin_pct": round(float(row["margin_pct"]), 1),
        }

    briefing_data = {
        "critical_stock_alerts": critical_items,
        "worst_rated_product": {
            "product_id": str(worst["product_id"]),
            "product_name": str(worst["product_name"]),
            "avg_rating": float(worst["avg_rating"]),
            "review_summary": worst_reviews.get("sentiment_summary", "N/A"),
        },
        "pricing_flag": pricing_flag,
    }

    prompt = (
        "Generate a professional daily intelligence briefing for the RetailMind "
        "Analytics dashboard for StyleCraft. Use the data below:\n\n"
        f"{json.dumps(briefing_data, indent=2)}\n\n"
        "Structure the briefing EXACTLY as follows:\n\n"
        "## 📋 Daily Intelligence Briefing\n\n"
        "### 📦 Stock Alerts\n"
        "List the top 3 critical items. For each: product name, current stock, "
        "days-to-stockout, revenue at risk in ₹. Use a bullet per item.\n\n"
        "### ⭐ Quality Watch\n"
        "Name the worst-rated product (with rating), and give ONE sentence "
        "explaining why customers are unhappy (based on review_summary).\n\n"
        "### 💰 Pricing Alert\n"
        "If a pricing flag exists, name the product and its gross margin %, "
        "then suggest ONE specific action (e.g. increase price by X% or reduce cost). "
        "If no flag exists, state all margins are healthy.\n\n"
        "Keep each section concise. Use ₹ for Indian Rupee amounts."
    )

    try:
        return llm.invoke(prompt).content
    except Exception as exc:
        return f"⚠️ Could not generate briefing: {exc}"
