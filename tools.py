"""
tools.py — RetailMind Analytics: The 6 core tool functions called by the LLM agent.

Each function reads from the CSV datasets and returns structured data dicts.
get_review_insights() accepts an optional llm argument for LLM-powered summarisation.
"""

import re
import json
from pathlib import Path

import pandas as pd

# ── File paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PRODUCTS_CSV = BASE_DIR / "Set-B retailmind_products.csv"
REVIEWS_CSV  = BASE_DIR / "Set-B retailmind_reviews.csv"


# ── Loaders ─────────────────────────────────────────────────────────────────
def load_products() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_CSV)


def load_reviews() -> pd.DataFrame:
    return pd.read_csv(REVIEWS_CSV)


# ── Tool 1: search_products ──────────────────────────────────────────────────
def search_products(query: str, category: str = None) -> dict:
    """
    Search products by free-text query and optional category filter.
    Returns top-5 matches ranked by word overlap.
    """
    df = load_products()

    # Optional category filter
    if category and category.lower() not in ("all", "none", ""):
        df = df[df["category"].str.lower() == category.lower()]

    if df.empty:
        return {"results": [], "total": 0, "message": f"No products found in category '{category}'."}

    # Score rows by query-word overlap with product name + category
    query_words = set(query.lower().split())

    def score_row(row):
        text = f"{row['product_name']} {row['category']}".lower()
        return sum(1 for w in query_words if w in text)

    df = df.copy()
    df["_score"] = df.apply(score_row, axis=1)
    df = df.sort_values("_score", ascending=False).head(5)

    results = [
        {
            "product_id": row["product_id"],
            "name": row["product_name"],
            "category": row["category"],
            "price": float(row["price"]),
            "stock": int(row["stock_quantity"]),
            "rating": float(row["avg_rating"]),
        }
        for _, row in df.iterrows()
    ]
    return {"results": results, "total": len(results)}


# ── Tool 2: get_inventory_health ─────────────────────────────────────────────
def get_inventory_health(product_id: str) -> dict:
    """
    Returns stock, avg daily sales, days-to-stockout, and a status flag
    (Critical < 7 days | Low 7–14 days | Healthy > 14 days).
    """
    df = load_products()
    row_df = df[df["product_id"].str.upper() == product_id.upper()]

    if row_df.empty:
        return {"error": f"Product '{product_id}' not found."}

    row = row_df.iloc[0]
    stock = float(row["stock_quantity"])
    avg_sales = float(row["avg_daily_sales"])

    days_to_stockout: float | str
    if avg_sales > 0:
        days_to_stockout = round(stock / avg_sales, 1)
        if days_to_stockout < 7:
            status = "Critical"
        elif days_to_stockout < 14:
            status = "Low"
        else:
            status = "Healthy"
    else:
        days_to_stockout = "N/A (no sales)"
        status = "Healthy"

    return {
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "current_stock": int(stock),
        "avg_daily_sales": avg_sales,
        "days_to_stockout": days_to_stockout,
        "status": status,
        "reorder_level": int(row["reorder_level"]),
    }


# ── Tool 3: get_pricing_analysis ─────────────────────────────────────────────
def get_pricing_analysis(product_id: str) -> dict:
    """
    Returns gross margin %, price positioning relative to category average,
    and a flag if margin is below 20 %.
    """
    df = load_products()
    row_df = df[df["product_id"].str.upper() == product_id.upper()]

    if row_df.empty:
        return {"error": f"Product '{product_id}' not found."}

    row = row_df.iloc[0]
    price = float(row["price"])
    cost  = float(row["cost"])

    gross_margin = (price - cost) / price * 100

    # Category-based price positioning
    cat_avg_price = float(
        df[df["category"] == row["category"]]["price"].mean()
    )
    if price > cat_avg_price * 1.2:
        positioning = "Premium"
    elif price < cat_avg_price * 0.8:
        positioning = "Budget"
    else:
        positioning = "Mid-Range"

    margin_below_20 = gross_margin < 20

    return {
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "price": price,
        "cost": cost,
        "gross_margin_pct": round(gross_margin, 1),
        "price_positioning": positioning,
        "category_avg_price": round(cat_avg_price, 1),
        "margin_below_20_flag": margin_below_20,
        "margin_warning": (
            "⚠️ Margin is below 20% — consider reviewing cost or price."
            if margin_below_20
            else "✅ Margin is healthy."
        ),
    }


# ── Tool 4: get_review_insights ──────────────────────────────────────────────
def get_review_insights(product_id: str, llm=None) -> dict:
    """
    Filters reviews CSV for the given product, then uses the LLM to produce a
    2-sentence sentiment summary plus top-2 positive and negative themes.
    Falls back to raw stats if no LLM is supplied.
    """
    products_df = load_products()
    reviews_df  = load_reviews()

    prod_df = products_df[products_df["product_id"].str.upper() == product_id.upper()]
    if prod_df.empty:
        return {"error": f"Product '{product_id}' not found."}

    prod_name = prod_df.iloc[0]["product_name"]
    rev_df    = reviews_df[reviews_df["product_id"].str.upper() == product_id.upper()]

    if rev_df.empty:
        return {
            "product_id": product_id,
            "product_name": prod_name,
            "avg_rating": float(prod_df.iloc[0]["avg_rating"]),
            "total_reviews": 0,
            "sentiment_summary": "No reviews available for this product.",
            "positive_themes": [],
            "negative_themes": [],
        }

    avg_rating = round(float(rev_df["rating"].mean()), 1)

    # Build a concise review blob for the LLM
    reviews_text = "\n".join(
        f"[{r['rating']}/5] {r['review_title']}: {r['review_text']}"
        for _, r in rev_df.iterrows()
    )

    if llm:
        prompt = (
            f"Analyse these customer reviews for the product \"{prod_name}\":\n\n"
            f"{reviews_text}\n\n"
            "Respond in EXACTLY this format (no extra text):\n"
            "SUMMARY: <2 sentences capturing overall sentiment>\n"
            "POSITIVE: <theme1> | <theme2>\n"
            "NEGATIVE: <theme1> | <theme2>"
        )
        try:
            content = llm.invoke(prompt).content
            s_match = re.search(r"SUMMARY:\s*(.+?)(?=POSITIVE:|$)", content, re.DOTALL)
            p_match = re.search(r"POSITIVE:\s*(.+?)(?=NEGATIVE:|$)", content, re.DOTALL)
            n_match = re.search(r"NEGATIVE:\s*(.+)",                  content, re.DOTALL)

            sentiment_summary = s_match.group(1).strip() if s_match else "Summary unavailable."
            pos_raw = p_match.group(1).strip() if p_match else ""
            neg_raw = n_match.group(1).strip() if n_match else ""

            positive_themes = [t.strip() for t in pos_raw.split("|")][:2] if pos_raw else []
            negative_themes = [t.strip() for t in neg_raw.split("|")][:2] if neg_raw else []
        except Exception as exc:
            sentiment_summary = f"LLM summarisation failed: {exc}"
            positive_themes, negative_themes = [], []
    else:
        sentiment_summary = "LLM not available — showing raw stats only."
        positive_themes, negative_themes = [], []

    return {
        "product_id": product_id,
        "product_name": prod_name,
        "avg_rating": avg_rating,
        "total_reviews": len(rev_df),
        "sentiment_summary": sentiment_summary,
        "positive_themes": positive_themes,
        "negative_themes": negative_themes,
    }


# ── Tool 5: get_category_performance ────────────────────────────────────────
def get_category_performance(category: str) -> dict:
    """
    Aggregates SKU count, avg rating, avg margin %, total stock, low/critical
    stock count, and top-3 revenue products for a given category (or 'All').
    """
    df = load_products()

    if category.lower() not in ("all", ""):
        cat_df = df[df["category"].str.lower() == category.lower()].copy()
    else:
        cat_df = df.copy()

    if cat_df.empty:
        return {"error": f"Category '{category}' not found."}

    cat_df["margin_pct"]    = (cat_df["price"] - cat_df["cost"]) / cat_df["price"] * 100
    cat_df["daily_revenue"] = cat_df["price"] * cat_df["avg_daily_sales"]
    cat_df["days_to_stockout"] = cat_df.apply(
        lambda r: r["stock_quantity"] / r["avg_daily_sales"]
        if r["avg_daily_sales"] > 0 else float("inf"),
        axis=1,
    )

    low_stock_items      = int((cat_df["days_to_stockout"] < 14).sum())
    critical_stock_items = int((cat_df["days_to_stockout"] <  7).sum())

    top_revenue = (
        cat_df.nlargest(3, "daily_revenue")[
            ["product_id", "product_name", "daily_revenue"]
        ]
        .to_dict("records")
    )

    return {
        "category": category,
        "total_skus": len(cat_df),
        "avg_rating": round(float(cat_df["avg_rating"].mean()), 2),
        "avg_margin_pct": round(float(cat_df["margin_pct"].mean()), 1),
        "total_stock": int(cat_df["stock_quantity"].sum()),
        "low_stock_items": low_stock_items,
        "critical_stock_items": critical_stock_items,
        "top_revenue_products": [
            {
                "product_id": r["product_id"],
                "name": r["product_name"],
                "daily_revenue": round(r["daily_revenue"], 1),
            }
            for r in top_revenue
        ],
    }


# ── Tool 6: generate_restock_alert ──────────────────────────────────────────
def generate_restock_alert(threshold_days: int = 7) -> dict:
    """
    Scans all products for stockout risk within `threshold_days`.
    Returns sorted alerts with urgency level and estimated revenue at risk.
    """
    df = load_products().copy()

    df["days_to_stockout"] = df.apply(
        lambda r: r["stock_quantity"] / r["avg_daily_sales"]
        if r["avg_daily_sales"] > 0 else float("inf"),
        axis=1,
    )
    df["revenue_at_risk"] = df["price"] * df["avg_daily_sales"] * threshold_days

    at_risk = df[df["days_to_stockout"] <= threshold_days].sort_values("days_to_stockout")

    alerts = []
    for _, row in at_risk.iterrows():
        days = float(row["days_to_stockout"])
        if days < 3:
            urgency = "CRITICAL"
        elif days < 5:
            urgency = "HIGH"
        else:
            urgency = "MEDIUM"

        alerts.append(
            {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "category": row["category"],
                "current_stock": int(row["stock_quantity"]),
                "days_to_stockout": round(days, 1),
                "revenue_at_risk": round(float(row["revenue_at_risk"]), 2),
                "urgency": urgency,
            }
        )

    return {
        "threshold_days": threshold_days,
        "total_at_risk": len(alerts),
        "total_revenue_at_risk": round(sum(a["revenue_at_risk"] for a in alerts), 2),
        "alerts": alerts,
    }
