# RetailMind Analytics — StyleCraft Product Intelligence Agent

An AI-powered Streamlit application built for **RetailMind Analytics** to serve **StyleCraft**, a D2C fashion brand with 80+ SKUs across 5 categories. The agent analyses inventory health, pricing margins, customer reviews, and catalog performance using Google Gemini 2.0 Flash via LangChain.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      app.py  (Streamlit UI)             │
│  Sidebar: API key · Category filter · Catalog summary   │
│  Main: Daily briefing · Chat interface · History        │
└──────────────────────┬──────────────────────────────────┘
                       │ user query
                       ▼
┌─────────────────────────────────────────────────────────┐
│                     agent.py  (Router)                  │
│                                                         │
│  1. classify_intent(query, llm)                         │
│     LLM classifies into: INVENTORY · PRICING ·          │
│     REVIEWS · CATALOG · GENERAL                         │
│                                                         │
│  2. Dispatch to tool(s) based on intent                 │
│     (LLM also extracts product_id / category)           │
│                                                         │
│  3. LLM synthesises final response with tool context    │
└──────────────────────┬──────────────────────────────────┘
                       │ tool calls
                       ▼
┌─────────────────────────────────────────────────────────┐
│                     tools.py  (6 Tool Functions)        │
│                                                         │
│  search_products()          get_inventory_health()      │
│  get_pricing_analysis()     get_review_insights()       │
│  get_category_performance() generate_restock_alert()    │
└─────────────────────────────────────────────────────────┘
                       │ reads
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Set-B retailmind_products.csv  (30 SKUs)               │
│  Set-B retailmind_reviews.csv   (40 reviews)            │
└─────────────────────────────────────────────────────────┘
```

### Router Pattern

Intent classification uses the LLM — **not** keyword matching. The agent sends the user's raw query to Gemini and asks it to classify into one of five buckets:

| Intent | Triggers | Tools called |
|--------|----------|--------------|
| INVENTORY | stock levels, stockout, reorder | `get_inventory_health`, `generate_restock_alert` |
| PRICING | margins, profitability, cost | `get_pricing_analysis` |
| REVIEWS | feedback, ratings, sentiment | `get_review_insights` |
| CATALOG | product search, category overview | `search_products`, `get_category_performance` |
| GENERAL | greetings, meta questions | LLM knowledge only |

---

## Tool Functions

| # | Function | Description |
|---|----------|-------------|
| 1 | `search_products(query, category)` | Text search across product catalog; returns top 5 matches |
| 2 | `get_inventory_health(product_id)` | Days-to-stockout, Critical/Low/Healthy flag |
| 3 | `get_pricing_analysis(product_id)` | Gross margin %, price positioning, margin warning |
| 4 | `get_review_insights(product_id)` | LLM-summarised sentiment, top themes |
| 5 | `get_category_performance(category)` | Aggregated SKU/rating/margin/stock stats |
| 6 | `generate_restock_alert(threshold_days)` | All at-risk products sorted by urgency + revenue at risk |

---

## Daily Briefing

Generated automatically when the app loads (or after "Clear Chat"):

- **📦 Stock Alerts** — Top 3 critically low-stock products with days-to-stockout and ₹ revenue at risk
- **⭐ Quality Watch** — Worst-rated product with a one-line unhappiness reason from reviews
- **💰 Pricing Alert** — Product with the lowest gross margin (if below 25%) with a suggested action

---

## Setup

### 1. Clone / enter the project directory

```bash
cd ai-agent-exam
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
```

Edit `.env`:

```
GOOGLE_API_KEY=AIza...your_key_here
```

Get a free key at: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 4. Run the app

```bash
python run.py
# or
python start.py
# or directly
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Datasets

### `Set-B retailmind_products.csv` — 30 rows

| Column | Description |
|--------|-------------|
| product_id | Unique SKU (SC001–SC030) |
| product_name | Product display name |
| category | Tops / Dresses / Bottoms / Outerwear / Accessories |
| price | Selling price (₹) |
| cost | Unit cost (₹) |
| stock_quantity | Current units in warehouse |
| avg_daily_sales | Average units sold per day |
| return_rate | Fraction returned (0–1) |
| avg_rating | Average customer rating (1–5) |
| review_count | Total number of reviews |
| launch_date | Product launch date |
| reorder_level | Minimum stock threshold |

### `Set-B retailmind_reviews.csv` — 40 rows

| Column | Description |
|--------|-------------|
| review_id | Unique review ID |
| product_id | Links to products CSV |
| reviewer_name | Customer name |
| rating | 1–5 star rating |
| review_title | Short title |
| review_text | Full review body |
| verified_purchase | TRUE / FALSE |
| helpful_votes | Community upvotes |
| review_date | Date of review |

---

## Project Structure

```
ai-agent-exam/
├── app.py                            # Streamlit UI
├── agent.py                          # LLM router + response synthesis
├── tools.py                          # 6 tool functions
├── run.py                            # Entry point (python run.py)
├── start.py                          # Alternate entry point
├── Set-B retailmind_products.csv     # Products dataset (30 SKUs)
├── Set-B retailmind_reviews.csv      # Reviews dataset (40 reviews)
├── requirements.txt                  # Python dependencies
├── .env                              # Your secrets (git-ignored)
├── .env.example                      # Template for .env
├── .gitignore                        # Excludes .env, __pycache__, etc.
└── README.md                         # This file
```

---

## Tech Stack

| Library | Role |
|---------|------|
| **Streamlit** | Web UI framework |
| **LangChain** | Agent orchestration, message history |
| **langchain-google-genai** | Gemini 2.0 Flash integration |
| **google-generativeai** | Google AI SDK |
| **Pandas** | CSV loading and data analysis |
| **python-dotenv** | `.env` file management |

---

## Example Queries

- `"Which products are about to run out of stock?"`
- `"What is the gross margin for SC018?"`
- `"Why are customers unhappy with SC010?"`
- `"Show me all Dresses and their performance"`
- `"Which products in Tops have low inventory?"`
- `"Give me a pricing overview — which products have poor margins?"`
