# ShopEasy Support Ticket Analyzer

An AI-powered Streamlit app for analyzing customer support tickets at ShopEasy. Powered by a LangChain agent and Google Gemini, it lets you ask natural-language questions about ticket trends, resolution performance, and agent effectiveness.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your API key

Copy `.env.example` to `.env` and add your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

Get a free API key at: https://aistudio.google.com/app/apikey

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Usage

1. Enter your Google API key in the sidebar (or set it in `.env`)
2. Upload `support_tickets.csv` (included in this repo)
3. Ask questions in the chat, for example:
   - "What is the most common complaint category?"
   - "What is the average resolution time by priority?"
   - "Which agent has the best satisfaction score?"
   - "How many tickets are currently pending?"
   - "Show monthly ticket trends"

## Dataset

`support_tickets.csv` contains 30 rows of ShopEasy customer support tickets with the following columns:

| Column | Description |
|--------|-------------|
| Ticket_ID | Unique ticket identifier |
| Date | Date ticket was created |
| Customer_Name | Name of the customer |
| Category | Complaint type: Delivery Delay, Product Defect, Refund Request, Account Issue, Payment Failed |
| Priority | Low, Medium, High, or Critical |
| Resolution_Time_Hours | Hours taken to resolve the ticket |
| Satisfaction_Score | Customer rating from 1 (worst) to 5 (best) |
| Status | Resolved, Pending, or Escalated |
| Agent_Name | Support agent who handled the ticket |

## Project Structure

```
ai-agent-exam/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── support_tickets.csv   # ShopEasy customer support dataset
└── README.md             # This file
```

## Tech Stack

- **Streamlit** — UI framework
- **LangChain** — Agent orchestration
- **Google Gemini 2.0 Flash** — LLM powering the agent
- **Pandas** — Data manipulation
- **langchain-experimental** — Pandas DataFrame agent
