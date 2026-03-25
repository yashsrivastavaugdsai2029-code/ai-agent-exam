# AI Dataset Analyst

A Streamlit app that uses a LangChain AI agent powered by Google Gemini to answer questions about your CSV data.

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
2. Upload a CSV file — use `sample_data.csv` to try it out
3. Ask questions in the chat, for example:
   - "What is the total revenue?"
   - "Which product sold the most units?"
   - "Show me revenue by region"
   - "What is the average revenue per transaction?"

## Project Structure

```
ai-agent-exam/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
├── sample_data.csv   # Sample sales dataset
└── README.md         # This file
```

## Tech Stack

- **Streamlit** — UI framework
- **LangChain** — Agent orchestration
- **Google Gemini 2.0 Flash** — LLM powering the agent
- **Pandas** — Data manipulation
- **langchain-experimental** — Pandas DataFrame agent
