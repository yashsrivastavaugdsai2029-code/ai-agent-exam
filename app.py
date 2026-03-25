import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

load_dotenv()

st.set_page_config(page_title="AI Dataset Analyst", page_icon="🤖", layout="wide")
st.title("🤖 AI Dataset Analyst")
st.caption("Upload a CSV and ask questions about your data using Google Gemini")

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Enter your Google Gemini API key"
    )
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Enter your API key")
    st.markdown("2. Upload a CSV file")
    st.markdown("3. Ask questions about your data")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

# File upload
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df

    st.subheader("Dataset Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df, use_container_width=True)

# Chat interface
st.subheader("Chat with your Data")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your dataset..."):
    if not api_key:
        st.error("Please enter your Google API Key in the sidebar.")
    elif st.session_state.df is None:
        st.error("Please upload a CSV file first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash",
                        google_api_key=api_key,
                        temperature=0
                    )

                    agent = create_pandas_dataframe_agent(
                        llm,
                        st.session_state.df,
                        verbose=False,
                        agent_type=AgentType.OPENAI_FUNCTIONS,
                        allow_dangerous_code=True
                    )

                    response = agent.invoke({"input": prompt})
                    answer = response.get("output", str(response))
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
