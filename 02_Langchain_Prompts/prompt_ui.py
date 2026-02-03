from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=api_key, temperature=0.5)

st.set_page_config(page_title="LangChain Frontend", page_icon="🤖", layout="centered")

st.title("💬 LangChain Static Prompt Interface")

prompt = st.text_area("Enter your prompt:", placeholder="Ask me anything...")

if st.button("Send"):
    if prompt.strip() == "":
        st.warning("Please enter a valid prompt.")
    else:
        with st.spinner("Thinking..."):
            response = llm.invoke(prompt).content
        st.success("Response:")
        st.write(response)


