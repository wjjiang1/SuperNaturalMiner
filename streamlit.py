import streamlit as st
from langchain_openai import OpenAI
from os import environ
from dotenv import load_dotenv

st.title("🦜🔗 Quickstart App")

load_dotenv()

openai_api_key = environ.get("OPENAI_API_KEY")


def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm.invoke(input_text))


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)
