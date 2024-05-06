import streamlit as st
from langchain_openai import OpenAI
from langchain import OpenAI, SQLDatabase
from os import environ
from langchain_experimental.sql import SQLDatabaseChain
from dotenv import load_dotenv

st.title("🦜🔗 Quickstart App")

load_dotenv()

OPENAI_API_KEY = environ.get("OPENAI_API_KEY")
username = environ.get("username")
password = environ.get("password")
host = environ.get("host")
port = environ.get("port")
dbname = environ.get("dbname")
pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}"
db = SQLDatabase.from_uri(pg_uri)


def generate_data_response(db_uri, input_text):
    PROMPT = """ 
    Given an input text, first create a syntactically correct postgresql query to run,  
    then look at the results of the query and return the answer. Remove the ``` in the response.  
    The input text: {input_text}
    """
    llm = OpenAI(temperature=0, api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    db_chain = SQLDatabaseChain(llm=llm, database=db_uri, verbose=True, top_k=3)
    data_text = db_chain.run(PROMPT.format(input_text=input_text))
    return data_text


def generate_semantic_response(input_text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal:test-1:9LcLIXjl",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    )


with st.form("my_form"):
    text = st.text_area("Enter text:")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)
