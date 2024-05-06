from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from os import environ

# Load Environemnt Variables
username = environ.get("username")
password = environ.get("password")
host = environ.get("host")
port = environ.get("port")
dbname = environ.get("dbname")
pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}"
db = SQLDatabase.from_uri(pg_uri)
OPENAI_API_KEY = environ.get("OPENAI_API_KEY")
question = "what is the most common order that customers place?"


def run_chain(question):
    llm = OpenAI(temperature=0, api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

    PROMPT = """ 
    Given an input question, first create a syntactically correct postgresql query to run,  
    then look at the results of the query and return the answer. Remove the ``` in the response.  
    The question: {question}
    """

    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True, top_k=3)
    # use db_chain.run(question) instead if you don't have a prompt
    db_chain.run(PROMPT.format(question=question))


run_chain(question)
