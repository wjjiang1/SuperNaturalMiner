from langchain import OpenAI, SQLDatabase
from langchain.chains import SQLDatabaseSequentialChain
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')

PROMPT = """ 
Given an input question, first create a syntactically correct postgresql query to run,  
then look at the results of the query and return the answer.  
The question: {question}
"""

llm.