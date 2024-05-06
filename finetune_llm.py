from openai import OpenAI
from os import environ
from dotenv import load_dotenv

load_dotenv()

# Load Environment Variables
OPENAI_API_KEY = environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

fileObject = client.files.create(file=open("data.jsonl", "rb"), purpose="fine-tune")
client.fine_tuning.jobs.create(training_file=fileObject.id, model="gpt-3.5-turbo-0125")
