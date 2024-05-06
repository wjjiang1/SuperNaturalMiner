from openai import OpenAI
from os import environ
from dotenv import load_dotenv

load_dotenv()

# Load Environment Variables
OPENAI_API_KEY = environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:personal:test-1:9LcLIXjl",
    messages=[
        {
            "role": "system",
            "content": "You are a highly efficient database expert that will read the definitions for each of the variables given and fill in the missing values to the best of your ability. You have read and understand the paper Demonstrating NaturalMiner: Searching Large Data Sets for Abstract Patterns Described in Natural Language and are now trying to fill in the missing variables based on this paper.",
        },
        {
            "role": "user",
            "content": "I require a filled-in dictionary based on the nl_pattern and the columns of the table provided.\nYour task is to understand the nl_pattern and using your understanding of each variable, fill the dictionary in based on the given nl_pattern.\nOnce filled, please output the results in a structured JSON format.\n\nThe following variables below are defined in order for you to better understand how to fill in the variables.\n\n1. nl_pattern: Search for facts that comply with a pattern, described in natural language. E.g., we can search for arguments for or against buying this laptop\n2. dim_cols (list): Table columns (typically categorical) for equality predicates\n3. dims_txt (list): Text templates to express equality predicates associated with dimension columns. The <V> placeholder will be substituted by the predicate constant\n4. agg_cols (list): Table columns (typically numerical) for calculating aggregates\n5. aggs_txt (list): Text templates used to express aggregates on aforementioned columns\n6. target (string): An SQL predicate describing the target data: we mine for facts that compare rows satisfying that predicate to all rows. Here, target data is associated with a specific laptop model and facts compare this to other laptops\n7. preamble (string): All facts start with this text, specifying what we compare to\n8. table_columns (list): A list of the DB table columns\n9. table_name (string): The name of the table\n\nThe length of dim_cols must match the length of dims_txt and the length of agg_cols must match the length of aggs_txt\n\n{\n 'nl_pattern': arguments for buying MateBook D Volta,\n 'table_columns': [brand text, laptop_name text, display_size numeric, processor_type text, graphics_card text, disk_space text, discount_price numeric, old_price numeric, ratings_5max text],\n 'table_name':'laptops'\n}\n\nHere is what an example output might look like\n\n{\n 'dim_cols': '{dim_cols}',\n 'dims_txt': '{dims_txt}',\n 'agg_cols': '{agg_cols}',\n 'aggs_txt': '{aggs_txt}',\n 'target': '{target}',\n 'preamble': '{preamble}'\n}\n\nFill in dim_cols, dims_txt, agg_cols, aggs_txt, target, and preamble.",
        },
    ],
)

print(completion.choices[0].message)
