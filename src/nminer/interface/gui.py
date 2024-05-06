"""
Created on May 4, 2022

@author: immanueltrummer
Modified by Travis Zhang
"""

import json
import os
import pandas as pd
import pathlib
import psycopg2.extras
import stable_baselines3
import streamlit as st
import sys
import time

import openai
from os import environ
from dotenv import load_dotenv
from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

cur_file_dir = os.path.dirname(__file__)
src_dir = pathlib.Path(cur_file_dir).parent.parent
sys.path.append(str(src_dir))
print(sys.path)

load_dotenv()

OPENAI_API_KEY = environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
username = environ.get("username")
password = environ.get("password")
host = environ.get("host")
port = environ.get("port")
dbname = environ.get("dbname")
pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}"
db = SQLDatabase.from_uri(pg_uri)

import nminer.algs.rl
import nminer.sql.pred
import nminer.text.sum

st.set_page_config(page_title="NaturalMiner")
st.markdown(
    """
NaturalMiner mines large data sets for patterns described in natural language.
"""
)

root_dir = src_dir.parent
scenarios_path = root_dir.joinpath("demo").joinpath("scenarios.json")
with open(scenarios_path) as file:
    scenarios = json.load(file)
nr_scenarios = len(scenarios)
print("Example scenarios loaded")

selected = st.selectbox(
    "Select Example (Optional)",
    options=range(nr_scenarios),
    format_func=lambda idx: scenarios[idx]["scenario"],
)

with st.expander("Data Source"):
    connection_info = st.text_input(
        'Database connection details (format: "<Database>:<User>:<Password>"):',
        value=scenarios[selected]["dbconnection"],
    ).split(":")
    db_name = connection_info[0]
    db_user = connection_info[1]
    db_pwd = connection_info[2] if len(connection_info) > 1 else ""
    table = st.text_input(
        "Name of database table to analyze:",
        max_chars=100,
        value=scenarios[selected]["table"],
    )


def generate_data_response(db, input_text):
    PROMPT = """
    Given an input text, first create a syntactically correct postgresql query to run,
    then look at the results of the query and return the answer. Remove the ``` in the response.
    The input text: {input_text}
    """
    llm = OpenAI(temperature=0, api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True, top_k=3)
    data_text = db_chain.run(PROMPT.format(input_text=input_text))
    return data_text


label = st.text_input("Input:", max_chars=100, value=scenarios[selected]["goal"])

nr_iterations = st.slider(
    "Number of iterations:", min_value=1, max_value=500, value=200
)

print("Generated input elements")


def compare_outputs(input_text, semantic_output, data_output):
    completion = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a highly efficient database expert that will read the definitions for each of the variables given and fill in the missing values to the best of your ability. You have read and understand the paper Demonstrating NaturalMiner: Searching Large Data Sets for Abstract Patterns Described in Natural Language and are now trying to fill in the missing variables based on this paper.",
            },
            {
                "role": "user",
                "content": "I require a leaderboard for two large language model outputs. I'll provide you with prompts given to these models and their corresponding responses. Your task is to assess these responses, ranking the models in order of preference from a human perspective. Once ranked, please output the results in a structured JSON format.\n\n## Prompt\n{\n 'input_text': '"
                + input_text
                + "',\n 'instruction': 'Evaluate the two outputs based on the given input_text. Write 1 in Preference if output_1 is better or 2 in Preference if output_2 is better'"
                + ",\n 'output_1': '"
                + semantic_output
                + "',\n 'output_2': '"
                + data_output
                + '\n}\n\n## Model Outputs Preference: "" ##Task\nEvaluate and rank the models based on the instruction above.',
            },
        ],
    )

    return completion.choices[0].message


def generate_db_specs(nl_pattern, table_name, table_cols):

    completion = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0125:personal:test-1:9LcLIXjl",
        messages=[
            {
                "role": "system",
                "content": "You are a highly efficient database expert that will read the definitions for each of the variables given and fill in the missing values to the best of your ability. You have read and understand the paper Demonstrating NaturalMiner: Searching Large Data Sets for Abstract Patterns Described in Natural Language and are now trying to fill in the missing variables based on this paper.",
            },
            {
                "role": "user",
                "content": "I require a filled-in dictionary based on the nl_pattern and the columns of the table provided.\nYour task is to understand the nl_pattern and using your understanding of each variable, fill the dictionary in based on the given nl_pattern.\nOnce filled, please output the results in a structured JSON format.\n\nThe following variables below are defined in order for you to better understand how to fill in the variables.\n\n1. nl_pattern: Search for facts that comply with a pattern, described in natural language. E.g., we can search for arguments for or against buying this laptop\n2. dim_cols (list): Table columns (typically categorical) for equality predicates\n3. dims_txt (list): Text templates to express equality predicates associated with dimension columns. The <V> placeholder will be substituted by the predicate constant\n4. agg_cols (list): Table columns (typically numerical) for calculating aggregates\n5. aggs_txt (list): Text templates used to express aggregates on aforementioned columns\n6. target (string): An SQL predicate describing the target data: we mine for facts that compare rows satisfying that predicate to all rows. Here, target data is associated with a specific laptop model and facts compare this to other laptops\n7. preamble (string): All facts start with this text, specifying what we compare to\n8. table_columns (list): A list of the DB table columns\n9. table_name (string): The name of the table\n\nThe length of dim_cols must match the length of dims_txt and the length of agg_cols must match the length of aggs_txt\n\n{\n 'nl_pattern': "
                + nl_pattern
                + ",\n 'table_columns': ["
                + ", ".join(table_cols)
                + "],\n 'table_name': "
                + table_name
                + "\n}\n\nHere is what an example output might look like\n\n{\n 'dim_cols': '{dim_cols}',\n 'dims_txt': '{dims_txt}',\n 'agg_cols': '{agg_cols}',\n 'aggs_txt': '{aggs_txt}',\n 'target': '{target}',\n 'preamble': '{preamble}'\n}\n\nFill in dim_cols, dims_txt, agg_cols, aggs_txt, target, and preamble.",
            },
        ],
    )

    return completion.choices[0].message


if st.button("Find Pattern!"):
    print("Generate data text stats")
    st.write("Generate data text stats...")

    data_message = generate_data_response(db, label)
    print(data_message)
    data_df = pd.DataFrame(columns=["output"], index=range(0))
    new_row = pd.DataFrame([[data_message]], columns=["output"])
    result_table = st.table(data_df)
    result_table.add_rows(new_row)

    print("Generating DB specifications...")
    st.write("Generating DB specifications...")

    with psycopg2.connect(
        database=db_name,
        user=db_user,
        cursor_factory=psycopg2.extras.RealDictCursor,
    ) as connection:
        with connection.cursor() as cursor:
            # Define the query to get column information
            query = """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s;
            """

            # Execute the query
            cursor.execute(query, (table,))

            # Fetch all rows
            rows = cursor.fetchall()

            # Create a list of strings containing column name and type
            column_info = [f"{row['column_name']} {row['data_type']}" for row in rows]

    message = generate_db_specs(label, table, column_info)
    content = message.content.replace("'", '"')
    idx = content.find("=")
    if idx != -1:
        content = content[: idx + 1] + "'" + content[idx + 2 :]
        next_idx = content.find('"', idx + 1)
        content = content[:next_idx] + "'" + content[next_idx + 1 :]
    db_specs = json.loads(content)

    print(content)
    print(db_specs)

    dim_cols = db_specs["dim_cols"]
    dims_txt = db_specs["dims_txt"]
    dims_info = [f"{x}:{y}" for x, y in zip(dim_cols, dims_txt)]

    agg_cols = db_specs["agg_cols"]
    aggs_txt = db_specs["aggs_txt"]
    aggs_info = [f"{x}:{y}" for x, y in zip(agg_cols, aggs_txt)]

    preamble = db_specs["preamble"]
    cmp_pred = db_specs["target"]

    nr_facts, nr_preds = 3, 3

    print("Searching for pattern ...")

    result_cols = ["Predicate", "Facts", "Quality"]
    result_df = pd.DataFrame(columns=result_cols, index=range(0))
    result_table = st.table(result_df)

    st.write(f'Analyzing data satisfying predicate "{cmp_pred}" ...')
    dims_col_text = [d.split(":") for d in dims_info]
    aggs_col_text = [a.split(":") for a in aggs_info]
    t = {
        "table": table,
        "dim_cols": [d[0] for d in dims_col_text],
        "agg_cols": [a[0] for a in aggs_col_text],
        "cmp_preds": [cmp_pred],
        "nr_facts": nr_facts,
        "nr_preds": nr_preds,
        "degree": 5,
        "max_steps": nr_iterations,
        "preamble": preamble,
        "dims_tmp": [d[1] for d in dims_col_text],
        "aggs_txt": [a[1] for a in aggs_col_text],
    }
    with psycopg2.connect(
        database=db_name,
        user=db_user,
        cursor_factory=psycopg2.extras.RealDictCursor,
    ) as connection:
        connection.autocommit = True

        start_s = time.time()
        all_preds = nminer.sql.pred.all_preds(
            connection, t["table"], t["dim_cols"], cmp_pred
        )
        sum_eval = nminer.text.sum.SumEvaluator(1, "facebook/bart-large-mnli", label)
        env = nminer.algs.rl.PickingEnv(
            connection,
            **t,
            all_preds=all_preds,
            c_type="proactive",
            cluster=True,
            sum_eval=sum_eval,
        )
        model = stable_baselines3.A2C(
            "MlpPolicy", env, verbose=True, gamma=1.0, normalize_advantage=True
        )
        model.learn(total_timesteps=nr_iterations)
        total_s = time.time() - start_s
        rated_sums = env.s_eval.text_to_reward
        actual_sums = [i for i in rated_sums.items() if i[0] is not None]
        sorted_sums = sorted(actual_sums, key=lambda s: s[1])
        if sorted_sums:
            b_sum = sorted_sums[-1]
        else:
            b_sum = ("(No valid summary generated)", -10)

        new_row = pd.DataFrame([[cmp_pred, b_sum[0], b_sum[1]]], columns=result_cols)
        print(new_row)
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    result_table.add_rows(result_df)

    print(b_sum[0])
    print(data_message)

    message = compare_outputs(label, b_sum[0], data_message)
    print(message)
    output = json.loads(message.content)
    pref = message.content

    st.write("All data subsets generated!")
