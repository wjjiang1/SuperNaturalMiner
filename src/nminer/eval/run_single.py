"""
Created on Jun 6, 2021

@author: immanueltrummer
"""

import argparse
import nminer.algs.base
import nminer.algs.sample
import nminer.eval.bench
import nminer.algs.rl
import nminer.sql.pred
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import time

from stable_baselines3 import A2C, PPO
from nminer.sql.pred import all_preds
from _collections import defaultdict

import openai
from dotenv import load_dotenv
from os import environ
import json

load_dotenv()

print(nminer.__file__)

OPENAI_API_KEY = environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
username = environ.get("username")
dbname = environ.get("dbname")


def print_details(env):
    """Prints details about RL results.

    Args:
        env: extract info from this environment
    """
    best_summary = env.best_summary()
    print(f'Best summary: "{best_summary}"')
    for best in [True, False]:
        print(f"Top-K summaries (Best: {best})")
        topk_summaries = env.topk_summaries(2, best)
        for s in topk_summaries:
            print(s)

    print("Facts generated:")
    for f in env.fact_to_text.values():
        print(f"{f}")


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


def run_supernaturalminer(
    connection, test_case, all_preds, nr_samples, c_type, cluster
):
    start_s = time.time()
    cmp_pred = test_case["cmp_pred"]
    table = test_case["table"]
    test_copy = test_case.copy()
    del test_copy["cmp_pred"]
    test_copy["cmp_preds"] = [cmp_pred]
    cmp_pred_split = cmp_pred.split("='")
    nl_pattern = cmp_pred_split[1][:-1]

    with psycopg2.connect(
        database=dbname,
        user=username,
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

    message = generate_db_specs(nl_pattern, table, column_info)
    content = message.content.replace("'", '"')
    idx = content.find("=")
    if idx != -1:
        content = content[: idx + 1] + "'" + content[idx + 2 :]
        next_idx = content.find('"', idx + 1)
        content = content[:next_idx] + "'" + content[next_idx + 1 :]
    db_specs = json.loads(content)
    db_specs["table"] = test_copy["table"]
    db_specs["cmp_preds"] = test_copy["cmp_preds"]
    dims_txt = db_specs["dims_txt"]
    del db_specs["dims_txt"]
    db_specs["dims_tmp"] = dims_txt
    del db_specs["target"]
    db_specs["nr_facts"] = test_copy["nr_facts"]
    db_specs["nr_preds"] = test_copy["nr_preds"]
    db_specs["degree"] = test_copy["degree"]
    db_specs["max_steps"] = test_copy["max_steps"]

    env = nminer.algs.rl.PickingEnv(
        connection, **db_specs, all_preds=all_preds, c_type=c_type, cluster=cluster
    )
    model = A2C("MlpPolicy", env, verbose=True, gamma=1.0, normalize_advantage=True)
    model.learn(total_timesteps=nr_samples)
    total_s = time.time() - start_s
    logging.debug(f"Optimization took {total_s} seconds")

    p_stats = {"time": total_s}
    p_stats.update(env.statistics())

    return env.s_eval.text_to_reward, p_stats


def run_rl(connection, test_case, all_preds, nr_samples, c_type, cluster):
    """Benchmarks primary method on test case.

    Args:
        connection: connection to database
        test_case: describes test case
        all_preds: ordered predicates
        c_type: type of cache to create
        cluster: whether to cluster search space

    Returns:
        summaries with reward, performance statistics
    """
    start_s = time.time()
    cmp_pred = test_case["cmp_pred"]
    test_copy = test_case.copy()
    del test_copy["cmp_pred"]
    test_copy["cmp_preds"] = [cmp_pred]
    env = nminer.algs.rl.PickingEnv(
        connection, **test_copy, all_preds=all_preds, c_type=c_type, cluster=cluster
    )
    model = A2C("MlpPolicy", env, verbose=True, gamma=1.0, normalize_advantage=True)
    model.learn(total_timesteps=nr_samples)
    total_s = time.time() - start_s
    logging.debug(f"Optimization took {total_s} seconds")

    p_stats = {"time": total_s}
    p_stats.update(env.statistics())

    return env.s_eval.text_to_reward, p_stats


def run_sampling(connection, test_case, all_preds, c_type):
    """Run sampling algorithm.

    Args:
        connection: connection to database
        test_case: summarize for this test case
        all_preds: all available predicates
        c_type: cache type to use for sampling

    Returns:
        summaries with quality, performance statistics
    """
    sampler = nminer.algs.sample.Sampler(
        connection, test_case, all_preds, 0.01, 5, c_type
    )
    text_to_reward, p_stats = sampler.run_sampling()

    return text_to_reward, p_stats


def run_random(connection, test_case, all_preds, nr_sums, timeout_s):
    """Run simple random generation baseline.

    Args:
        connection: connection to database
        test_case: summarize for this test case
        all_preds: all available predicates
        nr_sums: generate so many summaries
        timeout_s: sample until this timeout

    Returns:
        summaries with quality, performance statistics
    """
    return nminer.algs.base.rand_sums(
        nr_sums=nr_sums,
        timeout_s=timeout_s,
        connection=connection,
        all_preds=all_preds,
        **test_case,
    )


def run_gen(connection, test_case, all_preds, timeout_s):
    """Run generative learning baseline.

    Args:
        connection: connection to database
        test_case: summarize for this test case
        all_preds: all available predicates
        timeout_s: sample until this timeout

    Returns:
        summaries with quality, performance statistics
    """
    return nminer.algs.base.gen_rl(
        timeout_s=timeout_s, connection=connection, all_preds=all_preds, **test_case
    )


def run_viz(connection, test_case, all_preds, timeout_s):
    """Run baseline using Google Vizier platform.

    Args:
        connection: connection to database
        test_case: describes summarization task
        all_preds: all available predicates
        timeout_s: time limit for iterations

    Returns:
        summaries with quality, performance statistics
    """
    return nminer.algs.base.vizier_sums(
        timeout_s=timeout_s, all_preds=all_preds, connection=connection, **test_case
    )


def log_line(outfile, b_id, t_id, nr_facts, nr_preds, m_id, sums, p_stats):
    """Writes one line to log file.

    Args:
        outfile: pointer to output file
        b_id: batch ID
        t_id: test case ID
        nr_facts: summaries use that many facts
        nr_preds: number of predicates per fact
        m_id: method identifier
        sums: maps summaries to rewards
        p_stats: performance statistics
    """
    actual_sums = [i for i in sums.items() if i[0] is not None]
    sorted_sums = sorted(actual_sums, key=lambda s: s[1])
    if sorted_sums:
        b_sum = sorted_sums[-1]
        w_sum = sorted_sums[0]
    else:
        b_sum = ("-", -10)
        w_sum = ("-", -10)

    def_stats = defaultdict(lambda: -1)
    def_stats.update(p_stats)
    e_time = def_stats["evaluation_time"]
    time = def_stats["time"]
    cache_hits = def_stats["cache_hits"]
    cache_misses = def_stats["cache_misses"]
    nr_queries = def_stats["nr_queries"]
    no_merge_cost = def_stats["no_merge_cost"]
    merged_cost = def_stats["merged_cost"]

    outfile.write(
        f"{b_id}\t{t_id}\t{nr_facts}\t{nr_preds}\t"
        f"{m_id}\t{b_sum[0]}\t{b_sum[1]}\t{w_sum[0]}\t{w_sum[1]}\t"
        f"{time}\t{e_time}\t{cache_hits}\t{cache_misses}\t"
        f"{nr_queries}\t{no_merge_cost}\t{merged_cost}\n"
    )
    outfile.flush()


def main():

    parser = argparse.ArgumentParser(description="Run CP benchmark.")
    parser.add_argument("db", type=str, help="database name")
    parser.add_argument("user", type=str, help="database user")
    parser.add_argument("out", type=str, help="result file name")
    parser.add_argument("log", type=str, help="logging level")
    parser.add_argument("samples", type=int, help="Number RL samples")
    args = parser.parse_args()

    db = args.db
    user = args.user
    outpath = args.out
    nr_samples = args.samples

    log_level = getattr(logging, args.log.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log}")
    logging.basicConfig(level=log_level, filemode="w")

    with open(outpath, "w") as file:
        file.write(
            "scenario\ttestcase\tnrfacts\tnrpreds\t"
            "approach\tbest\tbquality\tworst\twquality\t"
            "time\tetime\tchits\tcmisses\tnrqueries\t"
            "nomergecost\tmergedcost\n"
        )
        with psycopg2.connect(
            database=db, user=user, cursor_factory=RealDictCursor
        ) as connection:
            connection.autocommit = True

            test_batches = nminer.eval.bench.generate_testcases()
            for b_id, b in enumerate(test_batches):
                for t_id, t in enumerate(b):
                    all_preds = nminer.sql.pred.all_preds(
                        connection, t["table"], t["dim_cols"], t["cmp_pred"]
                    )
                    pred_cnt = len(all_preds)
                    print(f"Predicate search space size: {pred_cnt}")

                    for nr_facts in [1, 2, 3]:
                        for nr_preds in [1, 2, 3]:

                            print(
                                f"Next: B{b_id}/T{t_id}; " f"{nr_facts}F, {nr_preds}P"
                            )
                            t["nr_facts"] = nr_facts
                            t["nr_preds"] = nr_preds

                            sums, p_stats = run_rl(
                                connection, t, all_preds, nr_samples, "proactive", False
                            )
                            log_line(
                                file,
                                b_id,
                                t_id,
                                nr_facts,
                                nr_preds,
                                "rlNCproactive",
                                sums,
                                p_stats,
                            )

                            sums, p_stats = run_supernaturalminer(
                                connection, t, all_preds, nr_samples, "proactive", False
                            )
                            log_line(
                                file,
                                b_id,
                                t_id,
                                nr_facts,
                                nr_preds,
                                "supernaturalminer_rlNCproactive",
                                sums,
                                p_stats,
                            )

                            # for c_type in ['empty', 'proactive']:
                            for c_type in ["proactive"]:
                                sums, p_stats = run_sampling(
                                    connection, t, all_preds, c_type
                                )
                                log_line(
                                    file,
                                    b_id,
                                    t_id,
                                    nr_facts,
                                    nr_preds,
                                    "sample" + c_type,
                                    sums,
                                    p_stats,
                                )

                                sums, p_stats = run_rl(
                                    connection, t, all_preds, nr_samples, c_type, True
                                )
                                log_line(
                                    file,
                                    b_id,
                                    t_id,
                                    nr_facts,
                                    nr_preds,
                                    "rl" + c_type,
                                    sums,
                                    p_stats,
                                )

                                sums, p_stats = run_supernaturalminer(
                                    connection, t, all_preds, nr_samples, c_type, True
                                )
                                log_line(
                                    file,
                                    b_id,
                                    t_id,
                                    nr_facts,
                                    nr_preds,
                                    "supernaturalminer_rl" + c_type,
                                    sums,
                                    p_stats,
                                )
                            timeout_s = p_stats["time"]

                            sums, p_stats = run_random(
                                connection, t, all_preds, 1, float("inf")
                            )
                            log_line(
                                file,
                                b_id,
                                t_id,
                                nr_facts,
                                nr_preds,
                                "rand1",
                                sums,
                                p_stats,
                            )

                            sums, p_stats = run_random(
                                connection, t, all_preds, float("inf"), timeout_s
                            )
                            log_line(
                                file,
                                b_id,
                                t_id,
                                nr_facts,
                                nr_preds,
                                "rand",
                                sums,
                                p_stats,
                            )

                            sums, p_stats = run_gen(connection, t, all_preds, timeout_s)
                            log_line(
                                file,
                                b_id,
                                t_id,
                                nr_facts,
                                nr_preds,
                                "gen",
                                sums,
                                p_stats,
                            )

                            # sums, p_stats = run_viz(connection, t, all_preds, timeout_s)
                            # log_line(
                            #     file,
                            #     b_id,
                            #     t_id,
                            #     nr_facts,
                            #     nr_preds,
                            #     "viz",
                            #     sums,
                            #     p_stats,
                            # )


if __name__ == "__main__":
    main()
