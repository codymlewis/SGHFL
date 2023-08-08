from functools import partial
import argparse
import os
import json
import pandas as pd
import numpy as np


def make_leaves_lists(data):
    improved_data = {}
    for k, v in data.items():
        improved_data[k] = {}
        if isinstance(v, dict):
            improved_data[k] = make_leaves_lists(v)
        else:
            improved_data[k] = {vk: [] for vk in v[0].keys()}
            for vitem in v:
                for ki, vi in vitem.items():
                    improved_data[k][ki].append(vi)
    return improved_data


def format_final_table(styler):
    styler = styler.hide()
    styler = styler.format(formatter=lambda s: s.replace('%', '\\%'))
    return styler


def env_process_fn(json_file, keyword=""):
    env_name = f"{json_file.replace(f'{keyword}_', '')}"
    env_name = env_name.replace('.json', '')
    return env_name


def process_train_test_results(data):
    new_results = {}
    for train_or_test, train_or_test_results in data.items():
        for k, v in train_or_test_results.items():
            p = train_or_test == "train" and k == "cosinesimilarity"
            q = train_or_test == "test" 
            if p or q:
                if k in ["accuracy", "cosinesimilarity"]:
                    new_results[k] = f"{np.mean(v):.3%} ({np.std(v):.3%})"
                else:
                    new_results[k] = f"{np.mean(v):.3f} ({np.std(v):.3f})"
    return new_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a LaTeX table from the experiment results.")
    parser.add_argument("-f", "--fairness", action="store_true", help="Use the fairness experiment results.")
    parser.add_argument("-p", "--performance", action="store_true", help="Use the performance experiment results.")
    args = parser.parse_args()

    keyword = 'performance' if args.performance else 'fairness'
    json_files = [f for f in os.listdir('results') if keyword in f]
    env_process = partial(env_process_fn, keyword=keyword)

    tabular_data = {}
    for json_file in json_files:
        with open(f"results/{json_file}", 'r') as f:
            data = make_leaves_lists(json.load(f))
        env_name = env_process(json_file)
        tabular_data[env_name] = process_train_test_results(data)

    df = pd.DataFrame(tabular_data).T
    df = df.reset_index()
    df['environment'] = df['index'] 
    cols = df.columns[-1:].tolist() + df.columns[:-1].tolist()
    df = df[cols]
    df = df.drop(columns="index")
    df = df.style.pipe(format_final_table).to_latex(position_float='centering')
    print(df)

