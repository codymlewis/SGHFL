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


def gradsim_env_process(json_file, environment):
    env_name = f"{json_file.replace('_gradient_similarity.json', '')}:{environment}"
    env_name = env_name.replace('gradient_similarity.json', 'small_mu2')
    env_name = env_name.replace('_', ' ')
    return env_name


def fairness_env_process(json_file, environment):
    env_name = f"{json_file.replace('fairness_', '')}:{environment}"
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
    parser = argparse.ArgumentParser(description="Create a LaTeX from the experiment results.")
    parser.add_argument("-f", "--fairness", action="store_true", help="Use the fairness experiment results.")
    parser.add_argument("-g", "--gradient-similarity", action="store_true", help="Use the gradient similarity experiment results.")
    args = parser.parse_args()

    if args.gradient_similarity:
        json_files = [f for f in os.listdir('results') if 'gradient_similarity' in f]
        env_process = gradsim_env_process

    if args.fairness:
        json_files = [f for f in os.listdir('results') if 'fairness' in f]
        env_process = fairness_env_process

    tabular_data = {}
    for json_file in json_files:
        with open(f"results/{json_file}", 'r') as f:
            data = make_leaves_lists(json.load(f))
        if args.gradient_similarity:
            for environment, results in data.items():
                env_name = env_process(json_file, environment)
                tabular_data[env_name] = process_train_test_results(results)
        elif args.fairness:
            env_name = env_process(json_file, "")
            tabular_data[env_name] = process_train_test_results(data)

    df = pd.DataFrame(tabular_data).T
    df = df.reset_index()
    df[['environment', 'algorithm']] = df['index'].str.split(":", expand=True)    
    cols = df.columns[-2:].tolist() + df.columns[:-2].tolist()
    df = df[cols]
    df = df.drop(columns="index")
    df = df.style.pipe(format_final_table).to_latex(position_float='centering')
    print(df)

