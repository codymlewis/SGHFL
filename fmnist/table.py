from functools import partial
import argparse
import os
import json
import re
import pandas as pd
import numpy as np


def make_leaves_lists(data):
    improved_data = {}
    for k, v in data.items():
        improved_data[k] = {}
        if isinstance(v, dict):
            improved_data[k] = make_leaves_lists(v)
        else:
            improved_data[k] = {
                vk: {vvk: [] for vvk in vv.keys()} if isinstance(vv, dict) else [] for vk, vv in v[0].items()
            }
            for vitem in v:
                for ki, vi in vitem.items():
                    if isinstance(vi, dict):
                        for vik, viv in vi.items():
                            improved_data[k][ki][vik].append(viv)
                    else:
                        improved_data[k][ki].append(vi)
    return improved_data


def format_final_table(styler):
    styler = styler.hide()
    styler = styler.format(formatter=lambda s: str(s).replace('%', '\\%'))
    return styler


def env_process_fn(json_file, keyword=""):
    env_name = f"{json_file.replace(f'{keyword}_', '')}"
    env_name = env_name.replace('.json', '')
    return env_name


def process_train_test_results(data):
    new_results = {}
    for train_or_test, train_or_test_results in data.items():
        for k, v in train_or_test_results.items():
            if isinstance(v, dict):
                for vk, vv in v.items():
                    p = train_or_test == "train" and vk == "cosinesimilarity"
                    q = train_or_test == "test"
                    if p or q:
                        if vk in ["accuracy", "dropped accuracy", "cosinesimilarity", "asr"]:
                            new_results[f"{k} {vk}"] = f"{np.mean(vv):.3%} ({np.std(vv):.3%})"
                        else:
                            new_results[f"{k} {vk}"] = f"{np.mean(vv):.3g} ({np.std(vv):.3g})"
            else:
                p = train_or_test == "train" and k == "cosinesimilarity"
                q = train_or_test == "test"
                if p or q:
                    if k in ["accuracy", "cosinesimilarity", "asr"]:
                        new_results[k] = f"{np.mean(v):.3%} ({np.std(v):.3%})"
                    else:
                        new_results[k] = f"{np.mean(v):.3g} ({np.std(v):.3g})"
    return new_results


def process_fairness_environment(env_data):
    environment = env_data[re.search('aggregator=', env_data).end():re.search('aggregator=.*_?(.*_)?', env_data).end()]
    return environment


def process_attack_environment(env_data):
    attack = env_data[re.search('attack=', env_data).end():re.search(r'attack=\w+_', env_data).end() - 1]
    if "from_y" in env_data:
        attack = f"Backdoor {attack}"
    aggregator = env_data[re.search('aggregator=', env_data).end():re.search(r'aggregator=[A-Za-z]+_', env_data).end() - 1]
    return f"{attack}, {aggregator}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a LaTeX table from the experiment results.")
    parser.add_argument("-f", "--fairness", action="store_true", help="Use the fairness experiment results.")
    parser.add_argument("-p", "--performance", action="store_true", help="Use the performance experiment results.")
    parser.add_argument("-a", "--attack", action="store_true", help="Use the attack experiment results.")
    args = parser.parse_args()

    keyword = 'performance' if args.performance else 'fairness' if args.fairness else 'attack'
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

    if args.fairness:
        df['environment'] = df['index'].apply(process_fairness_environment)
    elif args.attack:
        df['environment'] = df['index'].apply(process_attack_environment)
    else:
        df['environment'] = df['index']

    cols = df.columns[-1:].tolist() + df.columns[:-1].tolist()
    df = df[cols]
    df = df.drop(columns="index")
    df = df.drop(columns=[c for c in df.columns if "std mean" in c or "std std" in c])
    df = df.sort_values('environment')

    if args.attack:
        backdoor_rows = df['environment'].str.contains("Backdoor")
        bd_df = df[backdoor_rows]
        no_bd_df = df[~backdoor_rows]
        no_bd_df = no_bd_df.drop(columns="asr")
        bd_df = bd_df.style.pipe(format_final_table).to_latex(position_float='centering')
        no_bd_df = no_bd_df.style.pipe(format_final_table).to_latex(position_float='centering')
        print("Backdoor results:")
        print(bd_df)
        print()
        print("Other results:")
        print(no_bd_df)
    else:
        df = df.style.pipe(format_final_table).to_latex(position_float='centering')
        print(df)
