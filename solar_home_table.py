import argparse
import os
import json
import re
import pandas as pd


def env_process(json_file):
    env_name = json_file[json_file.find('_', 11):].replace('.json', '')
    env_name = env_name.replace('_', ' ')
    # env_name = re.sub("mu1=.+", "momentum", env_name)
    env_name = re.sub("num finetune episodes=\d", "intermediate finetuning", env_name)
    return env_name


def process_data(data):
    results = {}
    for location, loc_data in data.items():
        for k, v in loc_data.items():
            if k in ["cosine similarity"]:
                v = f"{v:.3%}"
            else:
                v = f"{v:.3g}"
            results[f"{location} {k}"] = v
    return results


def format_final_table(styler):
    styler = styler.hide()
    styler = styler.format(formatter=lambda s: s.replace('%', '\\%'))
    return styler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a LaTeX table from the solar home experiment results.")
    parser.add_argument("-f", "--fairness", action="store_true", help="Use the fairness experiment results.")
    parser.add_argument("-p", "--performance", action="store_true", help="Use the performance experiment results.")
    parser.add_argument("-a", "--attack", action="store_true", help="Use the attack experiment results.")
    args = parser.parse_args()

    keyword = 'performance' if args.performance else 'fairness' if args.fairness else 'attack'
    json_files = [f for f in os.listdir("results") if ("solar_home" in f and keyword in f)]

    tabular_data = {}
    for json_file in json_files:
        with open(f"results/{json_file}", 'r') as f:
            data = json.load(f)
        env_name = env_process(json_file)
        tabular_data[env_name] = process_data(data)

    df = pd.DataFrame(tabular_data).T
    df = df.reset_index()
    df = df.rename({"index": "environment"}, axis=1)
    df = df.drop(columns=[
        c for c in df.columns if ("train" in c and "cosine similarity" not in c) or "centralised" in c
    ])
    df = df.sort_values('environment')
    df = df.style.pipe(format_final_table).to_latex(position_float='centering')
    print(df)
