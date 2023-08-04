import os
import json
import pandas as pd
import numpy as np


def make_leaves_lists(data):
    improved_data = {}
    for environment, results in data.items():
        improved_data[environment] = {}
        for train_or_test, train_or_test_results in results.items():
            listed_results = {k: [] for k in train_or_test_results[0].keys()}
            for totr in train_or_test_results:
                for k, v in totr.items():
                    listed_results[k].append(v)
            improved_data[environment][train_or_test] = listed_results
    return improved_data


def format_final_table(styler):
    styler = styler.hide()
    styler = styler.format(formatter=lambda s: s.replace('%', '\\%'))
    return styler


if __name__ == "__main__":
    json_files = [f for f in os.listdir('results') if 'gradient_similarity' in f]
    tabular_data = {}
    for json_file in json_files:
        with open(f"results/{json_file}", 'r') as f:
            data = make_leaves_lists(json.load(f))
        for environment, results in data.items():
            env_name = f"{json_file.replace('_gradient_similarity.json', '')}:{environment}"
            env_name = env_name.replace('gradient_similarity.json', 'small_mu2')
            env_name = env_name.replace('_', ' ')
            tabular_data[env_name] = {}
            for train_or_test, train_or_test_results in results.items():
                for k, v in train_or_test_results.items():
                    p = train_or_test == "train" and k == "cosinesimilarity"
                    q = train_or_test == "test" 
                    if p or q:
                        if k in ["accuracy", "cosinesimilarity"]:
                            tabular_data[env_name][k] = f"{np.mean(v):.3%} ({np.std(v):.3%})"
                        else:
                            tabular_data[env_name][k] = f"{np.mean(v):.3f} ({np.std(v):.3f})"
    df = pd.DataFrame(tabular_data).T
    df = df.reset_index()
    df[['environment', 'algorithm']] = df['index'].str.split(":", expand=True)    
    cols = df.columns[-2:].tolist() + df.columns[:-2].tolist()
    df = df[cols]
    df = df.drop(columns="index")
    df = df.style.pipe(format_final_table).to_latex(position_float='centering')
    print(df)

