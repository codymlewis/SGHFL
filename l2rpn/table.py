import argparse
import functools
import pandas as pd


def format_final_table(styler):
    styler = styler.hide()
    styler = styler.format(formatter=lambda s: str(s).replace('%', '\\%'))
    return styler


def comparator(a, b):
    cval = compare_by_keyword(a, b, "mae")
    if cval:
        return cval
    cval = compare_by_keyword(a, b, "rmse")
    if cval:
        return cval
    cval = compare_by_keyword(a, b, "r2_score")
    if cval:
        return cval
    cval = compare_by_keyword(a, b, "cosine_similarity")
    return cval


def compare_by_keyword(a, b, keyword):
    if "dropped" in a and "dropped" not in b:
        return 1
    if "dropped" not in a and "dropped" in b:
        return -1

    if keyword in a or keyword in b:
        if keyword in a and keyword in b:
            return -1 if "std" in b else 1
        return -1 if keyword in a else 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a LaTeX table from the L2RPN experiment results.")
    parser.add_argument("-f", "--fairness", action="store_true", help="Use the fairness experiment results.")
    parser.add_argument("-p", "--performance", action="store_true", help="Use the performance experiment results.")
    parser.add_argument("-a", "--attack", action="store_true", help="Use the attack experiment results.")
    args = parser.parse_args()

    keyword = 'performance' if args.performance else 'fairness' if args.fairness else 'attack'
    df = pd.read_csv(f"results/l2rpn_{keyword}.csv")
    df = df.drop(columns=[
        "fairness",
        "seed",
        "episodes",
        "timesteps",
        "rounds",
        "forecast_window",
        "batch_size",
        "pct_adversaries",
        "num_middle_servers",
    ])

    if keyword == "attack":
        df = df.drop(columns=[
            "middle_server_aggregator",
            "middle_server_km",
            "middle_server_fp",
            "middle_server_mrcs",
            "intermediate_finetuning",
        ])
        grouping_cols = ["attack", "server_aggregator"]
    elif keyword == "fairness":
        df = df.drop(columns=[
            "attack",
            "server_aggregator",
        ])
        df.middle_server_aggregator += [" with km" if km else "" for km in df.middle_server_km]
        df.middle_server_aggregator += [" with fedprox" if fp else "" for fp in df.middle_server_fp]
        df.middle_server_aggregator += [" with mrcs" if mrcs else "" for mrcs in df.middle_server_mrcs]
        df.middle_server_aggregator += [" with if" if inter_ft > 0 else "" for inter_ft in df.intermediate_finetuning]
        df = df.drop(columns=["middle_server_km", "middle_server_fp", "middle_server_mrcs", "intermediate_finetuning",])
        grouping_cols = ["middle_server_aggregator"]
    else:
        df = df.drop(columns=[
            "attack",
            "middle_server_fp",
            "middle_server_mrcs",
            "server_aggregator",
        ])
        df.middle_server_aggregator += [" with km" if km else "" for km in df.middle_server_km]
        df.middle_server_aggregator += [" with if" if inter_ft > 0 else "" for inter_ft in df.intermediate_finetuning]
        df = df.drop(columns=["middle_server_km", "intermediate_finetuning"])
        grouping_cols = ["middle_server_aggregator"]
    grouped_df = df.groupby(grouping_cols)
    mean_df = grouped_df.mean().reset_index()
    std_df = grouped_df.std().reset_index()
    std_df = std_df.rename(columns={k: f"{k} std" for k in set(std_df.columns) - set(grouping_cols)})
    full_df = mean_df.merge(std_df)
    full_df = full_df.reindex(
        grouping_cols + sorted(list(set(full_df.columns) - set(grouping_cols)), key=functools.cmp_to_key(comparator)),
        axis=1
    )
    print(full_df.style.pipe(format_final_table).to_latex(position_float='centering'))
