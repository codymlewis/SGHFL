import itertools
import os
import re
import matplotlib.pyplot as plt
import pandas as pd


def process_label(label: str | int | float) -> str:
    if not isinstance(label, str):
        return label
    if re.match(r"e\d", label):
        return f"$e_{label[-1]}$"
    match label:
        case "padversaries":
            return "Adversaries"
        case "npoints":
            return "Points"
        case "trmean":
            return "TrMean"
    return label.title()


def create_plot(
    data: pd.DataFrame, param_key: str, dependent_key: str, legend_key: str, filename: str
) -> None:
    markers = itertools.cycle(['o', 's', '*', 'x', '^', '2', 'v'])
    for legend in pd.unique(data[legend_key]):
        legend_data = data.query(
            f"""`{legend_key}` == {f"'{legend}'" if isinstance(legend, str) else legend}"""
        )
        plt.plot(
            legend_data[param_key],
            legend_data[f"{dependent_key} mean"],
            label=process_label(legend),
            marker=next(markers)
        )
        plt.fill_between(
            legend_data[param_key],
            legend_data[f"{dependent_key} mean"] - legend_data[f"{dependent_key} std"],
            legend_data[f"{dependent_key} mean"] + legend_data[f"{dependent_key} std"],
            alpha=0.2,
        )
    plt.legend(title=process_label(legend_key))
    plt.xlabel(process_label(param_key))
    plt.ylabel(process_label(dependent_key))
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{filename}")
    plt.clf()
    print(f"Saved a plot to plots/{filename}")


def print_latex(data: pd.DataFrame, hide_index: bool = False) -> None:
    styler_output = data.style
    if hide_index:
        styler_output = styler_output.hide()
    print(styler_output.to_latex())


if __name__ == "__main__":
    sensitivity_data = pd.read_csv("results/sensitivity.csv")
    sensitivity_data = sensitivity_data.drop_duplicates()

    e1_data = sensitivity_data.query("`attack` == 'lie'")
    e1_data = e1_data.drop(columns=["attack", "seed", "repetitions", "npoints", "dimensions", "e2", "K"])
    print_latex(e1_data.corr())
    create_plot(e1_data, "e1", "error", "padversaries", "e1_vs_padv_err.pdf")
    create_plot(e1_data, "e1", "improvement", "padversaries", "e1_vs_padv_imp.pdf")

    e2_data = sensitivity_data.query("`attack` == 'shifted_random' and `K` == 3 and `npoints` == 1000")
    e2_data = e2_data.drop(columns=["attack", "seed", "repetitions", "npoints", "e1", "K"])
    print("Full e2 data:")
    print_latex(e2_data.corr())
    print()
    e2_padv_data = e2_data.query("`dimensions` == 2")
    e2_padv_data = e2_padv_data.drop(columns=["dimensions"])
    print_latex(e2_padv_data.corr())

    e2_dim_data = e2_data.query("`padversaries` == 0.4")
    e2_dim_data = e2_dim_data.drop(columns=["padversaries"])
    print_latex(e2_dim_data.corr())
    create_plot(e2_dim_data, "e2", "error", "dimensions", "e2_vs_dims_err.pdf")
    create_plot(e2_dim_data, "e2", "improvement", "dimensions", "e2_vs_dims_imp.pdf")

    K_data = sensitivity_data.query("`attack` == 'shifted_random' and `e1` == 0.01 and `e2` == 1.0")
    K_data = K_data.drop(columns=["attack", "seed", "repetitions", "e1", "e2"])
    print("Full K data:")
    print_latex(K_data.corr())
    print()
    K_dim_data = K_data.query("`npoints` == 1000 and `padversaries` == 0.4")
    K_dim_data = K_dim_data.drop(columns=["npoints", "padversaries"])
    print_latex(K_dim_data.corr())

    K_npoints_data = K_data.query("`dimensions` == 2 and `padversaries` == 0.4")
    K_npoints_data = K_npoints_data.drop(columns=["dimensions", "padversaries"])
    print_latex(K_npoints_data.corr())
    create_plot(K_npoints_data, "K", "error", "npoints", "K_vs_npoints_err.pdf")
    create_plot(K_npoints_data, "K", "improvement", "npoints", "K_vs_npoints_imp.pdf")

    K_padv_data = K_data.query("`npoints` == 1000 and `dimensions` == 2")
    K_padv_data = K_padv_data.drop(columns=["npoints", "dimensions"])
    print_latex(K_padv_data.corr())
    create_plot(K_padv_data, "K", "error", "padversaries", "K_vs_padv_err.pdf")
    create_plot(K_padv_data, "K", "improvement", "padversaries", "K_vs_padv_imp.pdf")

    ablation_data = pd.read_csv("results/ablation.csv")
    ablation_data = ablation_data.query("`aggregator` == 'topomean'")
    ablation_data = ablation_data.drop(columns=['seed', 'repetitions', 'aggregator'])
    print("10% Adversaries ablation:")
    print_latex(ablation_data.query("`padversaries` == 0.1").drop(columns="padversaries"), hide_index=True)
    print()
    print("40% Adversaries ablation:")
    print_latex(ablation_data.query("`padversaries` == 0.4").drop(columns="padversaries"), hide_index=True)

    comparison_data = pd.read_csv("results/comparison.csv")
    comparison_data = comparison_data.drop(columns=['seed', 'repetitions', 'npoints'])
    create_plot(comparison_data, "padversaries", "error", "aggregator", "comparison_err.pdf")
    create_plot(comparison_data, "padversaries", "improvement", "aggregator", "comparison_imp.pdf")
