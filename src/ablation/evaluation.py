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
        case "ssfgm":
            return "SSFGM"
        case "r":
            return "$r$"
        case "c":
            return "$C$"
    return label.title()


def create_plot(
    data: pd.DataFrame, param_key: str, dependent_key: str, legend_key: str, filename: str
) -> None:
    markers = itertools.cycle(['o', 's', '*', 'X', '^', 'D', 'v', "P"])
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

    r_data = sensitivity_data.query("`attack` == 'lie'")
    r_data = r_data.drop(
        columns=["attack", "seed", "repetitions", "npoints", "dimensions"]
    )
    print("r data")
    print_latex(r_data.corr())
    create_plot(r_data, "r", "error", "padversaries", "r_err.pdf")
    create_plot(r_data, "r", "improvement", "padversaries", "r_imp.pdf")
    print()

    ablation_data = pd.read_csv("results/ablation.csv")
    ablation_data = ablation_data.query("`aggregator` == 'ssfgm'")
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
