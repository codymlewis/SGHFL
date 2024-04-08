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

    e1_data = sensitivity_data.query("`attack` == 'lie'")
    e1_data = e1_data.drop(
        columns=["attack", "seed", "repetitions", "npoints", "dimensions", "e2", "c", "overlap_scaling_function"]
    )
    print("e1 data")
    print_latex(e1_data.corr())
    create_plot(e1_data, "e1", "error", "padversaries", "e1_err.pdf")
    create_plot(e1_data, "e1", "improvement", "padversaries", "e1_imp.pdf")
    print()

    e2_data = sensitivity_data.query("`attack` == 'shifted_random' and `e1` == 0.01 and `c` == 0.5")
    e2_data = e2_data.drop(
        columns=["attack", "seed", "repetitions", "npoints", "e1", "c", "dimensions", "overlap_scaling_function"]
    )
    print("e2 data:")
    print_latex(e2_data.corr())
    create_plot(e2_data, "e2", "error", "padversaries", "e2_err.pdf")
    create_plot(e2_data, "e2", "improvement", "padversaries", "e2_imp.pdf")
    print()

    c_data = sensitivity_data.query("`attack` == 'shifted_random' and `e1` == 0.01 and `e2` == 0.1")
    c_data = c_data.drop(
        columns=["attack", "seed", "repetitions", "npoints", "e1", "e2", "dimensions", "overlap_scaling_function"]
    )
    print("c data:")
    print_latex(c_data.corr())
    create_plot(c_data, "c", "error", "padversaries", "c_err.pdf")
    create_plot(c_data, "c", "improvement", "padversaries", "c_imp.pdf")
    print()

    osf_sr_data = sensitivity_data.query("`attack` == 'shifted_random' and `e1` == 0.01 and `e2` == 0.1 and `c` == 0.5")
    osf_sr_data = osf_sr_data.drop(
        columns=["attack", "seed", "repetitions", "npoints", "e1", "e2", "c", "dimensions"]
    )
    create_plot(osf_sr_data, "padversaries", "error", "overlap_scaling_function", "osf_sr_err.pdf")
    create_plot(osf_sr_data, "padversaries", "improvement", "overlap_scaling_function", "osf_sr_imp.pdf")
    print()

    osf_lie_data = sensitivity_data.query("`attack` == 'lie' and `e1` == 0.01 and `e2` == 0.1 and `c` == 0.5")
    osf_lie_data = osf_lie_data.drop(
        columns=["attack", "seed", "repetitions", "npoints", "e1", "e2", "c", "dimensions"]
    )
    create_plot(osf_lie_data, "padversaries", "error", "overlap_scaling_function", "osf_lie_err.pdf")
    create_plot(osf_lie_data, "padversaries", "improvement", "overlap_scaling_function", "osf_lie_imp.pdf")
    print()

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
