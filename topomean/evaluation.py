import itertools
import matplotlib.pyplot as plt
import pandas as pd


def create_plot(
    data: pd.DataFrame, param_key: str, dependent_key: str, legend_key: str
) -> None:
    markers = itertools.cycle(['o', 's', '*', 'x', '^', '+', 'v'])
    for legend in pd.unique(data[legend_key]):
        legend_data = data.query(f"`{legend_key}` == {legend}")
        plt.plot(legend_data[param_key], legend_data[f"{dependent_key} mean"], label=legend, marker=next(markers))
        plt.fill_between(
            legend_data[param_key],
            legend_data[f"{dependent_key} mean"] - legend_data[f"{dependent_key} std"],
            legend_data[f"{dependent_key} mean"] + legend_data[f"{dependent_key} std"],
            alpha=0.2,
        )
    plt.legend(title=legend_key)
    plt.xlabel(param_key)
    plt.ylabel(dependent_key)
    plt.show()


if __name__ == "__main__":
    sensitivity_data = pd.read_csv("results/sensitivity.csv")

    e1_data = sensitivity_data.query("`attack` == 'lie'")
    e1_data = e1_data.drop(columns=["attack", "seed", "repetitions", "npoints", "dimensions", "e2", "K"])
    print(e1_data)
    create_plot(e1_data, "e1", "error", "padversaries")

    e2_data = sensitivity_data.query("`attack` == 'shifted_random' and `K` == 3")
    e2_data = e2_data.drop(columns=["attack", "seed", "repetitions", "npoints", "e1", "K"])
    e2_padv_data = e2_data.query("`dimensions` == 2")
    e2_padv_data = e2_padv_data.drop(columns=["dimensions"])
    print(e2_padv_data)
    create_plot(e2_padv_data, "e2", "error", "padversaries")
    e2_dim_data = e2_data.query("`padversaries` == 0.4")
    e2_dim_data = e2_dim_data.drop(columns=["padversaries"])
    print(e2_dim_data)
    create_plot(e2_dim_data, "e2", "error", "dimensions")

    K_data = sensitivity_data.query("`attack` == 'shifted_random' and `e1` == 0.01 and `e2` == 1.0")
    K_data = K_data.drop(columns=["attack", "seed", "repetitions", "e1", "e2"])
    K_dim_data = K_data.query("`npoints` == 1000 and `padversaries` == 0.4")
    K_dim_data = K_dim_data.drop(columns=["npoints", "padversaries"])
    print(K_dim_data)
    create_plot(K_dim_data, "K", "error", "dimensions")
    K_npoints_data = K_data.query("`dimensions` == 2 and `padversaries` == 0.4")
    K_npoints_data = K_npoints_data.drop(columns=["dimensions", "padversaries"])
    print(K_npoints_data)
    create_plot(K_npoints_data, "K", "error", "npoints")
    K_padv_data = K_data.query("`npoints` == 1000 and `dimensions` == 2")
    K_padv_data = K_padv_data.drop(columns=["npoints", "dimensions"])
    print(K_padv_data)
    create_plot(K_padv_data, "K", "error", "padversaries")

    ablation_data = pd.read_csv("results/ablation.csv")
    ablation_data = ablation_data.drop(columns=['seed', 'repetitions'])
    print(ablation_data.style.to_latex())


    # TODO: Comparison data
