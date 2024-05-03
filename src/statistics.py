import numpy as np
import matplotlib.pyplot as plt
import polars as pl


def create_plot(input_df: pl.DataFrame):
    server_aggregators = [sa for sa in input_df["server_aggregator"].unique() if not sa.endswith("IF")]
    print(server_aggregators)
    middle_server_aggregators = input_df["middle_server_aggregator"].unique()
    fig, axes = plt.subplots(len(server_aggregators), len(middle_server_aggregators))
    axes = iter(axes.flatten())
    for server_aggregator in server_aggregators:
        for middle_server_aggregator in middle_server_aggregators:
            q = (
                input_df.lazy()
                .filter(
                    pl.col("server_aggregator").str.starts_with(server_aggregator) &
                    (pl.col("middle_server_aggregator") == middle_server_aggregator) &
                    (pl.col("drop_point") >= 1.1) &
                    (pl.col("attack") == "none")
                )
                .with_columns(pl.col("r2_score").clip(lower_bound=-0.05))
                .with_columns(pl.col("dropped r2_score").clip(lower_bound=-0.05))
            )
            df = q.collect()
            values = [
                df.filter(
                    ~pl.col("server_aggregator").str.ends_with("if") & (pl.col("attack") == "none")
                )["r2_score"],
                df.filter(
                    ~pl.col("server_aggregator").str.ends_with("if") & (pl.col("attack") == "none")
                )["cosine_similarity"],
                # df.filter(pl.col("server_aggregator").str.ends_with("if"))["r2_score"],
                # df.filter(pl.col("server_aggregator").str.ends_with("if"))["cosine_similarity"],
            ]
            ax = next(axes)
            ax.set_title(f"{server_aggregator}, {middle_server_aggregator}")
            if np.any([v.shape[0] == 0 for v in values]):
                print(f"{server_aggregator=}, {middle_server_aggregator=} failed")
                continue
            ax.violinplot(values)
            ax.set_ylim([-0.1, 1.1])
            ax.set_xticks(
                [i + 1 for i in range(len(values))],
                labels=["$r^2$", "cs"]
            )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    q = (
        pl.scan_csv("results/results.csv")
        .group_by([
            "dataset",
            "pct_adversaries",
            "pct_saturation",
            "server_aggregator",
            "middle_server_aggregator",
            "attack",
            "drop_point",
        ])
        .mean()
    )
    create_plot(q.collect())
