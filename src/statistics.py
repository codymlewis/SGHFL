from typing import List
import argparse
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl

from logger import logger


def process_agg_name(label_name: str) -> str:
    match label_name:
        case "trimmed_mean":
            label = "TrMean"
        case "space_sample_mean":
            label = "SSMean"
        case "fedavg":
            label = "FedAVG"
        case "ssfgm":
            label = "SSFGM"
        case "topk_ssfgm":
            label = "Top $k$ SSFGM"
        case "geomedian":
            label = "GeoMedian"
        case "fedprox":
            label = "FedProx"
        case "topk":
            label = "Top $k$"
        case "kickback_momentum":
            label = "KBM"
        case "mrcs":
            label = "MRCS"
        case _:
            label = label_name.title()
    return label


def aggregator_key(agg_name: str) -> int:
    match agg_name:
        case "fedavg":
            key = 0
        case "median":
            key = 1
        case "ssfgm" | "topk_ssfgm" | "space_sample_mean" | "kickback_momentum" | "mrcs":
            key = int.from_bytes(agg_name.encode("utf-8"), "big") + 2**5000
        case _:
            key = int.from_bytes(agg_name.encode("utf-8"), "big")
    return key


def find_fairness_values(df: pl.DataFrame, datset: str) -> List[pl.Series]:
    value_col = "r2_score" if dataset == "l2rpn" else "mae"
    return [
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 1.1) &
            (pl.col("attack") == "none")
        )[value_col],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 0.4) &
            (pl.col("attack") == "none")
        )[value_col],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 0.4) &
            (pl.col("attack") == "none")
        )[f"dropped {value_col}"],
        df.filter(
            pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 0.4) &
            (pl.col("attack") == "none")
        )[f"dropped {value_col}"],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 1.1) &
            (pl.col("attack") == "none")
        )["cosine_similarity"],
    ]


def find_attack_values(df: pl.DataFrame, dataset: str) -> List[pl.Series]:
    value_col = "r2_score" if dataset == "l2rpn" else "mae"
    return [
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 1.1) &
            (pl.col("attack") == "none")
        )[value_col],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 1.1) &
            (pl.col("pct_saturation") == 0.5) &
            (pl.col("attack") == "empty")
        )[value_col],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 1.1) &
            (pl.col("pct_saturation") == 1.0) &
            (pl.col("attack") == "empty")
        )[value_col],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 1.1) &
            (pl.col("pct_saturation") == 0.5) &
            (pl.col("attack") == "ipm")
        )[value_col],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 1.1) &
            (pl.col("pct_saturation") == 1.0) &
            (pl.col("attack") == "ipm")
        )[value_col],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 1.1) &
            (pl.col("pct_saturation") == 0.5) &
            (pl.col("attack") == "lie")
        )[value_col],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 1.1) &
            (pl.col("pct_saturation") == 1.0) &
            (pl.col("attack") == "lie")
        )[value_col],
    ]


def find_fairness_attack_values(df: pl.DataFrame, dataset: str) -> List[pl.Series]:
    value_col = "r2_score" if dataset == "l2rpn" else "mae"
    return [
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 0.4) &
            (pl.col("attack") == "none")
        )[value_col],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 0.4) &
            (pl.col("attack") == "none")
        )[f"dropped {value_col}"],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 0.4) &
            (pl.col("pct_saturation") == 0.5) &
            (pl.col("attack") == "ipm")
        )[value_col],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 0.4) &
            (pl.col("pct_saturation") == 0.5) &
            (pl.col("attack") == "ipm")
        )[f"dropped {value_col}"],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 0.4) &
            (pl.col("pct_saturation") == 0.5) &
            (pl.col("attack") == "lie")
        )[value_col],
        df.filter(
            ~pl.col("server_aggregator").str.ends_with("IF") &
            (pl.col("drop_point") == 0.4) &
            (pl.col("pct_saturation") == 0.5) &
            (pl.col("attack") == "lie")
        )[f"dropped {value_col}"],
    ]


def create_plot(input_df: pl.DataFrame, dataset: str, plot_type: str = "fairness"):
    cmap_name = "Reds_r" if dataset == "l2rpn" else "Reds"
    value_col = "r2_score" if dataset == "l2rpn" else "mae"
    server_aggregators = sorted(
        [sa for sa in input_df["server_aggregator"].unique() if not sa.endswith("IF")],
        key=aggregator_key,
    )
    middle_server_aggregators = sorted(
        [msa for msa in input_df["middle_server_aggregator"].unique() if msa != "space_sample_mean"],
        key=aggregator_key,
    )
    fig, axes = plt.subplots(len(server_aggregators), len(middle_server_aggregators), sharex=True, sharey=True)
    fig.set_size_inches(18, 18)
    for msa, ax in zip(middle_server_aggregators, axes[-1, :]):
        ax.set_xlabel(process_agg_name(msa))
    for sa, ax in zip(server_aggregators, axes[:, 0]):
        ax.set_ylabel(process_agg_name(sa))
    axes = iter(axes.flatten())
    cmap = mpl.colormaps[cmap_name]
    for server_aggregator in server_aggregators:
        for middle_server_aggregator in middle_server_aggregators:
            q = (
                input_df.lazy()
                .filter(
                    pl.col("server_aggregator").str.starts_with(server_aggregator) &
                    (pl.col("middle_server_aggregator") == middle_server_aggregator)
                )
                .with_columns(pl.col(f"{value_col}").clip(lower_bound=-0.05, upper_bound=1.0))
                .with_columns(pl.col(f"dropped {value_col}").clip(lower_bound=-0.05, upper_bound=1.0))
            )
            df = q.collect()
            if plot_type == "fairness":
                values = find_fairness_values(df, dataset)
            elif plot_type == "attack":
                values = find_attack_values(df, dataset)
            else:
                values = find_fairness_attack_values(df, dataset)
            ax = next(axes)
            colours = [cmap(0.0) for _ in values]
            for i, v in enumerate(values):
                if v.shape[0] == 0:
                    values[i] = np.array([-0.05])
                else:
                    colours[i] = cmap(np.mean(v.to_numpy()))
            parts = ax.violinplot(values, showmeans=True)
            for pc, colour in zip(parts['bodies'], colours):
                pc.set_facecolor(colour)
                pc.set_edgecolor("black")
                pc.set_alpha(1)
            parts['cmeans'].set_colors("black")
            parts['cmins'].set_colors("black")
            parts['cmaxes'].set_colors("black")
            parts['cbars'].set_colors("black")
            ax.set_ylim([-0.1, 1.1])
            if plot_type == "fairness":
                value_name = "$r^2$" if dataset == "l2rpn" else "MAE"
                labels = [
                    f"Normal {value_name}",
                    f"Participating {value_name}",
                    f"Dropped {value_name}",
                    f"Dropped with IF {value_name}",
                    "Cosine Similarity"
                ]
            elif plot_type == "attack":
                labels = ["No Attack", "Empty", "Saturated Empty", "IPM", "Saturated IPM", "LIE", "Saturated LIE"]
            elif plot_type == "fairness_attack":
                labels = [
                    "Participating No Attack",
                    "Dropped No Attack",
                    "Participating IPM",
                    "Dropped IPM",
                    "Participating LIE",
                    "Dropped LIE",
                ]
            ax.set_xticks([i + 1 for i in range(len(values))], labels=labels, rotation='vertical')
            ax.tick_params(
                bottom=server_aggregator == server_aggregators[-1],
                left=middle_server_aggregator == middle_server_aggregators[0],
            )
    fig.text(0.5, 0.0, 'Data Collector Aggregator', ha='center')
    fig.text(0.07, 0.5, 'Distribution Server Aggregator', va='center', rotation='vertical')
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.97, 0.15, 0.03, 0.7])
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap_name),
        label='Mean',
        cax=cbar_ax,
    )
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    filename = f"plots/{dataset}_{plot_type}.pdf"
    plt.savefig(filename, bbox_inches="tight")
    logger.info(f"Saved plot to {filename}")


def create_smart_grid_plot(input_df: pl.DataFrame, dataset: str, plot_type: str, aggregator: str):
    cmap_name = "Reds_r" if dataset == "l2rpn" else "Reds"
    cmap = mpl.colormaps[cmap_name]
    server_aggregator = aggregator
    middle_server_aggregator = aggregator
    if aggregator.find(":") >= 0:
        server_aggregator, middle_server_aggregator = aggregator.split(":", maxsplit=2)
    elif aggregator == "duttagupta":
        middle_server_aggregator = "fedavg"
    elif aggregator == "li":
        server_aggregator = "fedavg"
    value_col = "r2_score" if dataset == "l2rpn" else "mae"

    q = (
        input_df.lazy()
        .filter(
            (pl.col("server_aggregator") == server_aggregator) &
            (pl.col("middle_server_aggregator") == middle_server_aggregator)
        )
        .with_columns(pl.col(f"{value_col}").clip(lower_bound=-0.05, upper_bound=1.0))
        .with_columns(pl.col(f"dropped {value_col}").clip(lower_bound=-0.05, upper_bound=1.0))
    )
    df = q.collect()
    if plot_type == "fairness":
        values = [
            df.filter((pl.col("drop_point") == 1.1) & (pl.col("attack") == "none"))[value_col],
            df.filter((pl.col("drop_point") == 0.4) & (pl.col("attack") == "none"))[value_col],
            df.filter((pl.col("drop_point") == 0.4) & (pl.col("attack") == "none"))[f"dropped {value_col}"],
            df.filter((pl.col("drop_point") == 1.1) & (pl.col("attack") == "none"))["cosine_similarity"],
        ]
    elif plot_type == "attack":
        df = df.filter(pl.col("drop_point") == 1.1)
        values = [
            df.filter((pl.col("attack") == "none"))[value_col],
            df.filter((pl.col("pct_saturation") == 0.5) & (pl.col("attack") == "empty"))[value_col],
            df.filter((pl.col("pct_saturation") == 0.5) & (pl.col("attack") == "ipm"))[value_col],
            df.filter((pl.col("pct_saturation") == 0.5) & (pl.col("attack") == "lie"))[value_col],
        ]
    else:
        values = find_fairness_attack_values(df, dataset)

    colours = [cmap(0.0) for _ in values]
    for i, v in enumerate(values):
        if v.shape[0] == 0:
            values[i] = np.array([-0.05])
        else:
            colours[i] = cmap(np.mean(v.to_numpy()))
    fig, ax = plt.subplots()
    parts = ax.violinplot(values, showmeans=True)
    for pc, colour in zip(parts['bodies'], colours):
        pc.set_facecolor(colour)
        pc.set_edgecolor("black")
        pc.set_alpha(1)
    parts['cmeans'].set_colors("black")
    parts['cmins'].set_colors("black")
    parts['cmaxes'].set_colors("black")
    parts['cbars'].set_colors("black")
    ax.set_ylim([-0.1, 1.1])
    if plot_type == "fairness":
        colname = "$r^2$" if dataset == "l2rpn" else "MAE"
        labels = [
            f"Normal {colname}",
            f"Participating {colname}",
            f"Dropped {colname}",
            "Cosine Similarity"
        ]
    elif plot_type == "attack":
        labels = ["No Attack", "Empty", "IPM", "LIE"]
    elif plot_type == "fairness_attack":
        labels = [
            "Participating No Attack",
            "Dropped No Attack",
            "Participating IPM",
            "Dropped IPM",
            "Participating LIE",
            "Dropped LIE",
        ]
    ax.set_xticks([i + 1 for i in range(len(values))], labels=labels, rotation='vertical')
    filename = f"plots/smart_grid_{dataset}_{plot_type}_{aggregator}.pdf"
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    plt.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap_name),
        label='Mean',
        cax=cbar_ax,
    )
    plt.savefig(filename, bbox_inches="tight")
    plt.clf()
    plt.close(fig)
    logger.info(f"Saved plot to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument("-f", "--file", type=str, default=None, help="Results file to read.")
    parser.add_argument("-s", "--smart-grid", action="store_true", help="Plot the smart grid data.")
    args = parser.parse_args()

    os.makedirs("plots", exist_ok=True)
    datasets = ["l2rpn", "apartment", "solar_home"] if args.smart_grid else ["l2rpn"]

    if args.file is None:
        results_filename = "results/smart_grid_results.csv" if args.smart_grid else "results/l2rpn_results.csv"
    else:
        results_filename = args.file

    for dataset in datasets:
        q = (
            pl.scan_csv(results_filename)
            .filter(pl.col("dataset") == dataset)
        )
        results_data = q.collect()
        if len(results_data) == 0:
            continue

        for plot_type in ["fairness", "attack", "fairness_attack"]:
            if args.smart_grid:
                for aggregator in ["fedavg", "duttagupta", "li", "phocas:ssfgm", "ssfgm", "phocas:lissfgm", "lissfgm"]:
                    create_smart_grid_plot(results_data, dataset, plot_type, aggregator)
            else:
                create_plot(q.collect(), dataset, plot_type)
