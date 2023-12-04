import argparse
import matplotlib.pyplot as plt
import json
import itertools
import os
import re
import numpy as np


def make_leaves_lists(data):
    improved_data = {}
    for k, v in data.items():
        improved_data[k] = {}
        if isinstance(v, dict):
            improved_data[k] = make_leaves_lists(v)
        else:
            improved_data[k] = {vk: [] for vk in v[0].keys()}
            for vitem in v:
                for ki, vi in vitem.items():
                    improved_data[k][ki].append(vi)
    return improved_data


def plot_boxplot(results, key, train=True, save=True):
    plt.boxplot(
        [[r[key] for r in subresults['train' if train else 'test']] for subresults in results.values()],
        labels=list(results.keys())
    )
    if key == "cosinesimilarity":
        plt.ylabel("Cosine Similarity")
    else:
        plt.ylabel(key.title())
    plt.title(f"{'Training' if train else 'Testing'} results")
    plt.tight_layout()
    if save:
        plt.savefig(f"{'train' if train else 'test'}_{key}.png", dpi=320)
        plt.clf()
    else:
        plt.show()


def plot_attack_results(results, key, train=False, save=True):
    line_symbols = itertools.cycle(['-o', '-s', '-^', '-x', '-d', '-p', '-*'])
    for agg, agg_results in results.items():
        agg_results = agg_results['train' if train else 'test']
        plot_data = np.array([np.mean([d[key] for d in vd]) for vd in agg_results.values()])
        plt.plot(agg_results.keys(), plot_data, next(line_symbols), label=agg)
    plt.xlabel("Round")

    if key == "asr":
        ylabel = "ASR"
    elif "loss" in key:
        ylabel = "Loss"
    else:
        ylabel = key.title()

    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"{'train' if train else 'test'}_{key}.png", dpi=320)
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument('-s', "--save", action="store_true", help="Save the plots to png files.")
    parser.add_argument('-g', "--gradient-similarity", action="store_true",
                        help="Consider the gradient similarity experiment results")
    parser.add_argument('-f', "--fairness", action="store_true", help="Consider the fairness experiment results")
    parser.add_argument('-a', "--attack", action="store_true", help="Consider the attack experiment results")
    args = parser.parse_args()

    if args.gradient_similarity:
        with open("results/3_rounds_5_episodes_gradient_similarity.json", "r") as f:
            results = json.load(f)

        plot_boxplot(results, "cosinesimilarity", save=args.save)
        plot_boxplot(results, "accuracy", save=args.save)
        plot_boxplot(results, "loss", save=args.save)

        plot_boxplot(results, "accuracy", train=False, save=args.save)
        plot_boxplot(results, "loss", train=False, save=args.save)

    if args.fairness:
        with open("results/fairness_r5_e1_s1_dr2.json", "r") as f:
            drop_results = json.load(f)

        with open("results/fairness_r5_e1_s1_dr6.json", "r") as f:
            no_drop_results = json.load(f)

        results = {"Drop": drop_results, "Without Drop": no_drop_results}

        plot_boxplot(results, "accuracy", train=False, save=args.save)
        plot_boxplot(results, "crossentropy_loss", train=False, save=args.save)

        plot_boxplot(results, "accuracy std", train=False, save=args.save)
        plot_boxplot(results, "crossentropy_loss std", train=False, save=args.save)

    if args.attack:
        json_files = [f for f in os.listdir('results') if 'attack' in f and 'eval_every=1' in f]

        results = {}
        for json_file in json_files:
            with open(f"results/{json_file}", "r") as f:
                data = make_leaves_lists(json.load(f))
            env_name = json_file[re.search('aggregator=', json_file).end():re.search(r'aggregator=[A-Za-z]+_', json_file).end() - 1]
            results[env_name.title()] = data

        plot_attack_results(results, "accuracy", train=False, save=args.save)
        plot_attack_results(results, "crossentropy_loss", train=False, save=args.save)
        plot_attack_results(results, "asr", train=False, save=args.save)
