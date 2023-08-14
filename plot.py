from typing import Iterable
import argparse
import matplotlib.pyplot as plt
import json
import itertools
import numpy as np


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


# CHANGEME
def plot_attack_results(results, key, train=True):
    fig, axes = plt.subplots(1, len(results.keys()), figsize=(18, 6))
    axes_iter = iter(axes)
    y_min, y_max = 1.0, 0.0
    for strategy, strategy_results in results.items():
        ax = next(axes_iter)
        line_symbols = itertools.cycle(['-o', '-s', '-^', '-x', '-d', '-p', '-*'])
        for attack, attack_results in strategy_results.items():
            rounds = list(attack_results['train' if train else 'test'].keys())
            y_mean = np.array([np.mean(ar[key]) for ar in attack_results['train' if train else 'test'].values()])
            # y_std = np.array([np.std(ar[key]) for ar in attack_results['train' if train else 'test'].values()])
            y_min = min(y_mean.min(), y_min)
            y_max = max(y_mean.max(), y_max)
            ax.plot(rounds, y_mean, next(line_symbols), label=attack)
            # plt.errorbar(rounds, y_mean, yerr=y_std, label=attack)
        ax.set_xlabel("Round")
        ax.set_ylabel(key.title())
        ax.legend(title="Attack")
        ax.set_title(strategy)
    for ax in axes:
        ax.set_ylim((y_min - 0.01, y_max + 0.01))
    plt.tight_layout()
    # plt.show()
# CHANGEME end


def reorder_results(results):
    new_results = {}
    for strategy, strategy_results in results.items():
        for attack, attack_results in strategy_results.items():
            if not new_results.get(attack):
                new_results[attack] = {}
            new_results[attack][strategy] = {}
            for train_or_test in ["train", "test"]:
                new_results[attack][strategy][train_or_test] = {outer_k: {k: [] for k in ar.keys()} for outer_k, ar in attack_results[train_or_test][0].items()}
                for i in range(len(attack_results[train_or_test])):
                    for outer_k, vd in attack_results[train_or_test][i].items():
                        for k, v in vd.items():
                            new_results[attack][strategy][train_or_test][outer_k][k].append(v)
    return new_results


def plot_attack_results(results, key, train=False, save=True):
    y_min, y_max = 1.0, 0.0
    for attack, attack_results in results.items():
        line_symbols = itertools.cycle(['-o', '-s', '-^', '-x', '-d', '-p', '-*'])
        for strategy, strategy_results in attack_results.items():
            rounds = list(strategy_results['train' if train else 'test'].keys())
            y_mean = np.array([np.mean(ar[key]) for ar in strategy_results['train' if train else 'test'].values()])
            y_min = min(y_mean.min(), y_min)
            y_max = max(y_mean.max(), y_max)
            plt.plot(rounds, y_mean, next(line_symbols), label=strategy)
        plt.xlabel("Round")
        plt.ylabel(key.title())
        plt.legend()
        plt.ylim((y_min - 0.01, y_max + 0.01))
        plt.tight_layout()
        if save:
            plt.savefig(f"{attack}_{key}.png", dpi=320)
            plt.clf()
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--save", action="store_true", help="Save the plots to png files.")
    parser.add_argument("--gradient-similarity", action="store_true", help="Consider the gradient similarity experiment results")
    parser.add_argument("--fairness", action="store_true", help="Consider the fairness experiment results")
    parser.add_argument("--attack", action="store_true", help="Consider the attack experiment results")
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
        with open("results/attack.json", "r") as f:
            results = json.load(f)
        
        results = reorder_results(results)

        plot_attack_results(results, "accuracy", train=False, save=args.save)
        plot_attack_results(results, "loss", train=False, save=args.save)
        plot_attack_results(results, "asr", train=False, save=args.save)