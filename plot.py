import argparse
import matplotlib.pyplot as plt
import json


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
        plt.savefig(f"{'train' if train else 'test'}_{key}.png")
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--save", action="store_true", help="Save the plots to png files.")
    parser.add_argument("--gradient-similarity", action="store_true", help="Consider the gradient similarity experiment results")
    parser.add_argument("--fairness", action="store_true", help="Consider the fairness experiment results")
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
        plot_boxplot(results, "loss", train=False, save=args.save)

        plot_boxplot(results, "accuracy std", train=False, save=args.save)
        plot_boxplot(results, "loss std", train=False, save=args.save)