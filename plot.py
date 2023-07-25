import matplotlib.pyplot as plt
import json


def plot_boxplot(results, key, train=True):
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
    plt.savefig(f"{'train' if train else 'test'}_{key}.png")
    plt.clf()


if __name__ == "__main__":
    with open(f"results/3_rounds_5_episodes_gradient_similarity.json", "r") as f:
        results = json.load(f)

    plot_boxplot(results, "cosinesimilarity")
    plot_boxplot(results, "accuracy")
    plot_boxplot(results, "loss")

    plot_boxplot(results, "accuracy", train=False)
    plot_boxplot(results, "loss", train=False)