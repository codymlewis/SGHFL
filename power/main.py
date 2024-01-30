from functools import partial
import argparse
import json
import time
import os

import adversaries
import fl
import load_data


def get_experiment_config(all_exp_configs, exp_id):
    experiment_config = {k: v for k, v in all_exp_configs.items() if k != "experiments"}
    variables = all_exp_configs['experiments'][exp_id - 1]
    experiment_config.update(variables)
    return experiment_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments evaluating the solar home dataset.")
    parser.add_argument("-d", "--dataset", type=str, default="solar_home",
                        help="The dataset to train on.")
    parser.add_argument("-i", "--id", type=int, default=1,
                        help="Which of the experiments in the config to perform (counts from 1).")
    parser.add_argument("-p", "--performance", action="store_true",
                        help="Perform experiments evaluating the performance.")
    parser.add_argument("-a", "--attack", action="store_true",
                        help="Perform experiments evaluating the vulnerability to and mitigation of attacks.")
    parser.add_argument("-f", "--fairness", action="store_true", help="Perform experiments evaluating the fairness.")
    args = parser.parse_args()

    start_time = time.time()
    keyword = "performance" if args.performance else "attack" if args.attack else "fairness"
    with open(f"configs/{keyword}.json", 'r') as f:
        experiment_config = get_experiment_config(json.load(f), args.id)
    print(f"Performing {keyword} experiment with {experiment_config=}")
    experiment_config['experiment_type'] = keyword

    client_data, X_test, Y_test = load_data.load_data(args.dataset)
    if args.performance or args.fairness:
        regions = load_data.load_regions(args.dataset)
        network_arch = [
            fl.MiddleServer([fl.Client(client_data[r]) for r in region], experiment_config) for region in regions
        ]
    else:
        if experiment_config["attack"] == "empty":
            adversary_type = adversaries.EmptyUpdater
        else:
            corroborator = adversaries.Corroborator(len(client_data), round(len(client_data) * (1 - 0.5)))
            if experiment_config["attack"] == "lie":
                adversary_type = partial(adversaries.LIE, corroborator=corroborator)
            elif experiment_config["attack"] == "ipm":
                adversary_type = partial(adversaries.IPM, corroborator=corroborator)
            else:
                adversary_type = partial(adversaries.BackdoorLIE, corroborator=corroborator)
        network_arch = [
            adversary_type(d) if i + 1 > (len(client_data) * 0.5) else fl.Client(d)
            for i, d in enumerate(client_data)
        ]

    server = fl.Server(
        network_arch,
        experiment_config,
        Y_test.shape[1:] + X_test.shape[1:],
    )
    training_metrics = server.fit()

    if args.fairness:
        training_metrics, baseline_metrics = training_metrics

    testing_metrics = server.analytics()
    centralised_metrics = server.evaluate(X_test, Y_test)
    print(f"{training_metrics=}")
    print(f"{testing_metrics=}")
    print(f"{centralised_metrics=}")
    results = {"train": training_metrics, "test": testing_metrics, "centralised": centralised_metrics}

    if args.attack and experiment_config['attack'] == "backdoor_lie":
        backdoor_metrics = server.backdoor_analytics()
        results['backdoor'] = backdoor_metrics
        print(f"{backdoor_metrics=}")
    elif args.fairness:
        results['baseline'] = baseline_metrics
        print(f"{baseline_metrics=}")

    os.makedirs("results", exist_ok=True)
    filename = "results/{}_{}.json".format(
        args.dataset,
        '_'.join([f'{k}={v}' for k, v in experiment_config.items() if k not in ['repeat', 'round']]),
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")

    print(f"Experiment took {time.time() - start_time} seconds")
