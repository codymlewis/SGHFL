from functools import partial
import argparse
import json
import time
import math
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
    parser.add_argument("--pct-dc-adversaries", type=float, default=0.5,
                        help="Percentage of middle servers to assign as adversaries, if performing an attack evaluation")
    parser.add_argument("--pct-saturation", type=float, default=1.0,
                        help="The percentage of clients under adversary middle servers to assign as adversaries.")
    args = parser.parse_args()

    start_time = time.time()
    keyword = "performance" if args.performance else "attack" if args.attack else "fairness"
    with open(f"configs/{keyword}.json", 'r') as f:
        experiment_config = get_experiment_config(json.load(f), args.id)
    print(f"Performing {keyword} experiment with {experiment_config=}")
    experiment_config['experiment_type'] = keyword

    client_data, X_test, Y_test = load_data.load_data(args.dataset)
    regions = load_data.load_regions(args.dataset)
    if args.performance or args.fairness:
        network_arch = [
            fl.MiddleServer([fl.Client(client_data[r]) for r in region], experiment_config) for region in regions
        ]
    else:
        if experiment_config["attack"] == "empty":
            adversary_type = adversaries.EmptyUpdater
        else:
            corroborator = adversaries.Corroborator(len(client_data))
            if experiment_config["attack"] == "lie":
                adversary_type = partial(adversaries.LIE, corroborator=corroborator)
            elif experiment_config["attack"] == "ipm":
                adversary_type = partial(adversaries.IPM, corroborator=corroborator)
            else:
                adversary_type = partial(adversaries.BackdoorLIE, corroborator=corroborator)
        network_arch = [
            fl.MiddleServer(
                [
                    adversary_type(client_data[r])
                    if (dc + 1 > math.ceil(len(regions) * (1 - args.pct_dc_adversaries))) and (c + 1 > math.ceil(len(region) * (1 - args.pct_saturation)))
                    else fl.Client(client_data[r])
                    for c, r in enumerate(region)
                ],
                experiment_config
            ) for dc, region in enumerate(regions)
        ]

    server = fl.Server(
        network_arch,
        experiment_config,
        Y_test.shape[1:] + X_test.shape[1:],
    )
    training_metrics = server.fit()

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

    os.makedirs("results", exist_ok=True)
    filename = "results/{}_{}{}.json".format(
        args.dataset,
        '_'.join([f'{k}={v}' for k, v in experiment_config.items() if k not in ['repeat', 'round']]),
        f"_percent_dc_adversaries={args.pct_dc_adversaries}_saturation={args.pct_saturation}" if args.attack else ""
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")

    print(f"Experiment took {time.time() - start_time} seconds")
