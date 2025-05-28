# A Robust and Fair Hierarchical Federated Learning Framework for Smart Grid

Source code corresponding to ...

## Description

With the recent rapid expansion of the smart grid infrastructure paving the way for greater integration of computer and network technologies within the power grid, it has become well-suited for the application of machine learning techniques. However, machine learning requires vast amounts of data, which within the smart grid setting can reveal great amounts of personal details of the individuals using the grid. This work considers the application of a variant of distributed machine learning, federated learning, which enhances data privacy. We propose a Smart Grid Hierarchical Federated Learning (SGHFL) framework, which is tuned to common smart grid architectures in the real-world. We demonstrate how our SGHFL framework improves client dropout and poisoning robustness, using relatively lightweight models suitable for devices with limited computational capability. We provide theoretical justification underlying our design, and have evaluated our algorithms and framework with three datasets/environments of progressively increasing practicality.  We have also compared our framework with relevant works.

## Requirements

- [uv](https://docs.astral.sh/uv/) or [docker](https://www.docker.com)


## Setup

### Local Setup

Initialise the virtual environment:

```bash
uv venv
```

Then choose one of the following,

1. To use CUDA-based acceleration:

```bash
uv sync --extra cuda
```

2. To use the CPU only:

```bash
uv sync --extra cpu
```

and lastly activate the virtual environment whenever you want to run the code,

```bash
source .venv/bin/activate
```

### Docker Setup

Build the docker image (replace <GPU> with either cpu or cuda):

```bash
docker buildx build -t local --build-arg GPU_TYPE=<GPU> .
```

Start the docker image in interactive mode (`--gpus=all` can be removed if you are running a cpu only):

```bash
docker run --rm -it --gpus=all local bash
```


## Run the experiments

Firstly, download the data by running,

```bash
./download_data.sh
```

then change directory to the `src` folder and execute the 3 scripts in the `scripts` folder,

```bash
cd src/ && \
./run_experiments && \
./run_model_experiments && \
./run_smart_grid_experiments
```

The results will be placed in csv files in the `results` folder, and plots/summarising tables can be generated from them by running,

```bash
python statistics.py
```

## Citation

To be added upon publication.
