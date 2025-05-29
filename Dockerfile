FROM ghcr.io/astral-sh/uv:debian-slim

COPY pyproject.toml uv.lock .python-version download_data.sh /app/
COPY data_processing /app/data_processing/
COPY src/ /app/src/
RUN mkdir /app/src/results

WORKDIR /app
RUN uv venv && \
    uv sync && \
    source .venv/bin/activate && \
    bash ./download_data.sh

WORKDIR /app/src
ARG GPU_TYPE=cpu
RUN uv sync --extra $GPU_TYPE --compile-bytecode
