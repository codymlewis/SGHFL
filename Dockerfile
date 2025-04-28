FROM ghcr.io/nvidia/jax:jax
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock download_data.sh /app/
COPY data_processing /app/data_processing/
COPY src/ /app/src/

WORKDIR /app
RUN uv venv && \
    uv sync && \
    source .venv/bin/activate && \
    bash ./download_data.sh

WORKDIR /app/src
