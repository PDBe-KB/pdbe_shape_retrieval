# Inspired from https://github.com/astral-sh/uv-docker-example/blob/5748835918ec293d547bbe0e42df34e140aca1eb/multistage.Dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Omit development dependencies
ENV UV_NO_DEV=1

ENV UV_PYTHON_DOWNLOADS=0

ARG PIP_INDEX_URL
ENV UV_DEFAULT_INDEX=$PIP_INDEX_URL
ENV UV_EXTRA_INDEX_URL=https://pypi.org/simple

WORKDIR /app
COPY pyproject.toml uv.lock /app/
RUN uv sync --locked --no-install-project --no-dev

COPY shape_retrieval /app/shape_retrieval
RUN uv tool install shape_retrieval

FROM python:3.12-slim-bookworm
LABEL maintainer="Sreenath Sasidharan Nair <sreenath@ebi.ac.uk>"

RUN groupadd --system --gid 101 nonroot \
    && useradd --system --gid 101 --uid 101 --create-home nonroot

COPY --from=builder --chown=nonroot:nonroot /app /app

ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Disable core dumps
RUN echo -e "* soft core 0\n* hard core 0" >> /etc/security/limits.conf

USER nonroot

# Set the working directory to /app
WORKDIR /app

CMD ["shape_retrieval"]
