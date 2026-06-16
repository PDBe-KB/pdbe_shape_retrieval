FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAP2ZERNIKE_SETUP_DIR=/usr/local/bin \
    OBJ2GRID_PATH=/usr/local/bin/obj2grid \
    PATH="/usr/local/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip uv

COPY . /app

RUN install -m 0755 /app/bin/map2zernike /usr/local/bin/map2zernike \
    && install -m 0755 /app/bin/obj2grid /usr/local/bin/obj2grid \
    && uv pip install --system .

ENTRYPOINT ["shape_retrieval"]
