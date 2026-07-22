# Training container for the wedge CNN pipeline.
#
# Purpose: train on the Windows dev box inside a Linux userspace so Keras
# writes portable .weights.h5 files (no backslash HDF5 paths), pinned to the
# same TF/Python family as the Ubuntu deployment server.
#
# The LIVE MONITOR is NOT containerized — the server runs it natively.
#
# Build:  docker compose build          (from the repo root)
# Train:  docker compose run train      (see docker-compose.yml)
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    tensorflow-cpu==2.17.1 \
    pandas \
    numpy \
    scikit-learn \
    pyarrow \
    matplotlib

WORKDIR /app/source_code
