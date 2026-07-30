# Linux container matching the Ubuntu deployment server, used for two things on
# the Windows dev box:
#
#   1. TRAINING — so Keras writes portable .weights.h5 files (no backslash HDF5
#      paths), pinned to the same TF/Python family as the server.
#   2. VALIDATING MONITOR CHANGES — the committed weights are repaired to
#      forward-slash HDF5 keys for Linux and therefore will NOT load under
#      Windows Keras. Replaying or scoring anything with the deployed weights
#      has to happen in here.
#
# The live monitor still RUNS natively on the server; the container is only for
# training and for dev-box verification.
#
# Build:    docker compose build        (from the repo root)
# Train:    docker compose run train    (see docker-compose.yml)
# Validate: docker compose run monitor
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    tensorflow-cpu==2.17.1 \
    pandas \
    numpy \
    scikit-learn \
    pyarrow \
    matplotlib

# Runtime deps of live_monitor.py, kept in a separate layer so adding them does
# not invalidate the large TensorFlow layer above.
#   pandas_market_calendars: NYSE holiday / half-day calendar. Without it the
#     monitor's regular-session gate silently degrades to a weekday +
#     9:30-16:00 assumption, so a container missing it cannot faithfully
#     reproduce server behaviour.
#   yfinance: only needed for live polling, not for --replay, but installed so
#     the module imports cleanly and live mode can be smoke-tested.
RUN pip install --no-cache-dir \
    pandas_market_calendars \
    yfinance

WORKDIR /app/source_code
