# One image serves both roles; docker-compose overrides the command.
# Build context is the REPOSITORY ROOT, not this directory -- see the `context: ..`
# in docker-compose.yml. The previous Dockerfiles used `COPY ../requirements.txt`
# against a context of `docker/`, which can never work: BuildKit clamps paths to
# the context root, so every COPY failed before pip was ever reached.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first so code edits do not invalidate the (very large) pip layer.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY fl/ /app/fl/
COPY configs/ /app/configs/

# Generate the gRPC stubs at build time so the first request does not pay for it
# and so a missing grpcio-tools fails the build rather than the run.
RUN python -m fl.proto._generate

# Warm the Fashion-MNIST cache into the image. Without this, every client
# container downloads the dataset on startup and `--scale client=5` produces
# five simultaneous downloads.
RUN python -c "import tensorflow as tf; tf.keras.datasets.fashion_mnist.load_data()"

CMD ["python", "-m", "fl.server", "--config", "configs/docker.yaml"]
