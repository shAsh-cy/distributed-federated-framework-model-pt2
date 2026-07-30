# Federated Learning Starter (Flower + PyTorch)

**Description:** Minimal, well-documented starter repo to experiment with Federated Learning using Flower (flwr) and PyTorch. This repo trains a simple CNN on MNIST in a federated manner with multiple Dockerized clients and a central server.

## Contents
- `server.py` - Flower server that coordinates rounds.
- `client.py` - Flower client that trains on local data (MNIST partition).
- `model.py` - PyTorch model definition and utilities.
- `utils.py` - Data loading and helper functions.
- `docker/` - Dockerfiles and docker-compose to run server + multiple clients locally.
- `requirements.txt` - Python dependencies.
- `run_local.sh` - Script to run server and multiple clients locally without Docker.
- `README.md` - This file.

## Quick start (local, no Docker)
1. Create a virtualenv and install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start server (in one terminal):
   ```bash
   python server.py
   ```
3. Start N clients (in separate terminals), e.g., 2 clients:
   ```bash
   python client.py --cid 0
   python client.py --cid 1
   ```

## Run with Docker (recommended for simulating many clients)
See `docker/README.md` for instructions to run server + multiple clients using docker-compose.

## Git & GitHub
Suggested initial commits & messages are in this repository's top-level `GIT_SETUP.md`.

## References
- Flower: https://flower.dev
- PyTorch: https://pytorch.org
