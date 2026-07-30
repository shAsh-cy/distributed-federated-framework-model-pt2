# Docker setup

One image (`Dockerfile`) serves both roles; `docker-compose.yml` overrides the
command per service.

```bash
docker compose -f docker/docker-compose.yml up --build --scale client=5
```

Verified: 5 rounds to 77.13% test accuracy, with all five clients and the server
exiting 0.

## Two things that are easy to get wrong here

**The build context is the repository root, not this directory.** The compose
file sets `context: ..`. A previous version used `context: ./` with
`COPY ../requirements.txt`, which can never work: Compose resolves the context
relative to the compose file (so `docker/`), and BuildKit clamps `COPY` paths to
the context root, so every copy failed before `pip` was reached.

**`client` is a single scalable service.** Hardcoded `client0`/`client1` services
cannot be scaled — `--scale client=N` needs one service to replicate, and
per-replica environment variables cannot be set that way. Replicas need no
per-replica configuration because a client registering without an id is assigned
the next free shard by the server, so N replicas claim N distinct shards.

## Notes

- `configs/docker.yaml` declares `num_clients: 5`, matching `--scale client=5`.
  Starting more clients than that is refused (`all 5 shards already claimed`)
  rather than silently double-counting a shard.
- The image bakes in the Fashion-MNIST cache, so five replicas do not each
  download the dataset on startup.
- Dependencies are installed before the source is copied, so editing code does
  not invalidate the very large pip layer.
- The server writes per-round metrics to `/app/results/docker_run.json`.
