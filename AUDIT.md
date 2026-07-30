# Repository Audit

**Audited:** 2026-07-30
**Root:** `c:\Users\HP\Downloads\federated-learning-starter\federated-learning-starter`
**Method:** full file read of all 13 files; live `docker compose build` runs; live dependency resolution against PyPI on Python 3.10 / linux-amd64; live inspection of `flwr==1.7.0` library internals.

---

## Correction to the audit brief, up front

Three premises in the request do not hold against this working tree. Stating them here so the rest of the report reads correctly:

| Premise in brief | Actual state | Evidence |
|---|---|---|
| `__pycache__` is tracked | **not present** — and nothing is tracked, see below | `find . -type f` returns 13 files, no `__pycache__` |
| A file named `=1.3.0` exists | **not present** | `find . -name "*=*"` returns nothing |
| `simulation.py` exists / TFF is used | **not present** — no TensorFlow of any kind in this repo | grep for `tensorflow\|tff\|simulation` across the tree: **no matches** |

Most importantly: **this directory is not a git repository.** There is no `.git`. `git rev-parse --is-inside-work-tree` → `fatal: not a git repository (or any of the parent directories): .git`. The parent directory `c:\Users\HP\Downloads\federated-learning-starter\` contains only this folder and also has no `.git`.

Consequence: every question phrased as "is X *committed* / *tracked*" is unanswerable as asked, because there is no index and no history. There are no commits, no tracked files, and no `git rm --cached` to perform. The `.gitignore` at [.gitignore](.gitignore) is currently inert. [GIT_SETUP.md](GIT_SETUP.md) is a document *proposing* a commit sequence that has never been executed.

---

## 9. Version matrix: tensorflow / tensorflow-federated / tensorflow-privacy on Python 3.10

**Reported first per instruction. Result: it resolves — but only with a non-default package index. From PyPI alone it is unsatisfiable at every TFF version.**

### 9a. Applicability

**Not present in this repo.** No file imports or declares `tensorflow`, `tensorflow-federated`, or `tensorflow-privacy`. [requirements.txt](requirements.txt) lines 1–6 are `flwr`, `torch`, `torchvision`, `numpy`, `pillow`, `tqdm`. This is a **Flower + PyTorch** project. The item is answered below as a standalone dependency question, but nothing in this tree consumes the answer.

### 9b. The blocking conflict

`tensorflow-federated==0.87.0` (latest) declares an exact pin `jaxlib==0.4.14`.

**jaxlib 0.4.14 has been removed from PyPI.** The oldest jaxlib on PyPI today is `0.4.18`; the full available set begins `['0.4.18', '0.4.19', '0.10.0', '0.10.1', '0.10.2', '0.11.0', ...]` (41 versions). Because the pin is `==`, no resolver can satisfy it from PyPI:

```
× No solution found when resolving dependencies:
╰─▶ Because there is no version of jaxlib==0.4.14 and
    tensorflow-federated>=0.87.0 depends on jaxlib==0.4.14, we can conclude
    that tensorflow-federated>=0.87.0 cannot be used.
    And because you require tensorflow-federated==0.87.0, we can conclude
    that your requirements are unsatisfiable.
```

This is not specific to the newest release. Every TFF version carrying a jaxlib pin is affected — verified across the release history:

| TFF | requires_python | tensorflow | jaxlib pin | tf-privacy | jaxlib on PyPI? |
|---|---|---|---|---|---|
| 0.87.0 | `>=3.9,<3.12` | `==2.14.*` | `==0.4.14` | `==0.9.0` | **no** |
| 0.73.0 | `>=3.9,<3.12` | `==2.14.*` | `==0.4.14` | `==0.9.0` | **no** |
| 0.71.0 – 0.64.0 | `>=3.9,<3.12` | `==2.14.*` | `==0.4.14` | `==0.8.12` | **no** |
| 0.60.0 / 0.55.0 | `>=3.9.0,<3.11` | `~=2.12.0` | `==0.3.15` | `==0.8.10` / `0.8.8` | **no** |
| 0.50.0 | `>=3.9.0,<3.11` | `~=2.11.0` | `==0.3.14` | `==0.8.6` | **no** |

**The silent-failure trap:** an unpinned `pip install tensorflow-federated` does *not* error. The resolver backtracks past every real release and lands on `tensorflow-federated==0.1.0` — a 2019 placeholder that depends only on `tensorflow`, pulling `tensorflow==2.21.0`, `numpy==2.2.6`, `keras==3.12.4`. You get a package named `tensorflow_federated` that contains none of the federated API. Confirmed by live resolution. **Never install TFF unpinned.**

### 9c. The matrix that does resolve

Add the historical jax index, which still serves jaxlib 0.4.14 (confirmed present: `jaxlib-0.4.14-cp310-cp310-manylinux2014_x86_64.whl`, plus cp39/cp311 and win_amd64/macOS builds):

```bash
pip install "tensorflow-federated==0.87.0" \
  -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Verified end-to-end on `python:3.10-slim`, `linux/amd64`. Resolved set:

```
tensorflow==2.14.1
tensorflow-federated==0.87.0
tensorflow-privacy==0.9.0
tensorflow-probability==0.22.1
tensorflow-estimator==2.14.0
tensorflow-model-optimization==0.7.5
jax==0.4.14
jaxlib==0.4.14
keras==2.14.0
numpy==1.25.2
protobuf==4.25.9
ml-dtypes==0.2.0
scipy==1.9.3
absl-py==1.4.0
typing-extensions==4.5.0
dm-tree==0.1.8
dp-accounting==0.4.3
google-vizier==0.1.11
grpcio==1.74.0
attrs==23.1.0
cachetools==5.5.2
portpicker==1.6.0
```

You do not pin `tensorflow` or `tensorflow-privacy` yourself — TFF's own pins (`tensorflow==2.14.*,>=2.14.0`, `tensorflow-privacy==0.9.0`) determine them. Adding your own pins only creates new conflict surface.

### 9d. Platform constraint — relevant to this machine

**TFF publishes Linux-only wheels.** Every release ships exactly one artifact, e.g. `tensorflow_federated-0.87.0-py3-none-manylinux_2_31_x86_64.whl`. There is no Windows wheel, no macOS wheel, no aarch64 wheel, at any version. This audit ran on Windows 11. **TFF cannot be installed on this host directly** — it requires WSL2, Docker, or a Linux VM. `manylinux_2_31` needs glibc ≥ 2.31; `python:3.10-slim` (Debian 12, glibc 2.36) satisfies it.

Python bound: `>=3.9,<3.12`. Python 3.10 is valid. This host's interpreters are 3.14 (default), 3.13, 3.11 — **none of which can install TFF even on Linux.**

---

## 1. Source file inventory

13 files, 250 lines total (`wc -l`). No subpackages, no `__init__.py`, no test directory.

| File | Lines | Purpose |
|---|---|---|
| [client.py](client.py) | 66 | Flower `NumPyClient`: local SGD training loop, param get/set, local evaluation, CLI entrypoint. |
| [model.py](model.py) | 19 | `CNNMnist` — 2-conv + 2-FC PyTorch CNN. |
| [utils.py](utils.py) | 19 | `get_dataloader()` — downloads MNIST, partitions by contiguous index range per client. |
| [server.py](server.py) | 10 | Instantiates stock `FedAvg` and starts the Flower server. Contains no logic of its own. |
| [requirements.txt](requirements.txt) | 6 | flwr 1.7.0, torch 2.0.1, torchvision 0.15.2, numpy, pillow, tqdm. |
| [run_local.sh](run_local.sh) | 9 | Backgrounds server + 2 clients via bash. |
| [.gitignore](.gitignore) | 7 | venv, `__pycache__`, `*.pyc`, `.DS_Store`, `*.pth`, `*.pt`, `*.zip`. |
| [README.md](README.md) | 40 | Project overview and quickstart. |
| [GIT_SETUP.md](GIT_SETUP.md) | 29 | Proposed (never-executed) commit sequence. |
| [docker/docker-compose.yml](docker/docker-compose.yml) | 20 | Services `server`, `client0`, `client1`. |
| [docker/Dockerfile.client](docker/Dockerfile.client) | 8 | python:3.10-slim + deps + client sources. |
| [docker/Dockerfile.server](docker/Dockerfile.server) | 6 | python:3.10-slim + deps + server.py. |
| [docker/README.md](docker/README.md) | 13 | Docker instructions. |

Python source only: **114 lines** across 4 files.

**Not present:** `simulation.py`, any test file, any `.proto`, any generated `_pb2.py`/`_pb2_grpc.py`, `LICENSE`, `setup.py`/`pyproject.toml`, CI config, `.dockerignore`.

---

## 2. What `server.py` does on receiving a client update

### The literal contents

[server.py](server.py) is 10 lines and **contains zero aggregation arithmetic**:

```python
strategy = fl.server.strategy.FedAvg(min_fit_clients=2, min_available_clients=2)   # line 6
fl.server.start_server(server_address="0.0.0.0:8080",
                       config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)  # line 7
```

It constructs a stock `flwr.server.strategy.FedAvg` and hands control to the library. There is no `aggregate_fit` override, no custom strategy, no callback. **Searching this repository for aggregation arithmetic yields nothing** — the answer lies entirely in the installed `flwr` package.

### The actual arithmetic (in `flwr==1.7.0`, not in this repo)

`FedAvg.aggregate_fit` (`flwr/server/strategy/fedavg.py:218–248`) builds `(ndarrays, num_examples)` pairs and calls `aggregate()` (`flwr/server/strategy/aggregate.py:28–42`):

```python
def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    num_examples_total = sum(num_examples for (_, num_examples) in results)
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime
```

That is, per layer `L`:  **w_L ← ( Σ_k n_k · w_L^(k) ) / ( Σ_k n_k )**

`n_k` is whatever the client returns as its second `fit` return value. [client.py:38](client.py#L38) returns `len(self.trainloader.dataset)`.

### Direct answer to the three options offered

It is a **genuine weighted average by client sample count** — textbook FedAvg. It is *not* an unweighted mean and *not* a counter-triggered placeholder.

**Two qualifications that matter:**

1. **None of it is this project's code.** The weighting is a property of the Flower library, inherited by calling a stock constructor. The repository contributes no aggregation logic whatsoever.
2. **The weighting is numerically inert here.** [utils.py:14–17](utils.py#L14-L17) gives every client an identical contiguous chunk (`chunk_size = n // num_clients`), so all `n_k` are equal (the last client absorbs the remainder — at most `num_clients - 1` extra samples out of 60,000). With equal `n_k`, the weighted average collapses to the arithmetic mean. The sample-count weighting is real but exercises no non-uniform path.

Evaluation results take a separate path: `aggregate_evaluate` (`fedavg.py:254–281`) does `weighted_loss_avg` over losses. Because [server.py:6](server.py#L6) passes no `evaluate_metrics_aggregation_fn`, the `{"accuracy": ...}` dict returned by [client.py:53](client.py#L53) is **discarded**, and the library logs `WARNING: No evaluate_metrics_aggregation_fn provided` (`fedavg.py:280–281`). **Client accuracy is computed and then thrown away — it never reaches the server's output.**

---

## 3. What `client.py` trains on

**Dataset — MNIST.** [utils.py:9](utils.py#L9):
```python
dataset = datasets.MNIST('./data', train=train, download=True, transform=transform)
```
Downloaded at runtime to `./data`. Transform is `ToTensor()` only ([utils.py:8](utils.py#L8)) — no normalization, so inputs are `[0,1]` rather than the usual mean/std standardization.

**Partitioning.** [utils.py:11–17](utils.py#L11-L17): contiguous index-range split, `start = cid * chunk_size`, last client takes the remainder. The comment at [utils.py:10](utils.py#L10) calls this "Simple non-iid partitioning". **The comment is wrong in an important way.** `torchvision`'s MNIST is not sorted by label, so contiguous index ranges are approximately class-balanced — i.e. roughly **IID**, the opposite of the label skew "non-IID" implies in FL. Deterministic and disjoint, yes; non-IID, no.

**Architecture — `CNNMnist`** ([model.py:5–19](model.py#L5-L19)): `Conv2d(1,10,k=5)` → ReLU/maxpool2 → `Conv2d(10,20,k=5)` → ReLU/maxpool2 → flatten to 320 → `Linear(320,50)` → ReLU → `Linear(50,10)`. The classic PyTorch MNIST example net. Loss `CrossEntropyLoss`, optimizer plain `SGD(lr=0.01)` with no momentum ([client.py:15–16](client.py#L15-L16)). One local epoch per round ([client.py:31](client.py#L31), `for epoch in range(1)`), 3 rounds ([server.py:7](server.py#L7)).

**Held-out test set evaluation — yes, it exists, with caveats.**

- [client.py:18](client.py#L18) builds `self.testloader` with `train=False`, which loads the **official MNIST test split** — genuinely disjoint from the training split. Not a leak.
- [client.py:40–53](client.py#L40-L53) `evaluate()` computes loss and accuracy over that loader.
- **Caveat 1:** the test split is partitioned per client by the same function, so each client evaluates on its own ~5,000-sample shard, not the full 10,000-sample test set.
- **Caveat 2:** the resulting accuracy is discarded server-side (see §2) — no aggregated accuracy is ever reported.
- **Caveat 3:** there is no centralized/server-side evaluation anywhere. `FedAvg` is constructed with no `evaluate_fn` ([server.py:6](server.py#L6)), so the global model is never scored as a whole.
- **Caveat 4:** none of this runs, because of the defect in §5.

**Net:** held-out evaluation is *coded*; a reported held-out accuracy number is *not produced by any path in this repo*.

---

## 4. `simulation.py`

**Not present.** No such file exists anywhere in the tree (`find . -type f` — 13 files, listed in §1).

Consequently, the sub-questions are moot but answered explicitly for the record:

- **Is TensorFlow Federated genuinely used or only imported?** Neither. A case-insensitive grep for `tensorflow|tff|tensorflow_federated|tensorflow_privacy` across every file returns **no matches**. TFF is absent from both source and [requirements.txt](requirements.txt). There is no TensorFlow code in this project at any level — not real, not stubbed, not imported.
- **Is it connected to `server.py`/`client.py`, or a standalone parallel implementation?** Not applicable — there is no second implementation. The repo has exactly one FL path: Flower + PyTorch, via [server.py](server.py), [client.py](client.py), [model.py](model.py), [utils.py](utils.py).

---

## 5. Does the gRPC path carry real model weights?

**The weights are real, not faked or random — but they are never transmitted, because the client crashes before the first round.**

### The weights are genuine

Nothing in this repo synthesizes weights. There is no `np.random`, no placeholder tensor, no stub. [client.py:20–26](client.py#L20-L26):

```python
def get_parameters(self):
    return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

def set_parameters(self, parameters):
    params_dict = zip(self.model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    self.model.load_state_dict(state_dict, strict=True)
```

These are the actual trained `state_dict` tensors. `fit()` ([client.py:28–38](client.py#L28-L38)) performs real backprop before returning them. Flower serializes these ndarrays and moves them over gRPC internally — the transport is the library's real gRPC channel, not a mock.

### But the run dies before any weights move

[client.py:20](client.py#L20) declares `def get_parameters(self)`. The `flwr==1.7.0` interface is `def get_parameters(self, config)` (`flwr/client/numpy_client.py:95`), and the library invokes it **by keyword** (`flwr/client/numpy_client.py:225`):

```python
parameters = self.numpy_client.get_parameters(config=ins.config)
```

Reproduced live against `flwr==1.7.0`:

```
TypeError: C.get_parameters() got an unexpected keyword argument 'config'
```

This fires at the very first `GetParameters` call — the server's parameter initialization, before round 1 — so **no weights ever reach the wire**. The gRPC path is honest in design and dead in practice.

**Second, independent blocker (Docker only):** [client.py:63](client.py#L63) hardcodes `server_address="0.0.0.0:8080"`. Inside a client container `0.0.0.0` is that container itself, not the `server` service. Even with the signature fixed, the Dockerized clients would fail to connect; it needs `server:8080`. The `CID` env vars set at [docker-compose.yml:14,20](docker/docker-compose.yml#L14) are never read either — [client.py:57](client.py#L57) only parses `--cid`, and both Dockerfiles hardcode `--cid 0` ([Dockerfile.client:8](docker/Dockerfile.client#L8)), so both clients would train on the identical shard.

---

## 6. Tests

**Not present.** No test exists in this repository.

- No `tests/` directory, no `test_*.py`, no `*_test.py`, no `conftest.py` — `find . -type f` returns the 13 files in §1.
- Grep for `test_|pytest|unittest` across every file: **no matches**.
- No test runner in [requirements.txt](requirements.txt); no CI configuration of any kind.

Test count: **0**. There is no assertion anywhere in the codebase.

---

## 7. Repo hygiene

**Framing correction:** with no `.git`, nothing is *tracked* and nothing is *committed*. The findings below are about the working tree and about what the (never-applied) `.gitignore` would do.

### 7a. `__pycache__`

**Not present.** No `__pycache__` directory exists anywhere in the tree. Also correctly covered by [.gitignore:2–3](.gitignore#L2-L3) (`__pycache__/`, `*.pyc`), so it would not be picked up once a repo is initialized. Nothing to remediate.

### 7b. The `=1.3.0` file

**Not present.** No file matching `*=*` exists. The artifact you are thinking of — produced by e.g. `pip install foo >=1.3.0` under an unquoted shell redirect — is not in this tree. Nothing to remediate.

### 7c. `.gitignore` adequacy

Current contents ([.gitignore](.gitignore), 7 lines): `venv/`, `__pycache__/`, `*.pyc`, `.DS_Store`, `*.pth`, `*.pt`, `*.zip`.

Reasonable for a Python/PyTorch project, with one **material gap**:

- **`data/` is not ignored.** [utils.py:9](utils.py#L9) downloads MNIST to `./data` on first run, with `download=True`. That directory reaches roughly 110 MB of `.gz` and `.idx` files. `*.zip` at [.gitignore:7](.gitignore#L7) does **not** match `.gz`. The very first `git add .` after a training run would stage the entire MNIST dataset. This is the single highest-value line to add.

Lesser gaps: no `.venv/` (only `venv/`), no `.idea/`/`.vscode/`, no `*.egg-info/`, no `.pytest_cache/`.

Also missing and relevant to §8: **no `.dockerignore`** anywhere.

### 7d. Generated protobuf stubs

**Not present, and correctly so.** No `.proto` files, no `*_pb2.py`, no `*_pb2_grpc.py` in the tree. Flower ships its own precompiled protobuf/gRPC stubs inside the installed `flwr` package, so an application repo like this one has no reason to carry or generate any. This is the correct arrangement — nothing to remediate.

### 7e. Not asked, but worth flagging

- **No LICENSE file.** A public "starter" repo without one is not legally reusable.
- **[GIT_SETUP.md:19](GIT_SETUP.md#L19)** instructs `git add docker/ docker-compose.yml` — but `docker-compose.yml` is at [docker/docker-compose.yml](docker/docker-compose.yml), not the root. That command would fail with `pathspec 'docker-compose.yml' did not match any files`.

---

## 8. Do the Docker build and `docker-compose up --scale client=5` succeed?

**No. Both fail. Neither reaches the point of running any Python.** Verified live with Docker 28.4.0 / Compose v2.39.4.

### 8a. `docker compose build` — fails

Every `COPY` in both Dockerfiles references a path outside the build context. [docker-compose.yml:5–6](docker/docker-compose.yml#L5-L6) sets `context: ./`, which Compose resolves **relative to the compose file**, i.e. to `docker/` — not the project root. `docker/` contains no `requirements.txt`, no `server.py`, no `client.py`. BuildKit clamps `../x` to the context root, so it looks for `/x` inside `docker/` and finds nothing.

Server image (`docker compose build --no-cache server`), verbatim:

```
#8 [5/5] COPY ../server.py /app/server.py
#8 ERROR: failed to calculate checksum of ref ...: "/server.py": not found
#9 [3/5] COPY ../requirements.txt /app/requirements.txt
#9 ERROR: failed to calculate checksum of ref ...: "/requirements.txt": not found

Dockerfile.server:5
--------------------
   3 |     COPY ../requirements.txt /app/requirements.txt
   4 |     RUN pip install --no-cache-dir -r /app/requirements.txt
   5 | >>> COPY ../server.py /app/server.py
   6 |     CMD ["python","/app/server.py"]
--------------------
failed to solve: failed to compute cache key: failed to calculate checksum of ref ...: "/server.py": not found
```

Both `COPY` layers fail — `pip install` at [Dockerfile.server:4](docker/Dockerfile.server#L4) is never reached.

Client image (`docker compose build --no-cache client0`):

```
#11 ERROR: failed to calculate checksum of ref ...: "/utils.py": not found
   7 | >>> COPY ../utils.py /app/utils.py
failed to solve: failed to compute cache key: failed to calculate checksum of ref ...: "/utils.py": not found
```

Note the build context transferred as **2 B** (`#5 transferring context: 2B done`) — effectively empty, confirming the context is `docker/` and not the project root.

**Fix:** set `context: ..` in [docker-compose.yml](docker/docker-compose.yml) and rewrite the `COPY` lines to be context-relative (`COPY requirements.txt /app/`, etc.). Add a `.dockerignore` for `data/` and `venv/` at the same time, or the ~110 MB MNIST directory is uploaded into every build context.

### 8b. `docker compose up --scale client=5` — fails

```
$ docker compose up --scale client=5
no such service: client: not found
```

There is no service named `client`. [docker-compose.yml](docker/docker-compose.yml) defines three hardcoded services — `server` ([line 3](docker/docker-compose.yml#L3)), `client0` ([line 9](docker/docker-compose.yml#L9)), `client1` ([line 15](docker/docker-compose.yml#L15)). The file is structurally incapable of scaling: `--scale` requires a single service to replicate, and per-replica env vars cannot be set that way, so the `CID=0`/`CID=1` pattern at [lines 13–14, 19–20](docker/docker-compose.yml#L13-L20) does not generalize to N clients regardless.

This command has never worked, and would not work even if §8a were fixed.

### 8c. Three further failures queued behind these

Even after fixing the build context and adding a scalable `client` service, the stack still would not train:

1. `CID` is never read — [client.py:57](client.py#L57) parses only `--cid`; both Dockerfiles hardcode `--cid 0 --num-clients 2` ([Dockerfile.client:8](docker/Dockerfile.client#L8)). All replicas would train identical shards.
2. `server_address="0.0.0.0:8080"` ([client.py:63](client.py#L63)) does not resolve to the server container; needs `server:8080`.
3. The `get_parameters(self)` signature crash from §5 aborts the client at round 0 regardless of transport.

### 8d. Local (non-Docker) path

[README.md:16–30](README.md#L16-L30) is also broken on this host. [requirements.txt](requirements.txt) pins `torch==2.0.1` / `torchvision==0.15.2`, which publish wheels for **cp39, cp310, cp311 only**. This machine's interpreters are Python 3.14 (default), 3.13, and 3.11 — so `pip install -r requirements.txt` on the default interpreter fails with no matching distribution. Only the 3.11 interpreter would work. `flwr==1.7.0` is `py3-none-any` and unaffected.

Also, [docker-compose.yml:1](docker/docker-compose.yml#L1) `version: '3.8'` is obsolete and emits a warning on every invocation.

---

## Claims the code supports

1. **FedAvg aggregation is a true sample-count-weighted average.** `Σ n_k·w_k / Σ n_k`, via `flwr==1.7.0`'s `aggregate()` — not an unweighted mean, not a placeholder. (Supplied by the library, not by this repo; and numerically equal to an unweighted mean given the equal-sized partitions in [utils.py](utils.py).)
2. **Real MNIST, real CNN, real local training.** [utils.py:9](utils.py#L9) downloads the genuine dataset; [model.py](model.py) is a working PyTorch CNN; [client.py:28–38](client.py#L28-L38) is a real forward/backward/step loop. Nothing is mocked.
3. **Weights are genuine model state, never faked or random.** [client.py:20–26](client.py#L20-L26) reads and writes the actual `state_dict`. There is no synthetic-weight code path anywhere.
4. **Held-out test evaluation is implemented client-side.** [client.py:18](client.py#L18) uses the official MNIST test split (`train=False`), disjoint from training data. The code is correct as written.
5. **Data partitioning is deterministic and disjoint.** [utils.py:11–17](utils.py#L11-L17) gives each client a non-overlapping contiguous range.
6. **`.gitignore` correctly covers `__pycache__`, `.pyc`, venv, and model checkpoints.** No `__pycache__` and no `=1.3.0` file are present in the tree.
7. **No stray protobuf stubs.** Correctly relies on the stubs bundled inside `flwr`.
8. **The TF/TFF/tensorflow-privacy matrix does resolve on Python 3.10 — with the jax-releases index.** `tensorflow==2.14.1` + `tensorflow-federated==0.87.0` + `tensorflow-privacy==0.9.0` + `jaxlib==0.4.14`, verified by live resolution on linux/amd64.

## Claims the code does not support

1. **"Repository."** There is no `.git` here. Nothing is committed or tracked; [GIT_SETUP.md](GIT_SETUP.md) describes a history that was never created, and [.gitignore](.gitignore) is currently inert.
2. **That it runs at all.** [client.py:20](client.py#L20) `get_parameters(self)` mismatches the `flwr==1.7.0` interface `get_parameters(self, config)`; the library calls it with `config=` and raises `TypeError: got an unexpected keyword argument 'config'` at parameter initialization — before round 1, before any weights are transmitted. **No end-to-end federated round has ever completed with these pins.**
3. **"Multiple Dockerized clients"** ([README.md:3](README.md#L3)). Both images fail to build: every `COPY ../…` escapes the build context that [docker-compose.yml:5](docker/docker-compose.yml#L5) sets to `docker/`. `pip install` is never even reached.
4. **`docker-compose up --scale client=5`.** Errors with `no such service: client`. Only hardcoded `client0`/`client1` exist; the file cannot express a scalable client.
5. **That Docker clients would federate if built.** `server_address="0.0.0.0:8080"` ([client.py:63](client.py#L63)) points a client at itself, and `CID` env vars ([docker-compose.yml:14,20](docker/docker-compose.yml#L14)) are never read — both replicas hardcode `--cid 0`, training identical data.
6. **"Non-iid partitioning"** ([utils.py:10](utils.py#L10)). Contiguous index ranges over an unsorted MNIST are approximately **IID**. The comment asserts the opposite of what the code does.
7. **That any accuracy number is ever reported.** [server.py:6](server.py#L6) passes no `evaluate_metrics_aggregation_fn`, so the `{"accuracy": …}` from [client.py:53](client.py#L53) is discarded with `WARNING: No evaluate_metrics_aggregation_fn provided`. There is also no `evaluate_fn`, so the global model is never centrally evaluated.
8. **Any TensorFlow / TFF / tensorflow-privacy content.** Zero matches across the tree. No `simulation.py`. No parallel TFF implementation, real or stubbed.
9. **Any test coverage.** Zero tests, zero assertions, no test runner, no CI.
10. **`pip install tensorflow-federated` (unpinned) giving you TFF.** It silently resolves to the 2019 placeholder `tensorflow-federated==0.1.0` — no federated API — because `jaxlib==0.4.14` has been removed from PyPI. It also cannot be installed on this Windows host at all: TFF ships **Linux-only** `manylinux_2_31_x86_64` wheels and requires Python `>=3.9,<3.12`, while this machine runs 3.14/3.13/3.11.
11. **The local quickstart on this machine.** [README.md:16–30](README.md#L16-L30) fails on the default interpreter: `torch==2.0.1`/`torchvision==0.15.2` have no cp312+ wheels, and this host's default Python is 3.14.
12. **[GIT_SETUP.md:19](GIT_SETUP.md#L19)'s `git add docker-compose.yml`.** That path does not exist at the root; the file is at [docker/docker-compose.yml](docker/docker-compose.yml).
