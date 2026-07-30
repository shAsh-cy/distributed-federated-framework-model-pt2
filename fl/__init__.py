"""Federated learning on Fashion-MNIST with TensorFlow Federated.

Submodules
----------
config       Typed configuration object loaded from YAML.
models       Keras CNN definitions.
data         Dataset loading and IID / Dirichlet non-IID client partitioning.
aggregation  FedAvg arithmetic and differentially private aggregation.
server       gRPC coordinator: client sampling, round barrier, evaluation.
client       gRPC participant: local training against its own shard.
proto        Versioned wire format (fl_comm.proto) and generated stubs.
"""

__version__ = "0.2.0"
