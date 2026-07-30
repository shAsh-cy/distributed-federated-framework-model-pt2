import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_dataloader(cid=0, num_clients=2, batch_size=32, train=True):
    # Partition MNIST by client id for demo purposes
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=train, download=True, transform=transform)
    # Simple non-iid partitioning: split by index ranges
    n = len(dataset)
    indices = list(range(n))
    # deterministic split: each client gets contiguous chunk
    chunk_size = n // num_clients
    start = cid * chunk_size
    end = start + chunk_size if cid < num_clients - 1 else n
    subset = Subset(dataset, indices[start:end])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader
