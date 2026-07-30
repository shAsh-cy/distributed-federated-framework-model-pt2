import argparse
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNNMnist
from utils import get_dataloader
from typing import Dict, Tuple, Optional

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, cid, num_clients):
        self.model = model
        self.cid = cid
        self.num_clients = num_clients
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.trainloader = get_dataloader(cid=cid, num_clients=num_clients, train=True)
        self.testloader = get_dataloader(cid=cid, num_clients=num_clients, train=False)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # local epoch =1 for demo
            for data, target in self.trainloader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.testloader:
                output = self.model(data)
                loss += self.criterion(output, target).item() * len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(self.testloader.dataset)
        accuracy = correct / len(self.testloader.dataset)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cid', type=int, default=0)
    parser.add_argument('--num-clients', type=int, default=2)
    args = parser.parse_args()

    model = CNNMnist()
    client = FlowerClient(model, cid=args.cid, num_clients=args.num_clients)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

if __name__ == '__main__':
    main()
