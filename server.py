# Simple Flower server coordinating FedAvg
import flwr as fl

def main():
    # Start strategy with 2 rounds and fit/evaluate config
    strategy = fl.server.strategy.FedAvg(min_fit_clients=2, min_available_clients=2)
    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)

if __name__ == '__main__':
    main()
