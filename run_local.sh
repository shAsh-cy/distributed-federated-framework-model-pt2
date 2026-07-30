#!/bin/bash
# Run server in background and clients in separate terminals (for demo)
echo "Start Flower server on port 8080..."
python server.py &
sleep 2
echo "Starting 2 clients..."
python client.py --cid 0 --num-clients 2 &
python client.py --cid 1 --num-clients 2 &
echo "Done. Check logs in terminal."
