import concurrent.futures
import time
from client import DecentralizedClient
from data_loader import SpeechCommandsDataLoader
import os.path

# ----------------------------------------------
# Configuration
# ----------------------------------------------
EPOCHS = 4
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LR = 1e-3

NUM_CLASSES = 10
INPUT_DIM = 64
BLOCKCHAIN_FILE = "blockchain.json"

MINER_IDS = ["Miner_1", "Miner_2", "Miner_3"]
CLIENT_IDS = ["Device_A", "Device_B", "Device_C"]

if __name__ == "__main__":
    print("Starting a fully decentralized federated learning simulation...")
    
    # Initialize a data loader for all clients
    data_loader = SpeechCommandsDataLoader(num_clients=len(CLIENT_IDS))

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(CLIENT_IDS)) as executor:
        futures = {
            executor.submit(
                DecentralizedClient(
                    client_id=cid,
                    miner_id=MINER_IDS[i],
                    client_data=data_loader.get_client_data(client_id=i),
                    input_dim=INPUT_DIM,
                    output_dim=NUM_CLASSES,
                    target_words=data_loader.target_words
                ).run_client_loop
            )
            for i, cid in enumerate(CLIENT_IDS)
        }
    
    print("All clients have finished their tasks.")
