import torch
import copy
from collections import OrderedDict
from model import GKWS_CNN
from ehr_chain import EHRChain  # Assumes your blockchain file is named this

class FLServer:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.global_model = GKWS_CNN(num_classes=10) # Using your new GKWS model
        self.blockchain = EHRChain() # Initialize the blockchain
        print("Server initialized with a global model and blockchain.")

    def broadcast_global_model(self):
        """Sends the global model to all clients."""
        return self.global_model.state_dict()

    def aggregate_models(self, client_updates):
        """
        Performs Federated Averaging (FedAvg) on client model updates.
        
        Args:
            client_updates (list): A list of client state dictionaries.
        """
        print("Server: Starting model aggregation...")
        # Get the initial state of the global model
        global_state_dict = self.global_model.state_dict()
        
        # Initialize an empty state dict to hold the aggregated weights
        new_global_state_dict = OrderedDict()

        # Iterate through the layers of the global model
        for key in global_state_dict.keys():
            # Sum the weights from all clients for the current layer
            sum_weights = torch.zeros_like(global_state_dict[key])
            for client_sd in client_updates:
                sum_weights += client_sd[key]
            
            # Calculate the average weight for the layer
            avg_weights = sum_weights / len(client_updates)
            new_global_state_dict[key] = avg_weights

        # Update the global model with the new aggregated weights
        self.global_model.load_state_dict(new_global_state_dict)
        print("Server: Model aggregation complete.")

    def log_to_blockchain(self, round_number):
        """Logs the hash of the new global model to the blockchain."""
        # This is a conceptual example. In a real-world scenario, you'd
        # hash the entire model to get a unique identifier for the transaction.
        model_hash = hash(str(self.global_model.state_dict()))
        
        transaction = {
            'round': round_number,
            'model_hash': model_hash,
            'timestamp': torch.now()
        }
        self.blockchain.add_transaction(transaction)
        print(f"Server: Logged transaction for round {round_number} to blockchain.")

    def start_training_round(self, round_number, clients):
        """Orchestrates a single federated learning training round."""
        print(f"\n--- Starting Training Round {round_number} ---")
        
        # 1. Broadcast the global model to all clients
        global_model_state_dict = self.broadcast_global_model()
        
        client_updates = []
        for client in clients:
            # 2. Each client receives the global model
            client.receive_global_model(global_model_state_dict)
            
            # 3. Each client performs local training
            client.perform_local_training(local_epochs=5)
            
            # 4. Each client sends its updated model back
            updated_sd = client.get_model_updates()
            client_updates.append(updated_sd)
        
        # 5. Server aggregates the models
        self.aggregate_models(client_updates)
        
        # 6. Server logs the new model to the blockchain
        self.log_to_blockchain(round_number)
        
        print(f"--- Training Round {round_number} Complete ---")