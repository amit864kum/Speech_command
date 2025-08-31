import torch
import torch.optim as optim
from model import GKWS_CNN # Import your new model
from fl_trainer import FLTrainer
from data_loader import GKWSDataLoader # Import your new data loader

class FLNode:
    def __init__(self, node_id, num_clients, num_classes=10, learning_rate=0.01):
        self.node_id = node_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize a local copy of the global model
        self.local_model = GKWS_CNN(num_classes=num_classes).to(self.device)
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=learning_rate)
        
        self.trainer = FLTrainer(self.local_model, self.optimizer, learning_rate, self.device)
        self.data_loader = GKWSDataLoader(num_clients=num_clients, keywords=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'])
        
        # Load the specific data partition for this node
        self.local_data = self.data_loader.get_client_data(client_id=node_id)
    
    def receive_global_model(self, global_model_state_dict):
        """Receives the global model from the server and updates the local model."""
        self.local_model.load_state_dict(global_model_state_dict)
        print(f"Node {self.node_id}: Received global model.")
        
    def perform_local_training(self, local_epochs=5):
        """Performs local training on the node's data."""
        self.trainer.train_epoch(self.local_data, local_epochs=local_epochs)
        print(f"Node {self.node_id}: Local training complete.")

    def get_model_updates(self):
        """Returns the updated local model to the server."""
        return self.local_model.state_dict()

# Example of how the node would work in a round
if __name__ == '__main__':
    # This is a simplified, non-networked example
    node = FLNode(node_id=0, num_clients=10)
    print(f"Node {node.node_id} initialized with {len(node.local_data)} samples.")
    
    # Simulate receiving a global model from the server
    global_model = GKWS_CNN()
    node.receive_global_model(global_model.state_dict())
    
    # Perform a local training round
    node.perform_local_training(local_epochs=5)
    
    # The node is now ready to send its updated model back to the server
    updated_model = node.get_model_updates()
    print("Node is ready to send updates to the server.")