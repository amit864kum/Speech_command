import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import hashlib
import random
import time
from typing import Dict, List, Tuple
from fl_node import SimpleAudioClassifier
from data_loader import SpeechCommandsDataLoader
from ehr_chain import EHRChain
from block import Block
import numpy as np
import torch.nn.functional as F

# FL Utility functions (moved from fl_trainer.py)
def aggregate_local_models(local_weights: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    if not local_weights:
        return {}
    num_participating_clients = len(local_weights)
    clients = list(local_weights.keys())
    global_weights = {name: torch.from_numpy(param) for name, param in local_weights[clients[0]].items()}
    for client_id in clients[1:]:
        for layer_name in global_weights.keys():
            global_weights[layer_name] += torch.from_numpy(local_weights[client_id][layer_name])
    for layer_name in global_weights.keys():
        global_weights[layer_name] = torch.div(global_weights[layer_name], num_participating_clients)
    return global_weights

def compute_global_model_hash(global_weights: Dict[str, torch.Tensor]) -> str:
    weights_str = json.dumps({name: param.cpu().numpy().tolist() for name, param in global_weights.items()}, sort_keys=True)
    return hashlib.sha256(weights_str.encode()).hexdigest()

class DecentralizedClient:
    def __init__(self, client_id, miner_id, client_data, input_dim, output_dim, target_words):
        self.client_id = client_id
        self.miner_id = miner_id
        self.local_data = client_data
        self.target_words = target_words
        self.device = torch.device("cpu")
        self.model = SimpleAudioClassifier(input_dim, output_dim).to(self.device)
        self.private_key = hashlib.sha256(self.client_id.encode()).hexdigest()
        self.ehr_chain = EHRChain() # Each client has its own chain
        self.miner = Miner(self.miner_id, self.ehr_chain, difficulty=1)

    def get_model_weights(self):
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def update_model(self, global_weights):
        weights_tensor = {k: torch.tensor(v) for k, v in global_weights.items()}
        self.model.load_state_dict(weights_tensor)

    def local_train(self, epochs, batch_size, lr) -> Tuple[Dict, float, str, List]:
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        local_dataloader = DataLoader(self.local_data, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for i, (features, labels) in enumerate(local_dataloader):
                features, labels = features.to(self.device), labels.to(self.device)
                features = features.permute(0, 2, 1)
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        self.model.eval()
        correct, total, predictions = 0, 0, []
        with torch.no_grad():
            for features, labels in local_dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                features = features.permute(0, 2, 1)
                outputs = self.model(features)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                for i in range(len(predicted_class)):
                    predictions.append({"predicted_word": self.target_words[predicted_class[i]], "confidence": confidence[i].item() * 100})
                total += labels.size(0)
                correct += (predicted_class == labels).sum().item()
        
        accuracy = correct / total
        weights_serialized = self.get_model_weights()
        model_id = compute_global_model_hash({self.client_id: weights_serialized})
        
        sample_size = min(5, len(predictions))
        sample_predictions = random.sample(predictions, sample_size)
        
        return weights_serialized, accuracy, model_id, sample_predictions
    
    def run_client_loop(self):
        print(f"[✅] Client {self.client_id} is running its decentralized loop.")
        # Simplified decentralized loop
        for epoch in range(1, EPOCHS + 1):
            print(f"\n====================== EPOCH {epoch} ======================")
            
            # This client trains its local model
            weights, acc, model_id, preds = self.local_train(LOCAL_EPOCHS, BATCH_SIZE, LR)
            
            # In a real system, this client would broadcast its weights to peers.
            # For our simulation, we just mine a block.
            txs = [{"miner": self.miner_id, "predictions": preds}]
            winner_block = self.miner.mine_block(txs, model_id)
            
            if winner_block:
                self.ehr_chain.add_block(winner_block)
                print(f"[➡️] Client {self.client_id} successfully mined and appended Block #{winner_block.index}.")
            
            self.ehr_chain.save_to_file()
