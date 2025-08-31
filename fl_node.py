# fl_node.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import hashlib
import random
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List

# Model definition (same as before)
class SimpleAudioClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleAudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class FLNode:
    def __init__(self, node_id, local_data, input_dim, output_dim, device):
        self.node_id = node_id
        self.device = device
        self.model = SimpleAudioClassifier(input_dim=input_dim, output_dim=output_dim).to(device)
        self.local_data = local_data
        self.target_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

    def get_model_weights(self) -> Dict[str, np.ndarray]:
        """
        Returns the model's weights as a dictionary of NumPy arrays.
        """
        # The key change is here: we return .numpy() instead of .tolist()
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def update_model(self, global_weights: Dict[str, torch.Tensor]):
        """
        Updates the local model's weights with the global ones.
        """
        self.model.load_state_dict(global_weights)

    def compute_local_model_hash(self) -> str:
        """
        Computes a SHA256 hash of the local model weights.
        """
        weights = self.get_model_weights()
        weights_str = json.dumps({k: v.tolist() for k, v in weights.items()}, sort_keys=True)
        return hashlib.sha256(weights_str.encode()).hexdigest()

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
        correct = 0
        total = 0
        predictions = []
        with torch.no_grad():
            for features, labels in local_dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                features = features.permute(0, 2, 1)
                outputs = self.model(features)
                
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                for i in range(len(predicted_class)):
                    predictions.append({
                        "predicted_word": self.target_words[predicted_class[i]],
                        "confidence": confidence[i].item() * 100,
                        "true_word": self.target_words[labels[i]]
                    })
                
                total += labels.size(0)
                correct += (predicted_class == labels).sum().item()

        accuracy = correct / total
        weights_serialized = self.get_model_weights()
        model_id = self.compute_local_model_hash()
        
        sample_size = min(5, len(predictions))
        sample_predictions = random.sample(predictions, sample_size)
        
        return weights_serialized, accuracy, model_id, sample_predictions