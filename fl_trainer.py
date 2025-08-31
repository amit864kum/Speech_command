import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List
import hashlib
import json
import numpy as np

def aggregate_local_models(local_weights: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """
    Performs federated averaging (FedAvg) on a dictionary of local model weights.
    
    :param local_weights: A dictionary where keys are client IDs and values are
                          their model's state dictionaries (NumPy arrays).
    :return: The aggregated global model's state dictionary (PyTorch Tensors).
    """
    if not local_weights:
        return {}
    
    # Get the list of clients and the number of clients
    clients = list(local_weights.keys())
    num_clients = len(clients)
    
    # Initialize the global weights with the first client's weights,
    # converting them to a PyTorch Tensor
    global_weights = {
        name: torch.from_numpy(param) 
        for name, param in local_weights[clients[0]].items()
    }
    
    # Sum the weights from all other clients, converting them to Tensors
    for client_id in clients[1:]:
        for layer_name in global_weights.keys():
            global_weights[layer_name] += torch.from_numpy(local_weights[client_id][layer_name])
            
    # Average the weights
    for layer_name in global_weights.keys():
        global_weights[layer_name] = torch.div(global_weights[layer_name], num_clients)
        
    return global_weights

def compute_global_model_hash(global_weights: Dict[str, torch.Tensor]) -> str:
    """
    Computes a SHA256 hash of the global model weights.
    This provides a unique, verifiable identifier for the model.
    """
    # Create a string representation of the ordered weights
    weights_str = json.dumps({
        name: param.cpu().numpy().tolist() for name, param in global_weights.items()
    }, sort_keys=True)
    
    # Return the SHA256 hash
    return hashlib.sha256(weights_str.encode()).hexdigest()