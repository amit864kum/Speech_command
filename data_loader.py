# data_loader.py
import os
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import random_split
import torch
import torch.nn.functional as F

class SpeechCommandsDataLoader:
    def __init__(self, num_clients: int, download_dir: str = "./data"):
        self.num_clients = num_clients
        self.download_dir = download_dir
        self.target_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        
        # Load the dataset first to determine the max length
        self.dataset = self._load_dataset()
        self.max_mfcc_len = self._get_max_mfcc_length()
        print(f"Max MFCC length found: {self.max_mfcc_len}")
        
        # Now, process and pad the data
        self.processed_dataset = self._process_data()
        self.client_data = self._split_data()

    def _load_dataset(self):
        """Loads and downloads the Speech Commands dataset without processing."""
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        
        print("Downloading Speech Commands dataset...")
        dataset = SPEECHCOMMANDS(root=self.download_dir, download=True)
        print("Dataset loaded successfully.")
        return dataset

    def _get_max_mfcc_length(self):
        """Calculates the maximum MFCC feature length in the dataset."""
        max_len = 0
        feature_extractor = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=64,
            melkwargs={'n_mels': 64}
        )
        
        for waveform, sample_rate, label, speaker_id, utterance_number in self.dataset:
            if label in self.target_words:
                mfccs = feature_extractor(waveform)
                if mfccs.shape[2] > max_len:
                    max_len = mfccs.shape[2]
        return max_len

    def _process_data(self):
        """Processes and pads the dataset into features and labels."""
        processed_data = []
        label_map = {word: i for i, word in enumerate(self.target_words)}
        feature_extractor = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=64,
            melkwargs={'n_mels': 64}
        )
        
        for waveform, sample_rate, label, speaker_id, utterance_number in self.dataset:
            if label in self.target_words:
                mfccs = feature_extractor(waveform)
                
                # Pad the MFCC tensor to the max length
                pad_amount = self.max_mfcc_len - mfccs.shape[2]
                padded_mfccs = F.pad(mfccs, (0, pad_amount))
                
                # Corrected: Permute the dimensions to be [channels, features]
                # This ensures the tensor is in the correct format for the model.
                processed_data.append((padded_mfccs.permute(0, 2, 1).squeeze(0), label_map[label]))
                
        return processed_data

    def _split_data(self):
        """Splits the processed dataset among the clients (miners)."""
        num_samples = len(self.processed_dataset)
        split_sizes = [num_samples // self.num_clients] * self.num_clients
        remaining = num_samples % self.num_clients
        for i in range(remaining):
            split_sizes[i] += 1
            
        return random_split(self.processed_dataset, split_sizes)
        
    def get_client_data(self, client_id: int):
        """Returns the data for a specific client."""
        return self.client_data[client_id]