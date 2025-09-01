import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from data_loader import SpeechCommandsDataLoader  # Import the real data loader
from model import SimpleAudioClassifier  # Import your model

class TestGKWSModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load a small, real dataset once for all tests."""
        print("Loading real dataset for testing...")
        data_loader = SpeechCommandsDataLoader(num_clients=1)
        full_dataset = data_loader.get_client_data(client_id=0)
        
        # Take a small subset of the real data for fast testing
        dataset_size = len(full_dataset)
        subset_size = min(100, dataset_size) # Use at most 100 samples
        cls.test_dataset = Subset(full_dataset, range(subset_size))
        
        cls.input_dim = 64
        cls.num_classes = 10
        cls.device = torch.device("cpu")
        print("Real dataset loaded successfully for testing.")

    def test_forward_pass_output_shape(self):
        """
        Tests that the model's output shape is correct for the real data.
        """
        model = SimpleAudioClassifier(self.input_dim, self.num_classes)
        test_loader = DataLoader(self.test_dataset, batch_size=16)
        
        features, labels = next(iter(test_loader))
        
        # Permute the features for the Conv1D model
        features = features.permute(0, 2, 1)

        output = model(features)
        
        self.assertEqual(output.shape, torch.Size([16, self.num_classes]))
    
    def test_training_on_real_data(self):
        """
        Tests if the model's loss decreases after a few training steps on a real dataset.
        """
        model = SimpleAudioClassifier(self.input_dim, self.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Use a small subset of the real data for training
        test_loader = DataLoader(self.test_dataset, batch_size=16)

        # Get the initial loss
        model.eval()
        features, labels = next(iter(test_loader))
        features = features.permute(0, 2, 1)
        initial_outputs = model(features)
        initial_loss = criterion(initial_outputs, labels)
        
        # Train the model for one epoch
        model.train()
        for features_batch, labels_batch in test_loader:
            features_batch = features_batch.permute(0, 2, 1)
            optimizer.zero_grad()
            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
        
        # Get the final loss after training
        model.eval()
        final_outputs = model(features)
        final_loss = criterion(final_outputs, labels)

        # Assert that the loss has decreased
        self.assertLess(final_loss.item(), initial_loss.item(), "Loss should decrease after training.")

if __name__ == '__main__':
    unittest.main()
