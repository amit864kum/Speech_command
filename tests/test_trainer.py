import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from data_loader import SpeechCommandsDataLoader  # Import the real data loader
from model import SimpleAudioClassifier  # Import your model
from fl_trainer import FLTrainer # Import your FL Trainer

class TestFLTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Load a small, real dataset once for all tests in this class.
        """
        print("Loading real dataset for trainer testing...")
        data_loader = SpeechCommandsDataLoader(num_clients=1)
        full_dataset = data_loader.get_client_data(client_id=0)
        
        # Take a small subset of the real data for fast testing
        dataset_size = len(full_dataset)
        subset_size = min(100, dataset_size) # Use at most 100 samples
        cls.test_dataset = Subset(full_dataset, range(subset_size))
        
        cls.input_dim = 64
        cls.num_classes = 10
        cls.device = torch.device("cpu")
        print("Real dataset loaded successfully for trainer testing.")

    def test_local_model_update(self):
        """
        Tests that the model's parameters are updated after a training epoch.
        """
        model = SimpleAudioClassifier(self.input_dim, self.num_classes).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        trainer = FLTrainer(model, optimizer, 0.01, self.device)
        
        initial_state_dict = model.state_dict()
        
        # Get data from the real dataset loader
        data_samples = [self.test_dataset[i] for i in range(len(self.test_dataset))]
        trainer.train_epoch(data_samples, local_epochs=1)
        
        updated_state_dict = model.state_dict()
        
        weights_are_equal = True
        for key in initial_state_dict.keys():
            if not torch.equal(initial_state_dict[key], updated_state_dict[key]):
                weights_are_equal = False
                break
        
        self.assertFalse(weights_are_equal, "The model weights should have been updated after training.")

    def test_loss_reduction(self):
        """
        Tests that the loss decreases after one or more training epochs on a real dataset.
        """
        model = SimpleAudioClassifier(self.input_dim, self.num_classes).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        trainer = FLTrainer(model, optimizer, 0.01, self.device)
        
        # Get data from the real dataset loader
        data_samples = [self.test_dataset[i] for i in range(len(self.test_dataset))]
        
        model.eval()
        features = torch.stack([d[0] for d in data_samples]).unsqueeze(1).to(self.device)
        labels = torch.tensor([d[1] for d in data_samples], dtype=torch.long).to(self.device)
        initial_outputs = model(features)
        initial_loss = trainer.criterion(initial_outputs, labels).item()

        trainer.train_epoch(data_samples, local_epochs=5)
        
        model.eval()
        final_outputs = model(features)
        final_loss = trainer.criterion(final_outputs, labels).item()

        self.assertLess(final_loss, initial_loss, "Loss should decrease after training.")

if __name__ == '__main__':
    unittest.main()
