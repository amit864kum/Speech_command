import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from fl_trainer import FLTrainer
from model import GKWS_CNN

class TestFLTrainer(unittest.TestCase):
    def setUp(self):
        """
        Set up a simple model, optimizer, and dummy data for testing.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 10
        self.learning_rate = 0.01
        
        # Initialize the model
        self.model = GKWS_CNN(num_classes=self.num_classes).to(self.device)
        
        # Initialize the optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize the trainer
        self.trainer = FLTrainer(self.model, self.optimizer, self.learning_rate, self.device)
        
        # Create a dummy dataset for testing
        # 4 samples, each with a (1, 64, 49) spectrogram and a corresponding label
        self.dummy_data = []
        for i in range(4):
            # Create a spectrogram for a dummy "correct" prediction
            spectrogram = torch.randn(1, 64, 49)
            label = torch.tensor(i % self.num_classes)
            self.dummy_data.append((spectrogram, label))

    def test_local_model_update(self):
        """
        Tests that the model's parameters are updated after a training epoch.
        """
        # 1. Get the initial state of the model's weights
        initial_state_dict = self.model.state_dict()
        
        # 2. Perform a local training epoch
        self.trainer.train_epoch(self.dummy_data, local_epochs=1)
        
        # 3. Get the updated state of the model's weights
        updated_state_dict = self.model.state_dict()
        
        # 4. Compare the initial weights with the updated weights
        weights_are_equal = True
        for key in initial_state_dict.keys():
            if not torch.equal(initial_state_dict[key], updated_state_dict[key]):
                weights_are_equal = False
                break
        
        # Assert that the weights have changed
        self.assertFalse(weights_are_equal, "The model weights should have been updated after training.")

    def test_loss_reduction(self):
        """
        Tests that the loss decreases after one or more training epochs.
        """
        # Get the loss before training
        self.model.eval()
        with torch.no_grad():
            initial_outputs = self.model(torch.stack([s[0] for s in self.dummy_data]).to(self.device))
            initial_loss = self.trainer.criterion(initial_outputs, torch.tensor([s[1] for s in self.dummy_data], dtype=torch.long).to(self.device)).item()

        # Perform local training
        self.trainer.train_epoch(self.dummy_data, local_epochs=10)
        
        # Get the loss after training
        self.model.eval()
        with torch.no_grad():
            final_outputs = self.model(torch.stack([s[0] for s in self.dummy_data]).to(self.device))
            final_loss = self.trainer.criterion(final_outputs, torch.tensor([s[1] for s in self.dummy_data], dtype=torch.long).to(self.device)).item()

        # Assert that the final loss is less than the initial loss
        self.assertLess(final_loss, initial_loss, "The loss should decrease after training.")
        
if __name__ == '__main__':
    unittest.main()