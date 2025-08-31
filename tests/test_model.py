import unittest
import torch
from model import GKWS_CNN

class TestGKWSModel(unittest.TestCase):
    def test_forward_pass_output_shape(self):
        """
        Tests that the model's output shape is correct for a given input.
        """
        # Define expected parameters based on a batch of 16-kHz audio
        batch_size = 4
        num_classes = 10
        
        # Create a dummy input tensor matching the expected shape: (batch_size, channels, height, width)
        # Assuming a mono channel and 64 mels, a 1-second clip gives a width of ~49.
        dummy_input = torch.randn(batch_size, 1, 64, 49)
        
        # Initialize the model
        model = GKWS_CNN(num_classes=num_classes)
        
        # Perform the forward pass
        output = model(dummy_input)
        
        # Assert the output shape is as expected
        self.assertEqual(output.shape, torch.Size([batch_size, num_classes]))
        
if __name__ == '__main__':
    unittest.main()