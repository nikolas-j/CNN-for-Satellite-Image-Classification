# The ML model for ship classification
# to which weights will be loaded

from torch import nn
import torch

# Create a convolutional neural network
class CNNShipClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # Here we define the network structure. 
        # Repeated layers of 2D Convolutions with kernel size 4
        # Followed by a ReLU layer
        # Input downsized with MaxPool layers with kernel size 4 -> Chooses brigtest pixel value
        # The output shapes and choise of kernel sizes and network structure is somewhat arbitrary.
        # nn.Sequential executes the layers in a chain

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2), # 3 input color channels, 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4,stride=4), # Downsizing from 80x80 -> 20x20
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4,stride=4) # Downsize 20x20 -> 5x5
        )

        # Classifier block which flattens the output from the convolutional layers
        # Applies linear regression to map the inputs to 2 output neurons
        # Brighter neuron index is chosen as the final classification [0, 1]

        self.classifier = nn.Sequential(
            nn.Flatten(), # Flattens the output into of each channel into a 1D tensor
            nn.Linear(in_features=16*5*5, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2) # Final classification layer with linear regression
        )
    
    # Defines the forward pass of the input data when model(X) is called
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.classifier(x)
        return x

# Example usage:
# model = CNNShipClassifier()
# output = model(input_tensor)  # where input_tensor is a batch of images with shape (batch_size, 3, 80, 80)