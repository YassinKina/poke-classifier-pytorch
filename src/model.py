import torch.nn as nn
import torch
from typing import List, Tuple

class DynamicCNN(nn.Module):
    """
    A flexible Convolutional Neural Network with a dynamically calculated classifier.

    This model constructs a feature extractor based on specified layer counts and 
    filter sizes, then automatically determines the required input size for the 
    fully connected classifier based on the resulting feature map dimensions.
    """
    def __init__(
        self, 
        n_layers: int, 
        n_filters: List[int], 
        kernel_sizes: List[int], 
        dropout_rate: float, 
        fc_size: int, 
        num_classes: int, 
        input_shape: Tuple[int, int, int] = (3, 224, 224)
    ):
        """
        Initializes the DynamicCNN architecture with dynamic feature extraction and classification heads.

        Args:
            n_layers (int): The number of convolutional blocks to create.
            n_filters (List[int]): Number of output filters for each convolutional block.
            kernel_sizes (List[int]): Kernel size for each convolutional layer.
            dropout_rate (float): The dropout probability for the classifier layers.
            fc_size (int): The number of neurons in the hidden fully connected layer.
            num_classes (int): The number of output classes (e.g., 150 for Pokémon).
            input_shape (Tuple[int, int, int]): The (RGB, H, W) shape of input images.
        """
        super(DynamicCNN, self).__init__()
        
        # Initialize convolutional blocks
        blocks = []
        in_channels = 3 # RGB channels
        
        for i in range(n_layers):
            # Get params for curr layer
            out_channels = n_filters[i]
            kernel_size = kernel_sizes[i]
            # Calculate padding to maintain the input spatial dimensions 
            padding = (kernel_size - 1) // 2
            # Define conv block architecture
            block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Add block to blocks list
            blocks.append(block)
            # Adjust in channels for next layer
            in_channels = out_channels
        
        # Combine all blocks into a single feature extractor module
        self.features = nn.Sequential(*blocks)
        
        self.dropout_rate = dropout_rate
        self.fc_size = fc_size
        self.num_classes = num_classes
        
        # Calc flattened size dynamically
        with torch.no_grad():
            # Create a fake image and pass it through the features only
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
            flattened_size = int(dummy_output.numel())
       
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(flattened_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_size, num_classes)
        )
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output logits of shape (batch_size, num_classes).
        """
        # Pass input through feature extraction layers
        x = self.features(x)
        
        # Flatten all dims except batch size to pass into fully connected model
        flattened = torch.flatten(x, start_dim=1)
            
        # Pass flattened layers through classifier for final output  
        return self.classifier(flattened)