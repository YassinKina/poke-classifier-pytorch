import torch.nn as nn
import torch
class CNN(nn.Module):
    """A flexible CNN with a dynamically created classifer

    """
    def __init__(self, n_layers, n_filters, kernel_sizes, dropout_rate, fc_size, num_classes, input_shape=(3, 224, 224)):
        """
        Initializes the feature extraction part of the CNN.

        Args:
            n_layers: The number of convolutional blocks to create.
            n_filters: A list of integers specifying the number of output
                       filters for each convolutional block.
            kernel_sizes: A list of integers specifying the kernel size for
                          each convolutional layer.
            dropout_rate: The dropout probability to be used in the classifier.
            fc_size: The number of neurons in the hidden fully connected layer.
        """
        super(CNN, self).__init__()
        
        # Initialize convulational blocks
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
            flattened_size = dummy_output.numel() # Total number of elements
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(flattened_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_size, num_classes)
        )
      
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x: The input tensor of shape (batch_size, channels, height, width).

        Returns:
            The output logits from the classifier.
        """
        
        # Pass input through feature extraction layers
        x = self.features(x)
        
        # Flatten all dims except batch size to pass into fully connected model
        flattened = torch.flatten(x, start_dim=1)
        flattened_size = flattened.size(dim=1)
            
        # Pass flattened layers through classifer for final output  
        return self.classifier(flattened)