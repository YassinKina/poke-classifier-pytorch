import torch
from src import DynamicCNN
from src import create_dataloaders
from src import test_model
import yaml

def run_evaluation():
    """
    Executes the final model evaluation on the hold-out test dataset.

    This script loads the project configuration, initializes the DynamicCNN architecture, 
    and restores the optimal model weights from a local checkpoint. It performs a 
    complete forward pass over the test set to generate performance metrics, 
    serving as the final validation of the model's generalization capabilities 
    before deployment.

    The function utilizes a project-specific YAML configuration for architectural 
    consistency and manages device placement (MPS/CPU) automatically.

    Returns:
        None: Results are typically printed to the console or logged for analysis.
    """
    # 1. Load config to get hyperparameters
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 2. Get the Test DataLoader
    _, _, test_loader = create_dataloaders(clean_data_path="data/pokemon_clean", 
                                           batch_size=32)

    # 3. Initialize Model and Load Weights
    model = DynamicCNN(
        n_layers=cfg['model']['n_layers'],
        n_filters=cfg['model']['n_filters'],
        kernel_sizes=cfg['model']['kernel_sizes'],
        dropout_rate=cfg['model']['dropout_rate'],
        fc_size=cfg['model']['fc_size'],
        num_classes=cfg["model"]["num_classes"]
    ).to(device)
    
    # 1. Load the whole dictionary
    checkpoint = torch.load("models/pokemon_cnn_best.pth", map_location=device)

    # Extract only the state_dict (the weights) to load into the model
    model.load_state_dict(checkpoint['state_dict'])
    
    # 4. Run Test
    labels, preds = test_model(model, test_loader, device)
    

if __name__ == "__main__":
    run_evaluation()