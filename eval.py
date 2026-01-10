import torch
from src import DynamicCNN
from src import create_dataloaders
from src import test_model
import yaml

def run_evaluation():
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
        num_classes=150
    ).to(device)

    # Load the best weights you saved during training
    model.load_state_dict(torch.load("models/pokemon_cnn_best.pth"))
    
    # 4. Run Test
    labels, preds = test_model(model, test_loader, device)
    
   # Implement top 5 as well

if __name__ == "__main__":
    run_evaluation()