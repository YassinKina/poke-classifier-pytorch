from .utils import NestedProgressBar, get_best_val_accuracy, flatten_config
import torch
import os
import wandb
from omegaconf import OmegaConf, DictConfig
import optuna
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Optional, Any

def train_epoch(model: torch.nn.Module, 
                train_dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                loss_func: torch.nn.modules.loss._Loss, 
                device: torch.device, 
                pbar: Any) -> Tuple[float, float]:
    """
    Trains the model for a single epoch.

    Iterates over the training dataloader, performs forward and backward passes,
    updates model weights via the optimizer, and aggregates loss and accuracy.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_dataloader (DataLoader): The DataLoader containing the training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        loss_func (torch.nn.modules.loss._Loss): The loss function used for training.
        device (torch.device): The device (CPU/GPU/MPS) to perform training on.
        pbar (Any): A progress bar handler object to visualize training progress.

    Returns:
        Tuple[float, float]: A tuple containing the average epoch loss and accuracy.
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        pbar.update_batch(batch_idx + 1)
        inputs, labels = inputs.to(device), labels.to(device)
       
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted_class = outputs.max(1)
        total += labels.size(0)
        correct += predicted_class.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc
       

def validate_epoch(model: torch.nn.Module, 
                   val_dataloader: DataLoader, 
                   loss_func: torch.nn.modules.loss._Loss, 
                   device: torch.device) -> Tuple[float, float]:
    """
    Evaluates the model on the validation dataset for a single epoch.

    Performs a forward pass in evaluation mode without gradient calculations 
    to estimate the model's performance on unseen data.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        val_dataloader (DataLoader): The DataLoader containing validation samples.
        loss_func (torch.nn.modules.loss._Loss): The criterion used to calculate loss.
        device (torch.device): The hardware device for computation.

    Returns:
        Tuple[float, float]: A tuple containing average validation loss and accuracy.
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = running_loss / total
        avg_val_accuracy = correct / total
            
    return avg_val_loss, avg_val_accuracy    

def train_model(model: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                scheduler: Any, 
                loss_func: torch.nn.modules.loss._Loss, 
                train_dataloader: DataLoader, 
                val_dataloader: DataLoader, 
                device: torch.device, 
                n_epochs: int, 
                wandb_run: Optional[Any] = None, 
                trial: Optional[optuna.trial.Trial] = None) -> float:
    """
    Manages the full training and validation loop over multiple epochs.

    Handles epoch iteration, model checkpointing (saving the best model), 
    learning rate scheduling, W&B logging, and Optuna pruning logic.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating weights.
        scheduler (Any): Learning rate scheduler (e.g., ReduceLROnPlateau).
        loss_func (torch.nn.modules.loss._Loss): The loss function.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to perform training on.
        n_epochs (int): Total number of epochs to train.
        wandb_run (Optional[Any]): Initialized W&B run for logging.
        trial (Optional[optuna.trial.Trial]): Optuna trial for hyperparameter optimization.

    Returns:
        float: The best validation accuracy achieved during the training process.
    """
    pbar = NestedProgressBar(
        total_epochs=n_epochs,
        total_batches=len(train_dataloader),
        epoch_message_freq=1,
        mode="train",
        use_notebook=False
    )
    best_val_acc = get_best_val_accuracy()
    print("BEST VAL ACC: ", best_val_acc) # delete this after verifying
    
    for epoch in range(n_epochs):
        pbar.update_epoch(epoch + 1)
        
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, loss_func, device, pbar)
        val_loss, val_acc = validate_epoch(model, val_dataloader, loss_func, device)
        
        scheduler.step(val_loss)
        
        status_msg = f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}"
        pbar.maybe_log_epoch(epoch + 1, message=status_msg)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "train/lr": current_lr
            })
        # Save checkpoint if model is performing better than previously
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = "models/pokemon_cnn_best.pth"
            os.makedirs("models", exist_ok=True)
            
            checkpoint = {
                'state_dict': model.state_dict(),
                'accuracy': val_acc,
                'epoch': epoch,
                'run_id': wandb.run.id if wandb.run else None  
            }
            torch.save(checkpoint, save_path)
            
        # Prune optuna trial if it is performing poorly    
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                if wandb_run is not None:
                    wandb_run.finish(exit_code=1)
                raise optuna.exceptions.TrialPruned()
        
    pbar.close("Training complete!\n")
    if wandb_run is not None:
        wandb_run.finish()
        
    return best_val_acc


def init_wandb_run(config: DictConfig, run_name: str) -> Any:
    """
    Initializes a Weights & Biases run with flattened configuration parameters.

    Converts Hydra DictConfigs to standard dictionaries and prepares the environment
    for a stable W&B initialization, specifically optimized for macOS threading.

    Args:
        config (DictConfig): The Hydra configuration containing hyperparameters.
        run_name (str): The descriptive name for the W&B run.

    Returns:
        Any: The initialized W&B run object.
    """
    os.environ["WANDB_START_METHOD"] = "thread"
    config_dict = OmegaConf.to_container(config, resolve=True)
    flat_config = flatten_config(config_dict)

    run = wandb.init(
        entity="yassinbkina",
        project="pokemon-classification",
        config=flat_config, 
        name=run_name,
        group="optuna_hpo",
        settings=wandb.Settings(start_method="thread"), 
        reinit=True
    )
    
    print("WANDB config: ", wandb.config)
    return run

@torch.no_grad()
def test_model(model: torch.nn.Module, 
               test_loader: DataLoader, 
               device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates the model on the test dataset to calculate final hold-out accuracy.

    Iterates through the test set once, collecting all predictions and true labels
     to return for further analysis (e.g., confusion matrices).

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to perform evaluation on.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (ground_truth_labels, predicted_labels).
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Starting Final Test Evaluation...")
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    
    return all_labels, all_preds