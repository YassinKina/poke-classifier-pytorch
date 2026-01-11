from .utils import NestedProgressBar, get_best_val_accuracy, flatten_config, get_num_correct_in_top5, init_wandb_run
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
        Tuple[float, float, float]: A tuple containing the average epoch loss, top1 accuracy, and top5 accuracy.
    """
    model.train()
    
    top1_correct = 0
    top5_correct = 0
    total = 0
    running_loss = 0.0
    
    
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
        correct_in_top5 = get_num_correct_in_top5(outputs, labels)
        
        total += labels.size(0)
        # Get exactly correct predictions
        top1_correct += predicted_class.eq(labels).sum().item()
        # Get number of preds where one of top5 preds is the correct label
        top5_correct += correct_in_top5
        
        
    epoch_loss = running_loss / total
    top1_epoch_acc = top1_correct / total
    top5_epoch_acc = top5_correct / total
    
    return epoch_loss, top1_epoch_acc, top5_epoch_acc
       
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
        Tuple[float, float, float]: A tuple containing average validation loss, top1 accuracy, and top5 accuracy.
    """
    model.eval()
    
    top1_correct = 0
    top5_correct = 0
    running_loss = 0.0
    total = 0
    
    with torch.no_grad(): 
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct_top5 = get_num_correct_in_top5(outputs, labels)
            
            total += labels.size(0)
            top1_correct += predicted.eq(labels).sum().item()
            top5_correct += correct_top5
        
        avg_val_loss = running_loss / total
        avg_top1_val_accuracy = top1_correct / total
        avg_top5_val_accuracy = top5_correct / total
            
    return avg_val_loss, avg_top1_val_accuracy, avg_top5_val_accuracy   

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
        
        train_loss, top1_train_acc, top5_train_acc = train_epoch(model, train_dataloader, optimizer, loss_func, device, pbar)
        val_loss, top1_val_acc, top5_val_acc = validate_epoch(model, val_dataloader, loss_func, device)
        
        scheduler.step(val_loss)
        
        status_msg = f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Train Acc: {top1_train_acc:.2%} Train Top 5 Acc: {top5_train_acc:.2%} | "\
                                        f" Val Loss: {val_loss:.4f} Val Acc: {top1_val_acc:.2%} Val Top 5 Acc: {top5_val_acc:.2%} "
        pbar.maybe_log_epoch(epoch + 1, message=status_msg)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/acc": top1_train_acc,
                "train/top5_acc": top5_train_acc,  
                "val/loss": val_loss,
                "val/acc": top1_val_acc,
                "val/top5_acc": top5_val_acc,      
                "train/lr": current_lr
            })
        # Save checkpoint if model is performing better than previously
        if top1_val_acc > best_val_acc:
            best_val_acc = top1_val_acc
            save_path = "models/pokemon_cnn_best.pth"
            os.makedirs("models", exist_ok=True)
            
            checkpoint = {
                'state_dict': model.state_dict(),
                'accuracy': top1_val_acc,
                "top5_accuracy" : top5_val_acc,
                'epoch': epoch,
                'run_id': wandb.run.id if wandb.run else None  
            }
            torch.save(checkpoint, save_path)
            
        # Prune optuna trial if it is performing poorly    
        if trial is not None:
            trial.report(top1_val_acc, epoch)
            
            if trial.should_prune():
                if wandb_run is not None:
                    wandb_run.tags = wandb_run.tags + ("pruned",)
                    wandb_run.finish() 
                raise optuna.exceptions.TrialPruned()
        
    pbar.close("Training complete!\n")
    if wandb_run is not None:
        wandb_run.finish()
        
    return best_val_acc


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
    total = 0
    top1_correct = 0
    top5_correct = 0
    
    print("Starting Final Test Evaluation...")
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # Get number of exact correct predictions
        _, top1_preds = torch.max(outputs, 1)
        top1_correct += top1_preds.eq(labels).sum().item()
        
       # Expand labels from [batch_size] to [batch_size, 1] to compare against [batch_size, 5]
        # _, top5_indices = outputs.topk(5,1, largest=True, sorted=True)
        # labels_reshaped = labels.view(-1, 1).expand_as(top5_indices)
        # # See if the correct labels occurs in the top 5 predictions
        # correct_in_top5 = (top5_indices == labels_reshaped).any(dim=1).sum().item()
        
        correct_in_top5 = get_num_correct_in_top5(outputs, labels)
        top5_correct += correct_in_top5
        
        all_preds.extend(top1_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    
    
    print(f"Top-1 Accuracy: {top1_acc:.2%}")
    print(f"Top-5 Accuracy: {top5_acc:.2%}")
    
    return all_labels, all_preds



