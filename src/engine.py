from src.utils import NestedProgressBar, get_best_val_accuracy
import torch
import os
import wandb

def train_epoch(model, train_dataloader, optimizer, loss_func, device, pbar):
    """Trains the model for a single epoch.

    This function iterates over the training dataloader, performs the forward
    and backward passes, updates the model weights, and calculates the loss
    and accuracy for the entire epoch.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_dataloader (DataLoader): The DataLoader containing the training data.
        optimizer (optim.Optimizer): The optimizer for updating model weights.
        loss_fcn: The loss function used for training.
        device: The device to perform training on.
        pbar: A progress bar handler object to visualize training progress.

    Returns:
        tuple: A tuple containing the average loss and accuracy for the epoch.
    """
    # Training mode
    model.train()
    
    # Initialize metrics
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Loop through training data
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        # Update the batch progress bar
        pbar.update_batch(batch_idx + 1)
        # Move inputs and labels to device
        inputs, labels = inputs.to(device), labels.to(device)
       
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #Get total loss for each epoch
        running_loss += loss.item() * inputs.size(0)
        # Get predicted pokemon with highest score
        _, predicted_class = outputs.max(1)
        # Update total number of samples
        total += labels.size(0)
        # Update total humber of correctlz classifed pokemon
        correct += predicted_class.eq(labels).sum().item()
        
    # Average loss for the epoch
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc
       

def validate_epoch(model, val_dataloader, loss_func, device):
    model.eval() # Set model to evaluation mode
    
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
            
    return running_loss / total, correct / total            

def train_model(model, optimizer, loss_func, train_dataloader,val_dataloader, device, n_epochs, wandb_run=None):
    """Runs the training process for the model over multiple epochs.

    This function sets up a progress bar and manages the training loop,
    calling a helper function to handle the logic for each individual epoch.
    It also logs progress periodically.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        optimizer (optim.Optimizer): The optimizer for updating model weights.
        loss_func: The loss function used for training.
        train_dataloader (DataLoader): The DataLoader for the training data.
        device: The device to perform training on.
        n_epochs (int): The total number of epochs to train for.
    """
    # Initialize progress bar to track training
    pbar = NestedProgressBar(
        total_epochs=n_epochs,
        total_batches=len(train_dataloader),
        epoch_message_freq=1,
        mode="train",
        use_notebook=False
    )
    # Later update function logic
    best_val_acc = get_best_val_accuracy()
    
    # Loop through all epochs
    for epoch in range(n_epochs):
        pbar.update_epoch(epoch + 1)
        
        # 1. Train
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, loss_func, device, pbar)
        
        # 2. Validate
        val_loss, val_acc = validate_epoch(model, val_dataloader, loss_func, device)
        
        
        # 3. Log results
        status_msg = f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}"
        pbar.maybe_log_epoch(epoch + 1, message=status_msg)
        
        # 4. Log to wandb
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                # "val/acc_top5": val_acc5
            })
        
        # 4. Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"models/pokemon_cnn_best.pth"
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print("New best model saved successfully")
        
        
    # Close the progress bar and print a final completion message
    pbar.close("Training complete!\n")
    
    if wandb_run is not None:
        wandb_run.finish()
    
def init_wandb_run(learning_rate, architecture, dataset, num_epochs):
        # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="yassinbkina",
        # Set the wandb project where this run will be logged.
        project="pokemon-classification",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": learning_rate,
            "architecture": architecture,
            "dataset": dataset,
            "epochs": num_epochs,
        },
    )
    return run