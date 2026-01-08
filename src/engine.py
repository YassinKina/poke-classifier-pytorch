from src.utils import NestedProgressBar

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
       
            
def train_model(model, optimizer, loss_func, train_dataloader, device, n_epochs):
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
        epoch_message_freq=5,
        mode="train",
        use_notebook=False
    )
    
    # Loop through all epochs
    for epoch in range(n_epochs):
        #Update the outer progress bar for the current epoch
        pbar.update_epoch(epoch + 1)
        # Train model for one epoch
        train_loss, _ = train_epoch(
            model, train_dataloader, optimizer, loss_func, device, pbar
        )
        # Log the training loss for the current epoch at a set frequency
        pbar.maybe_log_epoch(
            epoch + 1,
            message=f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}",
        )
    # Close the progress bar and print a final completion message
    pbar.close("Training complete!\n")