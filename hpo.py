import optuna
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from src import train_model, init_wandb_run
from src import create_dataloaders
from src import DynamicCNN
from src import set_seed

def objective(trial: optuna.trial.Trial, cfg: DictConfig):
    """
    Evaluates a specific hyperparameter configuration suggested by Optuna.

    This function represents a single trial in a Hyperparameter Optimization (HPO) 
    study. It performs the following steps:
    1. Samples hyperparameters from defined ranges (Search Space).
    2. Overrides the base configuration with trial-specific values.
    3. Re-initializes DataLoaders, the DynamicCNN, and the optimizer for the trial.
    4. Orchestrates the training process and logs performance to Weights & Biases.
    5. Returns the final validation accuracy as the objective metric for Optuna to maximize.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object used to suggest parameters.
        cfg (DictConfig): The master Hydra configuration containing base settings 
            and HPO search space definitions.

    Returns:
        float: The best validation accuracy achieved during the trial.

    Raises:
        optuna.exceptions.TrialPruned: If the trial is stopped early by an Optuna pruner.
    """
    
    DATA_PATH = "./data/pokemon_clean"
    run_name = f"trial_{trial.number}"
    
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", cfg.hpo.lr_range[0], cfg.hpo.lr_range[1], log=True)
    dropout = trial.suggest_float("dropout_rate", cfg.hpo.dropout_range[0], cfg.hpo.dropout_range[1])
    n_layers = trial.suggest_int("n_layers", cfg.hpo.layers_range[0], cfg.hpo.layers_range[1])
    batch_size = trial.suggest_categorical("batch_size", cfg.hpo.batch_options)
    fc_size = trial.suggest_categorical("fc_size", cfg.hpo.fc_options)
    
    # Filters and n_layers must be equal in len
    base_filters = list(cfg.model.n_filters) # This is [32, 64, 128, 256]
    # if n_layers is 2, active_filters becomes [32, 64]
    active_filters = base_filters[:n_layers]
    
    # 2. Update a local copy of the config
    trial_cfg = OmegaConf.to_container(cfg, resolve=True) # convert DictConfig to dict
    trial_cfg["training"]["lr"] = lr
    trial_cfg["training"]["batch_size"] = batch_size
    trial_cfg["model"]["n_layers"] = n_layers
    trial_cfg["model"]["dropout_rate"] = dropout
    trial_cfg["model"]["fc_size"] = fc_size
    trial_cfg = OmegaConf.create(trial_cfg)

    # 3. Setup objects for THIS trial
    train_loader, val_loader, _ = create_dataloaders(clean_data_path=DATA_PATH, batch_size=trial_cfg.training.batch_size)
    
    model = DynamicCNN(
        n_layers=trial_cfg.model.n_layers,
        n_filters=active_filters,
        dropout_rate=trial_cfg.model.dropout_rate,
        num_classes=trial_cfg.training.num_classes,
        kernel_sizes=trial_cfg.model.kernel_sizes,
        fc_size=trial_cfg.model.fc_size
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    loss_func = torch.nn.CrossEntropyLoss()

    # 4. Initialize W&B for this specific trial
    run = init_wandb_run(trial_cfg, run_name=run_name)
    run.notes = f"Optuna Trial {trial.number}"

    # 5. Call training function
    accuracy = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=cfg.device,
        n_epochs=trial_cfg.training.epochs,
        wandb_run=run,
        trial=trial
    )
    
    return accuracy

@hydra.main(version_base=None, config_path="config", config_name="config")
def run_hpo(cfg: DictConfig):
    """
    Executes a Hyperparameter Optimization (HPO) study using Optuna and Hydra.

    This function initializes an Optuna study focused on maximizing validation accuracy. 
    It implements a MedianPruner to terminate underperforming trials early, based 
    on intermediate results compared to previous trials. The search space and 
    logic for each trial are defined in the objective function, which is optimized 
    over a set number of iterations.

    Args:
        cfg (DictConfig): The global Hydra configuration containing the HPO search 
            ranges, model defaults, and device settings.

    Returns:
        None: The best trial parameters and accuracy value are printed to the console 
            upon completion.
    """
    # Set random seed for reproducibilty 
    set_seed()
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
        n_startup_trials=3,  # Don't prune the first 3 trials to establish baseline
        n_warmup_steps=3,    # Don't prune any trial before epoch 3
        interval_steps=1     # Check for pruning every epoch
    ))
    # Pass our Hydra cfg into the objective
    study.optimize(lambda trial: objective(trial, cfg), n_trials=20)
    
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

if __name__ == "__main__":
    run_hpo()