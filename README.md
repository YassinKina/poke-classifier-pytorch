# Pokémon Image Classifier: Dynamic CNN with HPO

A high-performance deep learning pipeline designed to classify the original 150 Pokémon species. This project implements a custom **DynamicCNN** architecture that allows for automated architectural searches, combined with a rigorous Hyperparameter Optimization (HPO) workflow.



[Image of a convolutional neural network architecture for image classification]


## 🌐 Live Demo
**Check out the interactive web app here:** [Your Streamlit Link Here]
*Upload your own Pokémon image or choose from a curated sample gallery to see the model's Top-5 predictions in real-time.*

## 📊 Performance Summary
* **Top-1 Accuracy:** `67.86%` (Exact species match)
* **Top-5 Accuracy:** `89.40%` (Correct species in top 5 candidates)
* **Optimization:** 20-trial study using Bayesian TPE Sampling and Median Pruning.

---

## 🛠 Features

### 1. Dynamic Architecture
The `DynamicCNN` is a flexible PyTorch implementation that adapts to configuration-driven depth and width:
- **Variable Depth:** Supports dynamic `n_layers` configuration via Hydra.
- **Adaptive Width:** Adjusts `n_filters` and `fc_size` based on Optuna suggestions.
- **Regularization:** Integrated Dropout, Batch Normalization, and Weight Decay to combat overfitting on a domain-specific dataset.

### 2. Automated HPO Workflow
Leveraging **Optuna** and **Hydra**, the training pipeline explores a multi-dimensional search space:
- **Optimizer Params:** Learning Rate ($10^{-5}$ to $10^{-3}$), Weight Decay ($10^{-6}$ to $10^{-4}$).
- **Regularization:** Adaptive Dropout rates and Label Smoothing (up to $0.2$).
- **Early Stopping:** `MedianPruner` terminates underperforming trials early to optimize compute resources.



### 3. Professional Experiment Tracking
- **Weights & Biases (W&B):** Real-time logging of training/validation loss, Top-1/Top-5 accuracy, and gradient distributions.
- **Hydra:** Version-controlled configuration management for reproducible experiments.

---

## 📁 Project Structure
```text

├── app.py              # Interactive Streamlit Web Application
├── train.py            # Main training script for single-run execution
├── hpo.py              # Optuna optimization entry point (Bayesian Search)
├── eval.py             # Script for final test-set evaluation & metrics
├── predict.py          # CLI tool for single-image inference
├── config/             # Hydra YAML configurations
│   ├── config.yaml     # Default training settings
│   └── hpo/            # Optuna-specific search space configurations
├── data/               # Pokémon dataset (Cleaned & Preprocessed)
├── models/             # Saved checkpoints (.pth files + training metadata)
├── notebooks/          # Jupyter notebooks for EDA and prototyping
├── samples/            # Curated images for Streamlit demo testing
├── src/                # Modular source code package
│   ├── __init__.py     # Makes src a Python package
│   ├── dataset.py      # Custom PyTorch Dataset class
│   ├── data_setup.py   # DataLoaders and preprocessing pipelines
│   ├── model.py        # DynamicCNN architecture definition
│   ├── engine.py       # Core Train/Val/Top-K Evaluation loops
│   └── utils.py        # Logging, W&B setup, and stat calculations
└── requirements.txt    # Project dependencies
```

## 🚀 Getting Started

### 1. Installation
 ```pip install requirements.txt```

### 2. Run Hyperparameter Optimization
Launches a new study (of 20 trials) with Bayesian search
 ```python hpo.py``

### 3. Run Final Evaluation
Load the best weights from the ```models/ ```directory and evaluate on the hold-out test set:
``` python eval.py ```

### 4. Single Image Inference


## 🧪  Data Normalization
This project uses custom-calculated channel-wise statistics to account for the unique color distribution of Pokémon art 
(higher brightness and saturation compared to natural images) rather than standard ImageNet defaults:
* Mean: [0.5863, 0.5675, 0.5337]
* Std: [0.3464, 0.3312, 0.3421]


## Data Limitaions
Very few training data were pictures of pokemon cards. As a result, the model struggles to correctly classify
the input when given a pokemon card image.


