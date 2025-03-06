# Chest Cancer Classification

## Description
This project aims to classify types of lung cancer from chest X-ray images using a Convolutional Neural Network (CNN). The model distinguishes between three types of lung cancer and normal cells:
- **Adenocarcinoma**
- **Large cell carcinoma**
- **Squamous cell carcinoma**
- **Normal cell**

## Dataset
The dataset contains chest X-ray images categorized from kaggle : Chest CT-Scan images Dataset

## Tools Used

- **PyTorch**: Main framework for developing the convolutional neural network.
- **CNN (Convolutional Neural Network)**: VGG16 model architecture.
- **DVC (Data Version Control)**: Version control for data and experiments.
- **MLflow**: Experiment and hyperparameter tracking.

## Hyperparameters Used
- **Batch size**: 32
- **Learning rate**: 0.001 with adaptive decay.
- **Optimizer**: Adam
- **Number of epochs**: 50
- **Loss function**: Categorical Cross-Entropy

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in `src/config`
6. Update the components
7. Update the pipeline
8. Update `main.py`
9. Update `dvc.yaml`

## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

##### Command
- `mlflow ui`

### Dagshub
[Dagshub](https://dagshub.com/)

### DVC Commands

1. `dvc init`
2. `dvc repro`
3. `dvc dag`

## About MLflow & DVC

### MLflow
- Production-grade tool
- Tracks all your experiments
- Logs & tags your model

### DVC
- Lightweight for POC (Proof of Concept) only
- Lightweight experiment tracker
- Can perform orchestration (creating pipelines)

