from dataclasses import dataclass
from pathlib import Path

#output de 'get_data_ingestion_config()' lié à config.yaml donnant l'accès au téléchargement du dataset
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir : Path
    trained_model_path : Path
    updated_base_model_path: Path
    training_data : Path
    params_is_augmentation : bool
    params_image_size : list
    params_batch_size : int
    params_epochs : int

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int