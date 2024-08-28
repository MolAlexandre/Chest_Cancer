from dataclasses import dataclass
from pathlib import Path

#output de 'get_data_ingestion_config()' lié à config.yaml donnant l'accès au téléchargement du dataset
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path