from Chest_Cancer_Classifier import logger
from Chest_Cancer_Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Chest_Cancer_Classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from Chest_Cancer_Classifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from Chest_Cancer_Classifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=====================x")
    
except Exception as e :
    raise e


STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f">>>>>>> stage '{STAGE_NAME}' started <<<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage '{STAGE_NAME}' completed <<<<<<<\n\nx=====================x")
except Exception as e :
    raise e

STAGE_NAME = "Training"

try:
    logger.info(f">>>>>>> stage '{STAGE_NAME}' started <<<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage '{STAGE_NAME}' completed <<<<<<<\n\nx=====================x")
except Exception as e :
    raise e

STAGE_NAME = "Evaluation "

try:
    logger.info(f">>>>>>> stage '{STAGE_NAME}' started <<<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>>> stage '{STAGE_NAME}' completed <<<<<<<\n\nx=====================x")
except Exception as e :
    raise e
