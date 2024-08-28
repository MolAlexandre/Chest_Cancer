from Chest_Cancer_Classifier.components.evalutation import Evaluation
from Chest_Cancer_Classifier.config.configuration import ConfigurationManager
from Chest_Cancer_Classifier import logger

STAGE_NAME = "Model evaluation"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()



if __name__=='__main__':
    try:
        logger.info(f">>>>>>> stage '{STAGE_NAME}' started <<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>> stage '{STAGE_NAME}' completed <<<<<<<\n\nx===========x")
    except Exception as e :
        raise e
