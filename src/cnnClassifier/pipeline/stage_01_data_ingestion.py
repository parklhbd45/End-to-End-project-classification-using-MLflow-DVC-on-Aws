from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == '__main__':
    logger.info(f"\n{'=' * 50}")
    logger.info(f"{'=' * 15} {STAGE_NAME} Pipeline {'=' * 15}")
    logger.info(f"{'=' * 50}\n")
    
    try:
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        
        logger.info(f"\n{'=' * 50}")
        logger.info(f"{'=' * 10} {STAGE_NAME} Pipeline Completed Successfully {'=' * 10}")
        logger.info(f"{'=' * 50}\n")
    
    except Exception as e:
        logger.exception(f"\nAn unexpected error occurred in the {STAGE_NAME} pipeline:")
        
        logger.error(f"\n{'=' * 50}")
        logger.error(f"{'=' * 15} {STAGE_NAME} Pipeline Failed {'=' * 15}")
        logger.error(f"{'=' * 50}\n")
        
        raise e