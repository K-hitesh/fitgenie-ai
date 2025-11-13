import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class TrainPipeline:
    """
    Complete training pipeline
    """
    def __init__(self):
        pass
    
    def start_training(self, data_source='fashion_size_fit_dataset.csv'):
        """
        Execute complete training pipeline
        """
        try:
            logging.info("="*80)
            logging.info("STARTING TRAINING PIPELINE")
            logging.info("="*80)
            
            # Step 1: Data Ingestion
            logging.info("STEP 1: Data Ingestion")
            ingestion = DataIngestion()
            train_path, test_path, profiles_path = ingestion.initiate_data_ingestion(data_source)
            
            # Step 2: Data Transformation
            logging.info("STEP 2: Data Transformation")
            transformation = DataTransformation()
            X_train, X_test, y_train, y_test, preprocessor_path, label_encoder_path = transformation.initiate_data_transformation(
                train_path, test_path
            )
            
            # Step 3: Model Training
            logging.info("STEP 3: Model Training")
            label_encoder = load_object(label_encoder_path)
            
            trainer = ModelTrainer()
            model_path, best_model_name, model_scores = trainer.initiate_model_trainer(
                X_train, X_test, y_train, y_test, label_encoder
            )
            
            logging.info("="*80)
            logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("="*80)
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Model Location: {model_path}")
            logging.info("\nModel Performance:")
            for model_name, scores in model_scores.items():
                logging.info(f"{model_name}:")
                logging.info(f"  Accuracy: {scores['accuracy']:.4f}")
                logging.info(f"  F1 Score: {scores['f1_score']:.4f}")
                logging.info(f"  Precision: {scores['precision']:.4f}")
                logging.info(f"  Recall: {scores['recall']:.4f}")
            
            return {
                'model_path': model_path,
                'best_model': best_model_name,
                'scores': model_scores,
                'preprocessor_path': preprocessor_path,
                'label_encoder_path': label_encoder_path,
                'profiles_path': profiles_path
            }
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    result = pipeline.start_training()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Model: {result['best_model']}")
    print(f"Model Path: {result['model_path']}")