import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion
    """
    raw_data_path: str = os.path.join('artifacts', 'raw', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'processed', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'processed', 'test.csv')
    user_profiles_path: str = os.path.join('artifacts', 'processed', 'user_profiles.csv')


class DataIngestion:
    """
    Class for handling data ingestion from source
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self, data_source='fashion_size_fit_dataset.csv'):
        """
        Ingest data from source and split into train/test
        """
        logging.info("Starting data ingestion process")
        
        try:
            # Read data
            logging.info(f"Reading data from: {data_source}")
            df = pd.read_csv(data_source)
            
            logging.info(f"Data loaded successfully. Shape: {df.shape}")
            logging.info(f"Columns: {df.columns.tolist()}")
            
            # Create directories
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")
            
            # Create user profiles
            logging.info("Creating user profiles...")
            user_profiles = df.groupby('user_id').agg({
                'age': 'first',
                'gender': 'first',
                'height_cm': 'first',
                'weight_kg': 'first',
                'body_shape': 'first',
                'chest_bust_cm': 'first',
                'waist_cm': 'first',
                'hip_cm': 'first',
                'purchased_size': lambda x: x.mode()[0] if not x.mode().empty else 'M',
                'fit_feedback': lambda x: x.mode()[0] if not x.mode().empty else 'Good Fit',
                'category': lambda x: x.mode()[0] if not x.mode().empty else 'Tops',
                'brand': lambda x: x.mode()[0] if not x.mode().empty else 'Nike',
            }).reset_index()
            
            user_profiles.to_csv(self.ingestion_config.user_profiles_path, index=False)
            logging.info(f"User profiles saved: {len(user_profiles)} users")
            
            # Train-test split
            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df['fit_feedback']
            )
            
            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(f"Train data saved at: {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved at: {self.ingestion_config.test_data_path}")
            logging.info(f"Train shape: {train_set.shape}, Test shape: {test_set.shape}")
            
            logging.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.user_profiles_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test data ingestion
    obj = DataIngestion()
    train_path, test_path, profiles_path = obj.initiate_data_ingestion()
    
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")
    print(f"User profiles: {profiles_path}")