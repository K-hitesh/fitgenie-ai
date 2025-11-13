import os
import sys
import pandas as pd
import numpy as np
import joblib
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to a file using joblib
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        joblib.dump(obj, file_path)
        logging.info(f"Object saved successfully at: {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a file using joblib
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        
        obj = joblib.load(file_path)
        logging.info(f"Object loaded successfully from: {file_path}")
        return obj
        
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models and return their performance metrics
    """
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        report = {}
        
        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            
            report[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            }
            
            logging.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return report
        
    except Exception as e:
        raise CustomException(e, sys)


def calculate_body_features(data):
    """
    Calculate body measurement features
    """
    try:
        bmi = data['weight_kg'] / ((data['height_cm']/100) ** 2)
        chest_waist_ratio = data['chest_bust_cm'] / data['waist_cm']
        waist_hip_ratio = data['waist_cm'] / data['hip_cm']
        chest_hip_ratio = data['chest_bust_cm'] / data['hip_cm']
        height_weight_ratio = data['height_cm'] / data['weight_kg']
        
        return {
            'bmi': bmi,
            'chest_waist_ratio': chest_waist_ratio,
            'waist_hip_ratio': waist_hip_ratio,
            'chest_hip_ratio': chest_hip_ratio,
            'height_weight_ratio': height_weight_ratio
        }
    except Exception as e:
        raise CustomException(e, sys)