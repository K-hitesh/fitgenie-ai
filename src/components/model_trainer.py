import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """
    Configuration for model training
    """
    trained_model_path: str = os.path.join('artifacts', 'models', 'model_final.pkl')
    catboost_model_path: str = os.path.join('artifacts', 'models', 'catboost_model_final.pkl')
    xgboost_model_path: str = os.path.join('artifacts', 'models', 'xgboost_model_final.pkl')
    rf_model_path: str = os.path.join('artifacts', 'models', 'rf_model_final.pkl')
    stacking_meta_path: str = os.path.join('artifacts', 'models', 'stacking_meta_model_final.pkl')


class ModelTrainer:
    """
    Class for training machine learning models
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, X_train, X_test, y_train, y_test, label_encoder):
        """
        Train multiple models and create stacking ensemble
        """
        try:
            logging.info("Starting model training")
            
            # Define class weights for imbalanced data
            class_weights = {0: 1.0, 1: 3.5, 2: 1.5, 3: 4.0, 4: 5.0}
            
            # Initialize models
            models = {
                'CatBoost': CatBoostClassifier(
                    iterations=700,
                    learning_rate=0.1,
                    depth=6,
                    l2_leaf_reg=3,
                    class_weights=class_weights,
                    eval_metric='TotalF1',
                    random_seed=42,
                    verbose=100,
                    early_stopping_rounds=80
                ),
                'XGBoost': XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    random_state=42,
                    tree_method='hist',
                    eval_metric='mlogloss',
                    early_stopping_rounds=50
                ),
                'RandomForest': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
            }
            
            # Train individual models
            logging.info("Training individual models...")
            
            # CatBoost
            logging.info("Training CatBoost...")
            models['CatBoost'].fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                use_best_model=True
            )
            save_object(self.model_trainer_config.catboost_model_path, models['CatBoost'])
            
            # XGBoost
            logging.info("Training XGBoost...")
            models['XGBoost'].fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            save_object(self.model_trainer_config.xgboost_model_path, models['XGBoost'])
            
            # Random Forest
            logging.info("Training Random Forest...")
            models['RandomForest'].fit(X_train, y_train)
            save_object(self.model_trainer_config.rf_model_path, models['RandomForest'])
            
            # Evaluate individual models
            logging.info("Evaluating individual models...")
            
            model_scores = {}
            for model_name, model in models.items():
                y_pred = model.predict(X_test)
                y_pred_labels = label_encoder.inverse_transform(y_pred.astype(int))
                y_test_labels = label_encoder.inverse_transform(y_test)
                
                acc = accuracy_score(y_test_labels, y_pred_labels)
                f1 = f1_score(y_test_labels, y_pred_labels, average='macro')
                precision = precision_score(y_test_labels, y_pred_labels, average='macro')
                recall = recall_score(y_test_labels, y_pred_labels, average='macro')
                
                model_scores[model_name] = {
                    'accuracy': acc,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall
                }
                
                logging.info(f"{model_name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
            
            # Create stacking ensemble
            logging.info("Creating stacking ensemble...")
            
            X_meta_train = np.hstack([
                models['CatBoost'].predict_proba(X_train),
                models['XGBoost'].predict_proba(X_train),
                models['RandomForest'].predict_proba(X_train)
            ])
            
            X_meta_test = np.hstack([
                models['CatBoost'].predict_proba(X_test),
                models['XGBoost'].predict_proba(X_test),
                models['RandomForest'].predict_proba(X_test)
            ])
            
            # Meta model
            stacking_meta = LogisticRegression(
                solver='lbfgs',
                multi_class='multinomial',
                max_iter=500,
                random_state=42,
                n_jobs=-1
            )
            
            stacking_meta.fit(X_meta_train, y_train)
            
            # Evaluate stacking
            y_pred_stack = stacking_meta.predict(X_meta_test)
            y_pred_stack_labels = label_encoder.inverse_transform(y_pred_stack.astype(int))
            y_test_labels = label_encoder.inverse_transform(y_test)
            
            acc_stack = accuracy_score(y_test_labels, y_pred_stack_labels)
            f1_stack = f1_score(y_test_labels, y_pred_stack_labels, average='macro')
            precision_stack = precision_score(y_test_labels, y_pred_stack_labels, average='macro')
            recall_stack = recall_score(y_test_labels, y_pred_stack_labels, average='macro')
            
            model_scores['StackingEnsemble'] = {
                'accuracy': acc_stack,
                'f1_score': f1_stack,
                'precision': precision_stack,
                'recall': recall_stack
            }
            
            logging.info(f"Stacking - Accuracy: {acc_stack:.4f}, F1: {f1_stack:.4f}")
            
            # Save stacking meta model
            save_object(self.model_trainer_config.stacking_meta_path, stacking_meta)
            
            # Find best model
            best_model_name = max(model_scores, key=lambda x: model_scores[x]['f1_score'])
            best_score = model_scores[best_model_name]
            
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best F1 Score: {best_score['f1_score']:.4f}")
            
            # Save best model as main model
            if best_model_name == 'StackingEnsemble':
                # Save a package with all models for stacking
                best_model = {
                    'catboost': models['CatBoost'],
                    'xgboost': models['XGBoost'],
                    'random_forest': models['RandomForest'],
                    'meta_model': stacking_meta
                }
            else:
                best_model = models[best_model_name]
            
            save_object(self.model_trainer_config.trained_model_path, best_model)
            
            logging.info("Model training completed successfully")
            
            return (
                self.model_trainer_config.trained_model_path,
                best_model_name,
                model_scores
            )
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test model trainer
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    from src.utils import load_object
    
    ingestion = DataIngestion()
    train_path, test_path, _ = ingestion.initiate_data_ingestion()
    
    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, _, label_encoder_path = transformation.initiate_data_transformation(train_path, test_path)
    
    label_encoder = load_object(label_encoder_path)
    
    trainer = ModelTrainer()
    model_path, best_model_name, scores = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test, label_encoder)
    
    print(f"Best Model: {best_model_name}")
    print(f"Model saved at: {model_path}")
