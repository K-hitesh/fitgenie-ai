import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation
    """
    preprocessor_path: str = os.path.join('artifacts', 'models', 'preprocessor.pkl')
    label_encoder_path: str = os.path.join('artifacts', 'models', 'label_encoder.pkl')
    size_mapping_path: str = os.path.join('artifacts', 'models', 'size_mapping.pkl')
    group_averages_path: str = os.path.join('artifacts', 'models', 'group_averages.pkl')
    brand_equivalency_path: str = os.path.join('artifacts', 'models', 'brand_equivalency.pkl')


class DataTransformation:
    """
    Class for handling data transformation and feature engineering
    """
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def standardize_size(self, size_str):
        """Standardize size values"""
        if pd.isna(size_str):
            return "Unknown"
        s = str(size_str).strip().upper()
        if s in ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL', '4XL']:
            return s
        return "Other"
    
    def create_features(self, df):
        """
        Create all engineered features
        """
        try:
            logging.info("Starting feature engineering")
            
            # Time features
            df['purchase_datetime'] = pd.to_datetime(df['purchase_datetime'])
            df['purchase_year'] = df['purchase_datetime'].dt.year
            df['purchase_month'] = df['purchase_datetime'].dt.month
            df['purchase_hour'] = df['purchase_datetime'].dt.hour
            df['day_of_week'] = df['purchase_datetime'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['season'] = df['purchase_month'] % 12 // 3 + 1
            
            # Sort by user and time
            df = df.sort_values(['user_id', 'purchase_datetime'])
            
            # Temporal decay
            df['prev_purchase_datetime'] = df.groupby('user_id')['purchase_datetime'].shift(1)
            df['days_since_last_purchase'] = (df['purchase_datetime'] - df['prev_purchase_datetime']).dt.days.fillna(365)
            
            # Body measurements
            df['bmi'] = df['weight_kg'] / ((df['height_cm']/100) ** 2)
            df['chest_waist_ratio'] = df['chest_bust_cm'] / df['waist_cm']
            df['waist_hip_ratio'] = df['waist_cm'] / df['hip_cm']
            df['chest_hip_ratio'] = df['chest_bust_cm'] / df['hip_cm']
            df['height_weight_ratio'] = df['height_cm'] / df['weight_kg']
            
            df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], 
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            
            # Price features
            df['final_price'] = df['base_price'] * (1 - df['discount_pct'] / 100)
            df['is_discounted'] = (df['discount_pct'] > 0).astype(int)
            
            # User behavior
            df['user_purchase_count'] = df.groupby('user_id').cumcount() + 1
            df['user_total_purchases'] = df.groupby('user_id')['user_id'].transform('count')
            df['user_return_rate'] = df.groupby('user_id')['returned'].transform('mean')
            df['user_avg_price'] = df.groupby('user_id')['final_price'].transform('mean')
            
            df['prev_purchased_size'] = df.groupby('user_id')['purchased_size'].shift(1).fillna('No History')
            df['prev_fit_feedback'] = df.groupby('user_id')['fit_feedback'].shift(1).fillna('No History')
            
            # Brand/Category features
            df['brand_return_rate'] = df.groupby('brand')['returned'].transform('mean')
            df['category_return_rate'] = df.groupby('category')['returned'].transform('mean')
            
            # Size standardization
            df['size_standardized'] = df['purchased_size'].apply(self.standardize_size)
            df['gender_category'] = df['gender'].astype(str) + '_' + df['category'].astype(str)
            
            logging.info(f"Feature engineering completed. Total columns: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def build_size_mapping(self, df):
        """Build brand-category size mapping"""
        try:
            logging.info("Building size mapping")
            
            size_mapping = {}
            
            for category in df['category'].unique():
                cat_data = df[df['category'] == category]
                size_map = {}
                
                for brand in cat_data['brand'].unique():
                    brand_data = cat_data[cat_data['brand'] == brand]
                    sizes = brand_data['purchased_size'].value_counts().index.tolist()
                    
                    standard = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL']
                    ordered = [s for s in standard if s.upper() in [str(x).upper() for x in sizes]]
                    numeric = sorted([s for s in sizes if str(s).isdigit()], key=lambda x: int(x))
                    other = [s for s in sizes if s not in ordered and not str(s).isdigit()]
                    
                    size_map[brand] = ordered + numeric + other
                
                size_mapping[category] = size_map
            
            logging.info(f"Size mapping created: {len(size_mapping)} categories")
            
            return size_mapping
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def build_brand_equivalency(self, df):
        """Build brand size equivalency matrix"""
        try:
            logging.info("Building brand equivalency matrix")
            
            df_sorted = df.sort_values(['user_id', 'purchase_datetime'])
            brand_equivalency = {}
            
            for user_id, user_data in df_sorted.groupby('user_id'):
                user_data = user_data.reset_index(drop=True)
                
                for i in range(1, len(user_data)):
                    prev_row = user_data.iloc[i-1]
                    curr_row = user_data.iloc[i]
                    
                    if (prev_row['brand'] != curr_row['brand'] and 
                        prev_row['category'] == curr_row['category'] and
                        prev_row['fit_feedback'] == 'Good Fit'):
                        
                        key = (prev_row['brand'], curr_row['brand'], curr_row['category'])
                        prev_size = prev_row['purchased_size']
                        curr_size = curr_row['purchased_size']
                        
                        if key not in brand_equivalency:
                            brand_equivalency[key] = {}
                        if prev_size not in brand_equivalency[key]:
                            brand_equivalency[key][prev_size] = []
                        
                        brand_equivalency[key][prev_size].append(curr_size)
            
            # Convert to most common size
            for key in brand_equivalency:
                for size in brand_equivalency[key]:
                    sizes = brand_equivalency[key][size]
                    brand_equivalency[key][size] = max(set(sizes), key=sizes.count)
            
            logging.info(f"Brand equivalency matrix created: {len(brand_equivalency)} mappings")
            
            return brand_equivalency
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_delta_features(self, df, group_averages):
        """Create delta features based on group averages"""
        try:
            group_cols = ['brand', 'category']
            size_metrics = ['height_cm', 'chest_bust_cm', 'waist_cm', 'hip_cm']
            
            df = df.copy().reset_index(drop=True)
            df = pd.merge(df, group_averages, on=group_cols, how='left')
            
            for metric in size_metrics:
                avg_col = f'avg_{metric}'
                delta_col = f'size_delta_{metric}'
                global_mean = df[metric].mean()
                df[avg_col].fillna(global_mean, inplace=True)
                df[delta_col] = df[metric] - df[avg_col]
                df.drop(columns=[avg_col], inplace=True, errors='ignore')
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_data_transformer_object(self, numerical_features, categorical_features):
        """
        Create preprocessing pipeline
        """
        try:
            logging.info("Creating data transformer pipeline")
            
            numerical_transformer = Pipeline([
                ('scaler', RobustScaler())
            ])
            
            categorical_transformer = Pipeline([
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            preprocessor = ColumnTransformer([
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
            logging.info("Data transformer pipeline created")
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Main data transformation method
        """
        try:
            logging.info("Starting data transformation")
            
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f"Train data: {train_df.shape}, Test data: {test_df.shape}")
            
            # Feature engineering
            train_df = self.create_features(train_df)
            test_df = self.create_features(test_df)
            
            # Build mappings
            size_mapping = self.build_size_mapping(train_df)
            brand_equivalency = self.build_brand_equivalency(train_df)
            
            # Save mappings
            save_object(self.transformation_config.size_mapping_path, size_mapping)
            save_object(self.transformation_config.brand_equivalency_path, brand_equivalency)
            
            # Prepare features
            features_to_drop = [
                'order_id', 'user_id', 'purchase_datetime', 'fit_feedback', 'fit_score',
                'returned', 'return_reason', 'recommended_size', 'purchased_size',
                'size_standardized', 'size_system', 'prev_purchase_datetime', 'base_price'
            ]
            
            X_train = train_df.drop(columns=[col for col in features_to_drop if col in train_df.columns])
            y_train = train_df['fit_feedback']
            
            X_test = test_df.drop(columns=[col for col in features_to_drop if col in test_df.columns])
            y_test = test_df['fit_feedback']
            
            # Fill missing values
            for col in X_train.select_dtypes(include=np.number).columns:
                X_train[col] = X_train[col].fillna(X_train[col].median())
                X_test[col] = X_test[col].fillna(X_test[col].median())
            
            # Delta features
            group_cols = ['brand', 'category']
            size_metrics = ['height_cm', 'chest_bust_cm', 'waist_cm', 'hip_cm']
            
            group_averages = X_train.groupby(group_cols)[size_metrics].mean().reset_index()
            group_averages.columns = group_cols + [f'avg_{col}' for col in size_metrics]
            
            X_train = self.create_delta_features(X_train, group_averages)
            X_test = self.create_delta_features(X_test, group_averages)
            
            save_object(self.transformation_config.group_averages_path, group_averages)
            
            # Get feature lists
            numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X_train.select_dtypes(include='object').columns.tolist()
            
            logging.info(f"Numerical features: {len(numerical_features)}, Categorical: {len(categorical_features)}")
            
            # Get preprocessor
            preprocessor = self.get_data_transformer_object(numerical_features, categorical_features)
            
            # Transform data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Label encoding
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            
            # SMOTE
            logging.info("Applying SMOTE for class balancing")
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train_encoded)
            
            logging.info(f"After SMOTE: {X_train_balanced.shape}")
            
            # Save preprocessor and label encoder
            save_object(self.transformation_config.preprocessor_path, preprocessor)
            save_object(self.transformation_config.label_encoder_path, label_encoder)
            
            # Save column names for later use
            save_object(os.path.join('artifacts', 'models', 'feature_names.pkl'), X_train.columns.tolist())
            
            logging.info("Data transformation completed successfully")
            
            return (
                X_train_balanced,
                X_test_processed,
                y_train_balanced,
                y_test_encoded,
                self.transformation_config.preprocessor_path,
                self.transformation_config.label_encoder_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test data transformation
    from src.components.data_ingestion import DataIngestion
    
    ingestion = DataIngestion()
    train_path, test_path, _ = ingestion.initiate_data_ingestion()
    
    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, _, _ = transformation.initiate_data_transformation(train_path, test_path)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")