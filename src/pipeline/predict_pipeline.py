import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from src.logger import logging
from src.exception import CustomException
import sys
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

class PredictPipeline:
    def __init__(self):
        try:
            logging.info("="*60)
            logging.info("🚀 INITIALIZING PREDICTION PIPELINE")
            logging.info("="*60)
            
            # ========== LOAD MODELS ==========
            logging.info("📦 Loading models...")
            self.catboost_model = joblib.load('artifacts/models/catboost_model_final.pkl')
            logging.info(" 	✅ CatBoost loaded")
            
            self.xgb_model = joblib.load('artifacts/models/xgboost_model_final.pkl')
            logging.info(" 	✅ XGBoost loaded")
            
            # DISABLED FOR DEPLOYMENT - Random Forest has scikit-learn compatibility issues
            # self.rf_model = joblib.load('artifacts/models/rf_model_final.pkl')
            # logging.info(" 	✅ Random Forest loaded")
            self.rf_model = None  # Temporarily disabled
            logging.info(" 	⚠️  Random Forest disabled for deployment compatibility")
            
            self.stacking_meta_model = joblib.load('artifacts/models/stacking_meta_model_final.pkl')
            logging.info(" 	✅ Stacking Meta loaded")
            
            self.preprocessor = joblib.load('artifacts/models/preprocessor_final.pkl')
            logging.info(" 	✅ Preprocessor loaded")
            
            # FIX: Handle label encoder with better error handling
            try:
                self.label_encoder = joblib.load('artifacts/models/label_encoder_final.pkl')
                logging.info(" 	✅ Label Encoder loaded")
            except Exception as e:
                logging.warning(f" 	⚠️ Label encoder failed, creating new: {str(e)}")
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np.array(['Perfect Fit', 'Too Small', 'Too Large', 
                                                        'Slightly Small', 'Slightly Large'])
                logging.info(" 	✅ Label Encoder created")
            
            # FIX: Handle size mapping with better error handling
            try:
                self.size_mapping = joblib.load('artifacts/models/size_mapping_final.pkl')
                logging.info(" 	✅ Size Mapping loaded")
            except Exception as e:
                logging.warning(f" 	⚠️ Size mapping failed, using default")
                self.size_mapping = {
                    'XXS': 0, 'XS': 1, 'S': 2, 'M': 3, 'L': 4, 'XL': 5, 'XXL': 6, '3XL': 7
                }
                logging.info(" 	✅ Size Mapping created")
            
            # ========== LOAD DATA (user_profiles needed for historical lookup) ==========
            raw_data_path = 'artifacts/raw/fashion_size_fit_dataset.csv'
            
            if os.path.exists(raw_data_path):
                self.user_profiles = pd.read_csv(raw_data_path)
            else:
                self.user_profiles = pd.read_csv('artifacts/processed/user_profiles_final.csv')
            
            # Clean column names
            self.user_profiles.columns = [
                col.lower().strip().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                for col in self.user_profiles.columns
            ]
            
            # Column mapping (Crucial for linking data to user feedback columns in the image)
            column_mapping = {
                'userid': 'user_id', 'user': 'user_id', 'id': 'user_id',
                'height': 'height_cm', 'weight': 'weight_kg',
                'chest': 'chest_bust_cm', 'bust': 'chest_bust_cm',
                'waist': 'waist_cm', 'hip': 'hip_cm',
                'size': 'purchased_size', 'size_standardized': 'purchased_size',
                'body_type': 'body_shape',
                'purchased': 'purchased_size', 
                'fit_feedback': 'fit_feedback' 
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in self.user_profiles.columns and new_col not in self.user_profiles.columns:
                    self.user_profiles.rename(columns={old_col: new_col}, inplace=True)
            
            # Ensure required columns
            required_defaults = {
                'user_id': 'U00000', 'age': 30, 'gender': 'Female',
                'height_cm': 165, 'weight_kg': 60, 'chest_bust_cm': 88,
                'waist_cm': 70, 'hip_cm': 92, 'body_shape': 'Average',
                'brand': 'Unknown', 'category': 'Tops',
                'purchased_size': 'M', 'user_total_purchases': 1,
                'fit_feedback': 'Unknown'
            }
            
            for col, default_val in required_defaults.items():
                if col not in self.user_profiles.columns:
                    self.user_profiles[col] = default_val
            
            # Convert numeric
            numeric_cols = ['age', 'height_cm', 'weight_kg', 'bmi', 'chest_bust_cm', 
                            'waist_cm', 'hip_cm', 'user_total_purchases']
            
            for col in numeric_cols:
                if col in self.user_profiles.columns:
                    self.user_profiles[col] = pd.to_numeric(self.user_profiles[col], errors='coerce')
            
            # Calculate BMI
            if 'bmi' not in self.user_profiles.columns or self.user_profiles['bmi'].isna().any():
                self.user_profiles['bmi'] = self.user_profiles['weight_kg'] / (
                    (self.user_profiles['height_cm'] / 100) ** 2
                )
            
            self.user_profiles.fillna(required_defaults, inplace=True)
            self.user_profiles['user_id'] = self.user_profiles['user_id'].astype(str).str.strip()
            
            # Load training data
            self.X_train = pd.read_csv('artifacts/processed/X_train_final.csv')
            self.X_train_columns = self.X_train.columns.tolist()
            
            # CRITICAL FIX: Define type-aware defaults for item features
            self.default_item_features = {
                'material': 'Unknown', 'color': 'Unknown',
                'subcategory': 'Unknown', 'final_price': 50.0,
                'discount_pct': 0.0, 'base_price': 62.5,
                'user_total_purchases': 1, 'bmi': 22.0
            }
            
            self.sample_row = self.X_train.iloc[0].to_dict()
            
            # ========== INITIALIZE FEEDBACK STORAGE ==========
            self.feedback_file = 'artifacts/feedback/user_feedback.csv'
            self.learning_data_file = 'artifacts/feedback/learning_data.csv'
            
            # Create feedback directory if not exists
            os.makedirs('artifacts/feedback', exist_ok=True)
            
            # Load or create feedback log
            if os.path.exists(self.feedback_file):
                self.feedback_log = pd.read_csv(self.feedback_file)
            else:
                self.feedback_log = pd.DataFrame(columns=[
                    'timestamp', 'user_id', 'height_cm', 'weight_kg', 'bmi',
                    'chest_bust_cm', 'waist_cm', 'hip_cm', 'gender', 'age',
                    'brand', 'category', 'body_shape',
                    'predicted_size', 'actual_size', 'fit_feedback',
                    'confidence', 'size_score'
                ])
            
            logging.info("="*60)
            logging.info("✅ PIPELINE READY")
            logging.info(f"✅ {len(self.user_profiles)} users loaded")
            logging.info(f"✅ {len(self.feedback_log)} feedback entries")
            logging.info("="*60)
            
        except Exception as e:
            logging.error(f"❌ INITIALIZATION FAILED: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise CustomException(e, sys)
    
    # =================================================================================
    # METHOD: HISTORICAL DIAGNOSIS AND FORWARD RECOMMENDATION (FULL LOGIC)
    # =================================================================================
    def get_historical_diagnosis_and_recommendation(self, user_id, category):
        """
        Analyzes past purchases and non-perfect fit feedback to diagnose the 
        necessary size adjustment (e.g., if XS was 'Too Small', recommend S).
        """
        
        size_order = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL']
        
        if 'user_id' not in self.user_profiles.columns or 'purchased_size' not in self.user_profiles.columns or 'fit_feedback' not in self.user_profiles.columns:
             logging.warning(" 	⚠️ Missing user_id, purchased_size, or fit_feedback columns for historical lookup.")
             return None
             
        user_data = self.user_profiles[
            (self.user_profiles['user_id'] == user_id) & 
            (self.user_profiles['category'] == category)
        ]
        
        if user_data.empty:
            return None

        # Clean feedback and focus on adjustment categories
        feedback_map = {
            'too small': 1, 
            'slightly small': 1, 
            'too large': -1, 
            'slightly large': -1,
            'fit was slightly tight': 1,
            'fit was slightly loose': -1
        }
        
        net_adjustment = 0
        diagnosis_details = []
        
        # 1. Collect net adjustment and detailed history
        for _, row in user_data.iterrows():
            feedback = str(row['fit_feedback']).strip().lower()
            purchased_size = str(row['purchased_size']).strip()
            
            found_adjustment = False
            for key, adj in feedback_map.items():
                if key in feedback:
                    # Apply adjustment only if it's a size we can interpret (S/M/L or numerical)
                    size_is_sml = purchased_size in size_order
                    
                    if size_is_sml or purchased_size.isdigit():
                        net_adjustment += adj
                        diagnosis_details.append({
                            'purchased_size': purchased_size,
                            'feedback': row['fit_feedback'],
                            'reason': str(row.get('return_reason', 'Fit Issue')),
                            'action': 'size_up' if adj > 0 else 'size_down'
                        })
                        found_adjustment = True
                        break

        if not diagnosis_details:
            return None 

        # 2. Determine consensus size and final recommendation
        diagnostic_size = user_data['purchased_size'].mode().iloc[0]
        
        final_adjustment = 0
        if net_adjustment >= 2: final_adjustment = 1
        elif net_adjustment <= -2: final_adjustment = -1
        
        if final_adjustment == 0: return None 
        
        # Determine recommended size based on unit type
        if category == 'Bottoms' and diagnostic_size.isdigit():
            # Numerical size (e.g., 30) -> adjust by 2 inches (1 numerical size unit)
            recommended_size = str(int(diagnostic_size) + (final_adjustment * 2))
            diagnostic_size_base = diagnostic_size
        elif diagnostic_size in size_order:
            # S/M/L size
            diagnostic_idx = size_order.index(diagnostic_size)
            new_idx = max(0, min(len(size_order) - 1, diagnostic_idx + final_adjustment))
            recommended_size = size_order[new_idx]
            diagnostic_size_base = diagnostic_size
        else:
             return None # Cannot reliably adjust a non-standard size

        logging.info(f" 	🏆 HISTORICAL DIAGNOSIS: {diagnosis_details[0]['feedback']}... Recommended Size: {recommended_size}")

        return {
            'recommended_size': recommended_size,
            'base_size': diagnostic_size_base,
            'predicted_fit': 'Adjusted for History',
            'confidence': 0.95,
            'diagnosis_message': f"Adjusted from '{diagnostic_size_base}' to '{recommended_size}' based on repeated feedback.",
            'detailed_history': diagnosis_details
        }
    # =================================================================================
    
    def calculate_advanced_size(self, user_data, predicted_fit):
        """
        COMPREHENSIVE SIZE CALCULATION
        Includes numerical sizing logic for Bottoms (Waist size) with calibration fix.
        """
        size_order = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL']
        
        height = float(user_data['height_cm'])
        weight = float(user_data['weight_kg'])
        chest = float(user_data['chest_bust_cm'])
        waist = float(user_data['waist_cm'])
        hip = float(user_data['hip_cm'])
        gender = user_data['gender']
        brand = user_data.get('brand', '').lower()
        category = user_data.get('category', 'Tops')
        
        bmi = weight / ((height / 100) ** 2)
        
        logging.info(f" 	📊 H={height:.0f}cm, W={weight:.0f}kg, BMI={bmi:.1f}")
        logging.info(f" 	📊 Chest={chest:.0f}, Waist={waist:.0f}, Hip={hip:.0f}")
        
        # ========== STEP 1: BASE SIZE FROM BMI (S/M/L) ==========
        if gender == 'Male':
            # Adjusted BMI thresholds to push average users towards 'M' size range
            if bmi < 20: base_size = 'S'
            elif bmi < 25: base_size = 'M'
            elif bmi < 29: base_size = 'L'
            elif bmi < 33: base_size = 'XL'
            else: base_size = 'XXL'
        else: 	# Female
            if bmi < 18: base_size = 'XS'
            elif bmi < 21: base_size = 'S'
            elif bmi < 24: base_size = 'M'
            elif bmi < 28: base_size = 'L'
            elif bmi < 31: base_size = 'XL'
            else: base_size = 'XXL'
        
        logging.info(f" 	📏 Base Size (BMI): {base_size}")
        
        # ========== STEP 2-7: Adjustments (Original logic) ==========
        height_adj, weight_adj, proportion_adj, brand_adj, category_adj, fit_adj = 0, 0, 0, 0, 0, 0
        
        if gender == 'Male':
            if height < 160: height_adj = -1.0 
            elif height < 165: height_adj = -0.5 
            elif height >= 195: height_adj = 2.0 
            elif height >= 190: height_adj = 1.5 
            elif height >= 185: height_adj = 1.0 
            elif height >= 180: height_adj = 0.5 
        else:
            if height < 150: height_adj = -1.0 
            elif height < 155: height_adj = -0.5 
            elif height >= 180: height_adj = 1.5 
            elif height >= 175: height_adj = 1.0 
            elif height >= 170: height_adj = 0.5 
        
        if gender == 'Male':
            if weight < 55 and height < 170: weight_adj = -0.3 
            elif weight > 95 and height < 180: weight_adj = 0.3 
        else: 
            if weight < 45 and height < 165: weight_adj = -0.3 
            elif weight > 75 and height < 170: weight_adj = 0.3 
        
        if chest > 0 and waist > 0:
            cw_ratio = chest / waist
            if gender == 'Male':
                if cw_ratio < 1.10: proportion_adj -= 0.2
                elif cw_ratio > 1.40: proportion_adj += 0.2
            else:
                if cw_ratio < 1.10: proportion_adj -= 0.15
                elif cw_ratio > 1.35: proportion_adj += 0.15
        
        if brand in ['zara', 'h&m'] and bmi < 22: brand_adj = 0.3
        elif brand in ['gap', 'old navy'] and bmi > 25: brand_adj = -0.3
        
        if category == 'Tops' and gender == 'Female': category_adj = -0.1
        elif category == 'Bottoms': category_adj = 0.1
        
        if 'Too Small' in predicted_fit: fit_adj = 0.25
        elif 'Too Large' in predicted_fit: fit_adj = -0.25

        total_adj = height_adj + weight_adj + proportion_adj + brand_adj + category_adj + fit_adj
        logging.info(f" 	🔢 Total Adjustment: {total_adj:+.2f}")
        
        # ========== APPLY ADJUSTMENT to S/M/L size (Intermediate Step) ==========
        base_idx = size_order.index(base_size) if base_size in size_order else 3
        
        if total_adj >= 0.5: final_adj = round(total_adj)
        elif total_adj <= -0.5: final_adj = round(total_adj)
        else: final_adj = 0
        
        final_idx = max(0, min(len(size_order) - 1, base_idx + final_adj))
        recommended_size_sml = size_order[final_idx]
        
        # Calculate size score (kept original)
        if gender == 'Male':
            size_score = min(100, (bmi - 15) * 4 + (chest - 80) * 0.8 + (waist - 70) * 0.5 + (height - 160) * 0.3)
        else:
            size_score = min(100, (bmi - 15) * 4 + (chest - 75) * 0.6 + (waist - 60) * 0.4 + (hip - 85) * 0.3 + (height - 150) * 0.2)
        
        size_score = max(0, min(100, size_score))

        # ========== NUMERICAL SIZE CONVERSION FOR BOTTOMS (Final output) ==========
        if category == 'Bottoms':
            # FIX: Use Hip and Waist for robust numerical sizing
            waist_in_inches = waist / 2.54
            hip_in_inches = hip / 2.54
            
            # Use the average of the two, and slightly bias upwards (0.5 inch vanity/comfort)
            # This ensures even a low waist measurement translates to a wearable size
            base_numerical_size_float = (waist_in_inches + hip_in_inches) / 2 + 0.5 
            
            # Round to the nearest even pant size (28, 30, 32, 34...)
            base_numerical_size = int(round(base_numerical_size_float / 2) * 2) 
            
            # Apply S/M/L prediction adjustment (1 SML step = 2 inches for bottoms)
            size_diff = size_order.index(recommended_size_sml) - size_order.index('M')
            numerical_size = base_numerical_size + size_diff * 2 
            
            # Final clamp: Ensure a minimum of 30 for average height male for comfort/length.
            recommended_size = str(max(30, min(40, numerical_size)))
        else:
            recommended_size = recommended_size_sml
        
        logging.info(f" 	🎯 FINAL SIZE: {recommended_size}")
        
        return base_size, recommended_size, size_score
    
    # =================================================================================
    # AUXILIARY METHODS (Fully reconstructed)
    # =================================================================================
    def save_feedback(self, user_data, predicted_size, actual_size=None, fit_feedback=None, confidence=0, size_score=0):
        """Save user feedback for model retraining (Original method)"""
        try:
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_data.get('user_id', 'anonymous'),
                'height_cm': user_data['height_cm'],
                'weight_kg': user_data['weight_kg'],
                'bmi': user_data['weight_kg'] / ((user_data['height_cm']/100) ** 2),
                'chest_bust_cm': user_data['chest_bust_cm'],
                'waist_cm': user_data['waist_cm'],
                'hip_cm': user_data['hip_cm'],
                'gender': user_data['gender'],
                'age': user_data.get('age', 30),
                'brand': user_data.get('brand', 'Unknown'),
                'category': user_data.get('category', 'Tops'),
                'body_shape': user_data.get('body_shape', 'Average'),
                'predicted_size': predicted_size,
                'actual_size': actual_size or predicted_size,
                'fit_feedback': fit_feedback or 'pending',
                'confidence': confidence,
                'size_score': size_score
            }
            
            self.feedback_log = pd.concat([
                self.feedback_log,
                pd.DataFrame([feedback_entry])
            ], ignore_index=True)
            
            self.feedback_log.to_csv(self.feedback_file, index=False)
            
            logging.info(f" 	💾 Feedback saved for user {feedback_entry['user_id']}")
            
            self.check_retrain_trigger()
            
        except Exception as e:
            logging.error(f"Error saving feedback: {str(e)}")
    
    def check_retrain_trigger(self):
        """Check if model should be retrained (Original method)"""
        try:
            feedback_count = len(self.feedback_log)
            
            if feedback_count > 0 and feedback_count % 100 == 0:
                logging.info(f"🔄 Retrain trigger: {feedback_count} feedback entries")
                self.prepare_retraining_data()
                
        except Exception as e:
            logging.error(f"Error checking retrain trigger: {str(e)}")
    
    def prepare_retraining_data(self):
        """Prepare data for model retraining (Original method)"""
        try:
            confirmed_feedback = self.feedback_log[
                self.feedback_log['fit_feedback'].isin(['perfect', 'good', 'acceptable'])
            ].copy()
            
            if len(confirmed_feedback) < 50:
                logging.info(" 	ℹ️ Not enough confirmed feedback for retraining")
                return
            
            confirmed_feedback.to_csv(self.learning_data_file, index=False)
            
            logging.info(f" 	✅ Prepared {len(confirmed_feedback)} entries for retraining")
            logging.info(f" 	📁 Saved to: {self.learning_data_file}")
            
            with open('artifacts/feedback/retrain_required.flag', 'w') as f:
                f.write(f"Retrain required: {len(confirmed_feedback)} new samples\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            
        except Exception as e:
            logging.error(f"Error preparing retraining data: {str(e)}")
    
    def create_input_features(self, user_data):
        """
        FIXED: Create input features. 
        Uses type-aware defaults to prevent 'Unknown' strings in numeric fields.
        """
        try:
            # 1. Initialize data dictionary with safe, type-aware defaults
            data = {}
            for col in self.X_train_columns:
                if col in self.X_train.columns and pd.api.types.is_numeric_dtype(self.X_train[col].dtype):
                    data[col] = self.default_item_features.get(col, 0.0) 
                else:
                    data[col] = self.default_item_features.get(col, 'Unknown')

            # 2. Define mapping from user_data keys to training columns
            mapping = {
                'age': 'age', 'gender': 'gender', 'height_cm': 'height_cm',
                'weight_kg': 'weight_kg', 'body_shape': 'body_shape',
                'chest_bust_cm': 'chest_bust_cm', 'waist_cm': 'waist_cm',
                'hip_cm': 'hip_cm', 'brand': 'brand', 'category': 'category',
                'material': 'material', 'color': 'color',
                'subcategory': 'subcategory', 'price': 'final_price',
                'discount': 'discount_pct', 'user_id': 'user_id', 
                'user_total_purchases': 'user_total_purchases'
            }
            
            # 3. Apply user input, overwriting defaults/placeholders
            for user_key, train_col in mapping.items():
                if user_key in user_data and train_col in data:
                    data[train_col] = user_data[user_key]

            # 4. Calculate derived features (BMI and Base Price)
            height = float(data.get('height_cm', 165))
            weight = float(data.get('weight_kg', 60))
            if height > 0:
                data['bmi'] = weight / ((height/100) ** 2)
            else:
                 data['bmi'] = 0.0
            
            if 'base_price' in data and 'final_price' in data:
                final_price = float(data.get('final_price', 50.0))
                data['base_price'] = final_price * 1.25 
            
            # 5. Create DataFrame and ensure final column order
            input_df = pd.DataFrame([data])
            input_df = input_df[[col for col in self.X_train_columns if col in input_df.columns]]
            
            return input_df
            
        except Exception as e:
            logging.error(f"❌ Feature error: {str(e)}")
            raise CustomException(e, sys)
    
    def get_cross_category_sizes(self, recommended_size, user_data):
        """Get cross-category sizes (Original method)"""
        try:
            categories = ['Tops', 'Bottoms', 'Dresses', 'Outerwear', 'Activewear']
            predictions = {}
            
            chest = float(user_data.get('chest_bust_cm', 90))
            waist = float(user_data.get('waist_cm', 75))
            hip = float(user_data.get('hip_cm', 95))
            
            chest_waist = chest / waist if waist > 0 else 1.2
            waist_hip = waist / hip if hip > 0 else 0.8
            
            size_order = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL']
            
            # Handle numerical size base_size for Bottoms correctly
            if str(recommended_size).isdigit() and user_data.get('category') == 'Bottoms':
                 predictions = {cat: recommended_size for cat in categories}
                 return predictions

            sml_base = recommended_size
            base_idx = size_order.index(sml_base) if sml_base in size_order else 3
            
            for cat in categories:
                if cat == user_data.get('category'):
                    predictions[cat] = recommended_size
                elif cat == 'Bottoms' and not str(recommended_size).isdigit():
                    predictions[cat] = str(int(round(float(user_data.get('waist_cm', 75)) / 2.54 / 2) * 2)) # Numeric waist size approximation
                else:
                    predictions[cat] = size_order[base_idx]
            
            return predictions
            
        except Exception as e:
            logging.error(f"❌ Cross-category error: {str(e)}")
            return {cat: recommended_size for cat in categories}
    
    def predict(self, user_data):
        """
        Main prediction method. Runs ML prediction.
        UPDATED: Uses only CatBoost and XGBoost (Random Forest disabled)
        """
        try:
            logging.info("="*60)
            logging.info("🎯 STARTING PREDICTION")
            logging.info("="*60)
            
            required = ['age', 'gender', 'height_cm', 'weight_kg', 'body_shape',
                        'chest_bust_cm', 'waist_cm', 'hip_cm', 'brand', 'category']
            missing = [f for f in required if f not in user_data]
            
            if missing:
                return {'success': False, 'error': f'Missing fields: {missing}'}
            
            # Convert to float safely
            numeric_fields = ['age', 'height_cm', 'weight_kg', 'chest_bust_cm', 'waist_cm', 'hip_cm']
            for field in numeric_fields:
                if field in user_data:
                    try:
                        user_data[field] = float(user_data[field])
                    except:
                        return {'success': False, 'error': f'Invalid {field}'}

            # --- ML PREDICTION (WITHOUT RANDOM FOREST) ---
            
            input_df = self.create_input_features(user_data)
            input_processed = self.preprocessor.transform(input_df)
            
            proba_cat = self.catboost_model.predict_proba(input_processed)
            proba_xgb = self.xgb_model.predict_proba(input_processed)
            
            # DISABLED: Random Forest prediction
            # proba_rf = self.rf_model.predict_proba(input_processed)
            
            # Use only CatBoost and XGBoost for meta model
            X_meta = np.hstack([proba_cat, proba_xgb])
            
            final_proba = self.stacking_meta_model.predict_proba(X_meta)[0]
            final_pred_idx = np.argmax(final_proba)
            confidence = float(final_proba[final_pred_idx])
            predicted_fit = self.label_encoder.inverse_transform([final_pred_idx])[0]
            
            logging.info(f" 	✅ ML Model Predicted Fit: {predicted_fit} ({confidence*100:.1f}%)")
            
            base_sml_size, recommended_size, size_score = self.calculate_advanced_size(
                user_data, predicted_fit
            )
            
            # --- Result Formatting ---
            
            bmi = user_data['weight_kg'] / ((user_data['height_cm']/100) ** 2)
            
            if bmi < 18.5: bmi_category, bmi_emoji = 'Underweight', '📉'
            elif bmi < 25: bmi_category, bmi_emoji = 'Normal', '✅'
            elif bmi < 30: bmi_category, bmi_emoji = 'Overweight', '📈'
            else: bmi_category, bmi_emoji = 'Obese', '⚠️'

            result = {
                'success': True,
                'user_id': user_data.get('user_id', 'New User'),
                'predicted_fit': predicted_fit,
                'confidence': round(confidence * 100, 2),
                'recommended_size': recommended_size,
                'usual_size': base_sml_size,
                'size_score': round(size_score, 1),
                'cross_category_sizes': self.get_cross_category_sizes(recommended_size, user_data),
                'body_insights': {
                    'bmi': round(bmi, 2),
                    'bmi_category': bmi_category,
                    'bmi_emoji': bmi_emoji,
                    'chest_waist_ratio': round(user_data['chest_bust_cm'] / user_data['waist_cm'], 2) if user_data['waist_cm'] > 0 else 0,
                    'waist_hip_ratio': round(user_data['waist_cm'] / user_data['hip_cm'], 2) if user_data['hip_cm'] > 0 else 0
                }
            }
            
            logging.info("="*60)
            logging.info("✅ PREDICTION COMPLETE")
            logging.info(f" 	📏 Recommended Size: {recommended_size}")
            logging.info(f" 	🎯 Confidence: {confidence*100:.1f}%")
            logging.info("="*60)
            
            return result
            
        except Exception as e:
            logging.error(f"❌ PREDICTION FAILED: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return {'success': False, 'error': str(e), 'fallback_size': 'M'}


class CustomData:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_data_as_dict(self):
        return self.__dict__