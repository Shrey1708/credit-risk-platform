import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
try:
    import shap
except ImportError:
    pass

# Add src to path to import data modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.preprocessor import DataPreprocessor
from src.ml.rules import BusinessRulesEngine
from src.utils.logger import get_logger
from src.utils.config import Config
from src.utils.helpers import classify_risk_band

logger = get_logger("risk_predictor")

FEATURE_DISPLAY_NAMES = {
    "EXT_SOURCE_1": "External Bureau Rating 1 (EXT_SOURCE_1)",
    "EXT_SOURCE_2": "External Bureau Rating 2 (EXT_SOURCE_2)",
    "EXT_SOURCE_3": "External Bureau Rating 3 (EXT_SOURCE_3)",
    "AMT_INCOME_TOTAL": "Total Annual Income",
    "AMT_CREDIT": "Loan Credit Amount Requested",
    "AMT_ANNUITY": "Loan Annuity (Annual Payment)",
    "DAYS_BIRTH": "Applicant Age",
    "DAYS_EMPLOYED": "Employment Duration",
    "CODE_GENDER_M": "Gender: Male",
    "CODE_GENDER_F": "Gender: Female",
    "NAME_EDUCATION_TYPE_Highereducation": "Education: Higher Education",
    "NAME_EDUCATION_TYPE_Secondarysecondaryspecial": "Education: Secondary / Special",
    "NAME_EDUCATION_TYPE_Lowersecondary": "Education: Lower Secondary",
    "NAME_EDUCATION_TYPE_Incompletehigher": "Education: Incomplete Higher",
    "INSTAL_PAYMENT_DELAY_sum": "Past Installment Payment Delays (sum)",
    "INSTAL_PAYMENT_DELAY_max": "Max Past Installment Delay",
    "INSTAL_AMT_INSTALMENT_sum": "Total Past Installment Obligations",
    "BUREAU_AMT_CREDIT_SUM_mean": "Credit Bureau Past Loan Amounts (mean)",
    "BUREAU_AMT_CREDIT_SUM_DEBT_mean": "Credit Bureau Outstanding Debt (mean)",
    "CC_CNT_DRAWINGS_CURRENT_min": "Credit Card Monthly Drawings",
    "FLAG_DOCUMENT_3": "Required Identity Document 3 Submitted",
    "NAME_INCOME_TYPE_Working": "Income Source: Working",
    "NAME_INCOME_TYPE_Pensioner": "Income Source: Pensioner",
    "NAME_INCOME_TYPE_Stateservant": "Income Source: State Servant",
    "NAME_INCOME_TYPE_Commercialassociate": "Income Source: Commercial Associate",
    "OCCUPATION_TYPE_Laborers": "Occupation: Laborer",
    "OCCUPATION_TYPE_LowskillLaborers": "Occupation: Low-skill Laborer",
    "ORGANIZATION_TYPE_Restaurant": "Organization: Restaurant",
    "ORGANIZATION_TYPE_Industrytype9": "Organization: Industry Type 9",
}

class RiskPredictor:
    """
    Handles inference for credit risk default probabilities.
    Outputs a Probability, a Risk Score (0-1000), and a Risk Band.
    """
    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir if models_dir is not None else str(Config.MODELS_DIR)
        self.rules_engine = BusinessRulesEngine()
        self._load_best_model()

    def _load_best_model(self):
        model_path = os.path.join(self.models_dir, "AutoML_Best_Model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError("AutoML_Best_Model.pkl not found. Please run train.py first.")
        
        logging.info(f"Loading AutoML model from {model_path}")
        self.model = joblib.load(model_path)
        
        # Load fitted preprocessor
        prep_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        self.preprocessor = joblib.load(prep_path)
        
        # Setup SHAP Explainer
        try:
            # Extract the underlying model from the FLAML wrapper (e.g., XGBoost, LightGBM)
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'estimator'):
                underlying_estimator = self.model.model.estimator
            else:
                underlying_estimator = self.model # Fallback if not wrapped by FLAML
                
            self.underlying_estimator = underlying_estimator
            logging.info("Initializing SHAP TreeExplainer...")
            self.explainer = shap.TreeExplainer(self.underlying_estimator)
        except Exception as e:
            logging.warning(f"Could not initialize SHAP Explainer (Explainability will be disabled): {e}")
            self.explainer = None



    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts raw input data, preprocesses it, and generates predictions.
        Returns a DataFrame with Probability, Risk Score, and Risk Band.
        """
        logging.info("Preprocessing input data...")
        df_clean = self.preprocessor.transform(input_data)
        
        # Prepare features
        drop_cols = ['TARGET', 'SK_ID_CURR']
        X = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])
        
        logging.info("Generating predictions...")
        # Get probability of class 1 (Default)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Calculate SHAP values if explainer is available
        shap_values_list = None
        if self.explainer is not None:
            logging.info("Calculating SHAP values for explainability...")
            try:
                # Fix feature dimension mismatch (FLAML internally drops zero-variance features)
                # Ensure we only pass the exact features the underlying booster expects
                if hasattr(self.underlying_estimator, 'feature_names_in_'):
                    X_shap = X[self.underlying_estimator.feature_names_in_]
                elif hasattr(self.underlying_estimator, 'get_booster'):
                    X_shap = X[self.underlying_estimator.get_booster().feature_names]
                else:
                    X_shap = X

                # TreeExplainer usually returns an array of shape (n_samples, n_features) 
                # or a list of arrays for multi-class. We want the positive class (usually index 1 or just the array).
                shap_output = self.explainer.shap_values(X_shap)
                if isinstance(shap_output, list):
                    shap_values_list = shap_output[1] # Positive class
                else:
                    shap_values_list = shap_output
                    
                # Store the feature names used for shap calculations to correctly map the explanations
                shap_feature_names = X_shap.columns.tolist()
            except Exception as e:
                logging.warning(f"Failed to calculate SHAP values: {e}")
        
        results = []
        for i, prob in enumerate(probabilities):
            # Calibrate Risk Score (0 to 1000 scale) and Normalized Probability based on actual portfolio distribution
            # In this imbalanced dataset (8.07% default rate), raw probabilities range from ~0.055 to ~0.330
            P_MIN = 0.055
            P_MAX = 0.330
            norm_p = float(np.clip((prob - P_MIN) / (P_MAX - P_MIN), 0.0, 1.0))
            calibrated_score = int(norm_p * 900 + 50)
            normalized_prob = round(norm_p, 4)
            relative_risk = round(float(prob) / 0.0807, 1)
            portfolio_percentile = min(99.9, max(1.0, round(norm_p * 98.0 + 1.0, 1)))
            
            # Determine Risk Band calibrated to risk percentiles
            if prob < 0.10:
                risk_band = "Low"
            elif prob < 0.18:
                risk_band = "Medium"
            else:
                risk_band = "High"
                
            prediction_record = {
                'Default_Probability': normalized_prob,
                'Raw_Default_Probability': round(prob, 4),
                'Normalized_Default_Probability': normalized_prob,
                'Risk_Score': calibrated_score,
                'Risk_Band': risk_band,
                'Relative_Risk': relative_risk,
                'Portfolio_Percentile': portfolio_percentile,
                'Top_Risk_Drivers': '',
                'Top_Mitigating_Factors': ''
            }
            
            # Process SHAP explanations for this specific row
            if shap_values_list is not None:
                row_shap = shap_values_list[i]
                
                # Combine feature names with their shap values using the exact features SHAP saw
                shap_dict = {shap_feature_names[j]: row_shap[j] for j in range(len(shap_feature_names))}
                
                # Sort features by SHAP value
                sorted_features = sorted(shap_dict.items(), key=lambda item: item[1])
                
                # Helper to format feature names nicely
                def _fmt(name: str) -> str:
                    return FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " "))

                # Top positive contributors (pushing risk UP)
                positive_contributors = [f"{_fmt(k)} (+{v:.3f})" for k, v in reversed(sorted_features[-3:]) if v > 0]
                # Top negative contributors (pushing risk DOWN)
                negative_contributors = [f"{_fmt(k)} ({v:.3f})" for k, v in sorted_features[:3] if v < 0]
                
                prediction_record['Top_Risk_Drivers'] = " | ".join(positive_contributors)
                prediction_record['Top_Mitigating_Factors'] = " | ".join(negative_contributors)
                
            # Evaluate Business Rules interception
            # Convert row to dict for rules engine
            row_dict = input_data.iloc[i].to_dict()
            rule_eval = self.rules_engine.evaluate(row_dict)
            
            if rule_eval['rejected']:
                # Override the ML prediction with hard rule rejection
                prediction_record['Default_Probability'] = 1.0
                prediction_record['Normalized_Default_Probability'] = 1.0
                prediction_record['Raw_Default_Probability'] = 1.0
                prediction_record['Risk_Score'] = 1000
                prediction_record['Risk_Band'] = "High Risk (Rule Reject)"
                prediction_record['Top_Risk_Drivers'] = rule_eval['reason']
                prediction_record['Top_Mitigating_Factors'] = "N/A - Overridden by Business Rule"

            results.append(prediction_record)
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    models_dir = os.path.join(base_dir, 'models')
    
    try:
        predictor = RiskPredictor(models_dir=models_dir)
        
        # Create a dummy payload matching raw data structure
        # (In a real scenario, this would come from an API or database)
        dummy_data = pd.DataFrame({
            'AMT_INCOME_TOTAL': [150000, 50000],
            'AMT_CREDIT': [500000, 1000000],
            'DAYS_BIRTH': [-15000, -8000],
            'DAYS_EMPLOYED': [-2000, 365243], # 365243 is the pensioner anomaly
            'NAME_EDUCATION_TYPE': ['Higher education', 'Secondary / secondary special'],
            'CODE_GENDER': ['M', 'F']
        })
        
        print("\n--- Inference Results & Explanations ---")
        predictions = predictor.predict(dummy_data)
        
        # Print transposed for readability since we have many columns now
        for idx, row in predictions.iterrows():
            print(f"\n--- Applicant {idx + 1} ---")
            for col, val in row.items():
                print(f"{col:25}: {val}")
        print("\n----------------------------------------\n")
    except Exception as e:
        logging.error(f"Inference failed: {e}")
