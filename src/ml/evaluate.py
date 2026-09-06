import os
import glob
import logging
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_models(models_dir: str):
    """
    Evaluates all trained models in the models directory and selects the best one.
    """
    test_data_path = os.path.join(models_dir, 'test_data.pkl')
    if not os.path.exists(test_data_path):
        logging.error(f"Test data not found at {test_data_path}. Please run train.py first.")
        return

    logging.info("Loading test data...")
    X_test, y_test = joblib.load(test_data_path)

    model_path = os.path.join(models_dir, 'AutoML_Best_Model.pkl')
    if not os.path.exists(model_path):
        logging.error("AutoML model not found. Please run train.py first.")
        return

    logging.info("Evaluating AutoML_Best_Model...")
    
    try:
        model = joblib.load(model_path)
        
        # Predict probabilities and classes
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print("\n--- AutoML Model Evaluation Results ---")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print("---------------------------------------\n")
            
    except Exception as e:
        logging.error(f"Failed to evaluate AutoML_Best_Model: {e}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    models_dir = os.path.join(base_dir, 'models')
    evaluate_models(models_dir)
