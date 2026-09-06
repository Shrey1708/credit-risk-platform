import os
import sys
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.talk_to_data.query_runner import QueryRunner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_database():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(base_dir, 'data')
    db_path = os.path.join(base_dir, 'credit_risk.db')
    
    runner = QueryRunner(db_path=db_path)
    
    # Define which CSV files to load into the SQLite database.
    # We load the main application_train table along with all major historical tables
    # so the LLM can perform complex JOINs to answer user questions.
    tables_to_load = {
        'application_train': 'application_train.csv',
        'bureau': 'bureau.csv',
        'previous_application': 'previous_application.csv',
        'pos_cash_balance': 'POS_CASH_balance.csv',
        'installments_payments': 'installments_payments.csv',
        'credit_card_balance': 'credit_card_balance.csv'
    }
    
    logging.info(f"Initializing database at {db_path}...")
    
    for table_name, file_name in tables_to_load.items():
        csv_path = os.path.join(data_dir, file_name)
        if os.path.exists(csv_path):
            success = runner.load_csv_to_db(csv_path, table_name)
            if success:
                logging.info(f"Successfully created table: {table_name}")
            else:
                logging.error(f"Failed to create table: {table_name}")
        else:
            logging.warning(f"File not found, skipping: {csv_path}")

    logging.info("Database initialization complete!")

if __name__ == "__main__":
    initialize_database()
