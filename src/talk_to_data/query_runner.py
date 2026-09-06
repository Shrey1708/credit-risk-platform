import sqlite3
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QueryRunner:
    """
    Handles database connectivity and SQL execution.
    Defaults to a SQLite database.
    """
    def __init__(self, db_path: str = "credit_risk.db"):
        self.db_path = db_path
        self._cached_schema = None

    def _get_connection(self):
        """Returns a new connection to the SQLite database with execution timeout."""
        return sqlite3.connect(self.db_path, timeout=5.0)

    def execute_query(self, query: str, max_rows: int = 100):
        """
        Executes a given SQL query and returns the results.
        Enforces a maximum row cap to prevent memory saturation.
        Returns a tuple: (column_names, rows)
        """
        logging.info(f"Executing SQL Query:\n{query}")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchmany(max_rows)
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                else:
                    columns = []
                return columns, rows
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            return [], str(e)

    def get_schema(self) -> str:
        """
        Retrieves the database schema (tables and columns) formatted as a string.
        This is injected into the LLM prompt and cached in memory.
        """
        if self._cached_schema:
            return self._cached_schema

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                schema_str = ""
                for table in tables:
                    table_name = table[0]
                    schema_str += f"Table: {table_name}\n"
                    # Get columns for each table
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    col_details = [f"  - {col[1]} ({col[2]})" for col in columns]
                    schema_str += "\n".join(col_details) + "\n\n"
                self._cached_schema = schema_str.strip()
                return self._cached_schema
        except Exception as e:
            logging.error(f"Error retrieving schema: {e}")
            return ""

    def load_csv_to_db(self, csv_path: str, table_name: str, chunksize: int = 10000):
        """
        Helper method to automatically load a CSV file into the SQLite database.
        """
        if not os.path.exists(csv_path):
            logging.error(f"CSV file not found: {csv_path}")
            return False

        logging.info(f"Loading {csv_path} into table '{table_name}'...")
        try:
            with self._get_connection() as conn:
                # Read CSV in chunks to avoid memory issues with large files
                for chunk in pd.read_csv(csv_path, chunksize=chunksize):
                    chunk.to_sql(table_name, conn, if_exists='append', index=False)
            logging.info(f"Successfully loaded data into '{table_name}'.")
            return True
        except Exception as e:
            logging.error(f"Error loading CSV to DB: {e}")
            return False
