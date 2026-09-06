import logging
from typing import Optional, Dict, Any, Union
import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import APIError

from .query_runner import QueryRunner
from .prompt_templates import SQL_GENERATION_PROMPT, NL_RESPONSE_PROMPT
from .guardrails import RateLimiter, InputGuardrail, SQLGuardrail
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger("nl_to_sql")

# Modern Google GenAI safety settings to prevent harmful content generation
GEMINI_SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
]

class TalkToData:
    """
    Translates Natural Language queries into SQL, executes them against SQLite,
    and formats the response back into natural language with rigorous guardrails:
    - Sliding-window rate limiting (15 RPM / 500 RPD)
    - Input injection detection
    - Read-only SQL enforcement
    - Execution timeout & row caps
    """
    def __init__(
        self, 
        api_key: str = None, 
        model_name: str = Config.GEMINI_MODEL_NAME, 
        db_path: str = str(Config.DATABASE_PATH),
        max_rpm: int = Config.RATE_LIMIT_RPM,
        max_rpd: int = Config.RATE_LIMIT_RPD
    ):
        actual_key = api_key if api_key else Config.GEMINI_API_KEY
        # Configure modern Google GenAI Client
        self.client = genai.Client(api_key=actual_key)
        self.model_name = model_name
        self.config = types.GenerateContentConfig(
            safety_settings=GEMINI_SAFETY_SETTINGS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
        self.query_runner = QueryRunner(db_path=db_path)
        self.rate_limiter = RateLimiter(max_rpm=max_rpm, max_rpd=max_rpd)
        self.input_guardrail = InputGuardrail(min_length=3, max_length=600)
        self.sql_guardrail = SQLGuardrail(default_limit=100, max_limit=500)
        self.last_error = None

    def get_rate_limit_status(self):
        """Returns the current rate limit usage stats."""
        return self.rate_limiter.get_status()

    def _generate_sql(self, question: str, schema: str) -> str:
        """Calls the LLM to generate SQL based on the natural language question and database schema."""
        prompt = SQL_GENERATION_PROMPT.format(schema=schema, question=question)
        logging.info("Calling Gemini API to generate SQL...")
        self.last_error = None
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.config
            )
            self.rate_limiter.record_call()
            sql_query = response.text.strip()
            
            # Clean up markdown formatting if the model included it
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            elif sql_query.startswith("```"):
                sql_query = sql_query[3:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
                
            return sql_query.strip()
        except APIError as e:
            if getattr(e, 'code', None) == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                self.rate_limiter.record_resource_exhausted(cooldown_seconds=30.0)
                self.last_error = "Gemini API rate limit reached (HTTP 429 ResourceExhausted). Please wait 30s."
            else:
                self.last_error = f"Gemini API Error: {e}"
            logging.error(f"Error generating SQL: {self.last_error}")
            return ""
        except Exception as e:
            self.last_error = str(e)
            logging.error(f"Error generating SQL: {e}")
            return ""

    def _generate_nl_response(self, question: str, sql_query: str, sql_results: str) -> str:
        """Calls the LLM to summarize the SQL results into a readable natural language format."""
        prompt = NL_RESPONSE_PROMPT.format(
            question=question, 
            sql_query=sql_query, 
            sql_results=sql_results
        )
        logging.info("Calling Gemini API to generate natural language response...")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.config
            )
            self.rate_limiter.record_call()
            return response.text.strip()
        except APIError as e:
            if getattr(e, 'code', None) == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                self.rate_limiter.record_resource_exhausted(cooldown_seconds=30.0)
                return "Gemini API rate limit reached (HTTP 429). Please wait 30s before generating another natural language response."
            logging.error(f"Error generating NL response: {e}")
            return f"Gemini API Error: {e}"
        except Exception as e:
            logging.error(f"Error generating NL response: {e}")
            return "Sorry, I encountered an error while formulating the response."

    def ask(self, question: str, return_dict: bool = False):
        """
        Main method to process a user's natural language question.
        Applies input validation, rate limiting, and SQL security guardrails.
        
        Args:
            question: The natural language question to ask.
            return_dict: If True, returns a dict {'answer': str, 'sql': Optional[str]}.
                         If False, returns a clean formatted string.
        """
        def format_return(answer_text: str, sql_query: Optional[str] = None):
            if return_dict:
                return {"answer": answer_text, "sql": sql_query}
            if sql_query:
                return f"{answer_text}\n\n**Executed SQL Query:**\n```sql\n{sql_query}\n```"
            return answer_text

        # 1. Input Guardrail: Validate query against injections and length restrictions
        is_valid, input_err = self.input_guardrail.validate(question)
        if not is_valid:
            logging.warning(f"Input Guardrail Blocked Query: {input_err}")
            return format_return(f"🛡️ **Security Guardrail Notice**: {input_err}")

        # 2. Rate Limiting Check (before SQL generation)
        allowed, wait_sec, rate_msg = self.rate_limiter.check_allowance()
        if not allowed:
            logging.warning(f"Rate Limiter Blocked Request: {rate_msg}")
            return format_return(f"⏱️ **Rate Limit Alert**: {rate_msg}")

        # 3. Retrieve database schema (cached)
        schema = self.query_runner.get_schema()
        if not schema:
            return format_return("Error: Could not retrieve database schema. Is the database initialized?")

        # 4. Generate SQL
        raw_sql_query = self._generate_sql(question, schema)
        if not raw_sql_query:
            err_msg = f": {self.last_error}" if hasattr(self, 'last_error') and self.last_error else "."
            return format_return(f"Error: Could not generate a valid SQL query for your question{err_msg}")

        # 5. SQL Security Guardrail: Validate read-only nature and sanitize
        is_safe_sql, sanitized_sql, sql_err = self.sql_guardrail.validate_and_sanitize(raw_sql_query)
        if not is_safe_sql:
            logging.warning(f"SQL Guardrail Blocked Query: {sql_err}")
            return format_return(f"🚨 **SQL Guardrail Alert**: {sql_err}\n\n*Attempted Query:*\n```sql\n{raw_sql_query}\n```")

        # 6. Execute SQL with row limits and timeout
        columns, rows = self.query_runner.execute_query(sanitized_sql)
        
        # Format results for the prompt
        if isinstance(rows, str): 
            # If rows is a string, it means an exception occurred during execution
            results_str = f"Execution Error: {rows}"
        elif not rows:
            results_str = "No results found."
        else:
            # Format as a list of dictionaries for readability by the LLM
            results_str = str([dict(zip(columns, row)) for row in rows])

        # 7. Rate Limiting Check (before Natural Language Response generation)
        allowed_nl, wait_sec_nl, rate_msg_nl = self.rate_limiter.check_allowance()
        if not allowed_nl:
            # Graceful degradation: return raw SQL results if rate limit is reached
            fallback_table = pd.DataFrame(rows, columns=columns).to_markdown() if columns and rows and not isinstance(rows, str) else str(rows)
            msg = (
                f"⏱️ **Rate Limit Alert**: {rate_msg_nl}\n\n"
                f"To conserve quota, here are the raw query results directly from the database:\n\n"
                f"{fallback_table}"
            )
            return format_return(msg, sanitized_sql)

        # 8. Generate Natural Language Response
        final_answer = self._generate_nl_response(question, sanitized_sql, results_str)
        return format_return(final_answer, sanitized_sql)

