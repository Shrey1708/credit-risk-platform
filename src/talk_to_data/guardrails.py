import time
import re
import threading
from typing import Tuple, Dict, Any, List

class RateLimiter:
    """
    Thread-safe sliding-window rate limiter designed for Google Gemini Free Tier quotas:
    - 15 Requests Per Minute (RPM)
    - 500 Requests Per Day (RPD)
    """
    def __init__(self, max_rpm: int = 15, max_rpd: int = 500):
        self.max_rpm = max_rpm
        self.max_rpd = max_rpd
        self.lock = threading.Lock()
        
        # Track request timestamps
        self.minute_calls: List[float] = []
        self.day_calls: List[float] = []
        self.cooldown_until: float = 0.0

    def _cleanup_old_calls(self, current_time: float):
        """Purges timestamps older than the sliding windows."""
        # 60-second sliding window for RPM
        minute_cutoff = current_time - 60.0
        self.minute_calls = [t for t in self.minute_calls if t > minute_cutoff]

        # 86,400-second (24-hour) sliding window for RPD
        day_cutoff = current_time - 86400.0
        self.day_calls = [t for t in self.day_calls if t > day_cutoff]

    def check_allowance(self) -> Tuple[bool, float, str]:
        """
        Checks if an API request is permitted under RPM, RPD, and cooldown.
        Returns:
            (allowed: bool, wait_seconds: float, reason: str)
        """
        with self.lock:
            now = time.time()
            self._cleanup_old_calls(now)

            # Check if active cooldown is in effect (e.g., after a Google 429)
            if now < self.cooldown_until:
                wait_time = self.cooldown_until - now
                return False, wait_time, f"Active API cooldown in effect. Please wait {wait_time:.1f}s."

            # Check RPM (15 requests per 60 seconds)
            if len(self.minute_calls) >= self.max_rpm:
                oldest_in_minute = self.minute_calls[0]
                wait_time = max(0.5, 60.0 - (now - oldest_in_minute))
                return False, wait_time, f"Rate limit reached ({self.max_rpm} requests/min). Please wait {wait_time:.1f}s."

            # Check RPD (500 requests per 24 hours)
            if len(self.day_calls) >= self.max_rpd:
                oldest_in_day = self.day_calls[0]
                wait_time = max(1.0, 86400.0 - (now - oldest_in_day))
                return False, wait_time, f"Daily limit reached ({self.max_rpd} requests/day). Limit resets in {wait_time / 3600:.1f}h."

            return True, 0.0, ""

    def record_call(self):
        """Records an executed API request timestamp."""
        with self.lock:
            now = time.time()
            self.minute_calls.append(now)
            self.day_calls.append(now)

    def record_resource_exhausted(self, cooldown_seconds: float = 30.0):
        """Sets a mandatory cooldown period if Google API returns a 429 ResourceExhausted error."""
        with self.lock:
            self.cooldown_until = time.time() + cooldown_seconds

    def get_status(self) -> Dict[str, Any]:
        """Returns the current usage statistics for display."""
        with self.lock:
            now = time.time()
            self._cleanup_old_calls(now)
            return {
                "rpm_current": len(self.minute_calls),
                "rpm_max": self.max_rpm,
                "rpd_current": len(self.day_calls),
                "rpd_max": self.max_rpd,
                "rpm_available": max(0, self.max_rpm - len(self.minute_calls)),
                "rpd_available": max(0, self.max_rpd - len(self.day_calls)),
            }


class InputGuardrail:
    """
    Validates natural language questions to prevent prompt injections, 
    excessively large inputs (token exhaustion), and malicious queries.
    """
    # Regex patterns indicating prompt injection or extraction attempts
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above|system)\s+instructions?",
        r"disregard\s+(all\s+)?(previous|prior|above|system)\s+instructions?",
        r"system\s+prompt",
        r"you\s+are\s+now\s+(a|an|in)\b",
        r"jailbreak",
        r"reveal\s+(the\s+)?(password|secret|api\s*key|database\s+credentials?)",
        r"bypass\s+(the\s+)?rules?",
        r"prompt\s+leak",
        r"(drop|delete|truncate|alter)\s+table\b",
    ]

    def __init__(self, min_length: int = 3, max_length: int = 600):
        self.min_length = min_length
        self.max_length = max_length
        self.injection_regex = re.compile(
            "|".join(self.INJECTION_PATTERNS), 
            re.IGNORECASE
        )

    def validate(self, question: str) -> Tuple[bool, str]:
        """
        Validates user input.
        Returns:
            (is_valid: bool, error_message: str)
        """
        if not question or not question.strip():
            return False, "Input query cannot be empty."

        clean_text = question.strip()

        if len(clean_text) < self.min_length:
            return False, f"Query is too short (minimum {self.min_length} characters required)."

        if len(clean_text) > self.max_length:
            return False, f"Query exceeds maximum allowed length ({len(clean_text)}/{self.max_length} characters)."

        # Check for prompt injection patterns
        match = self.injection_regex.search(clean_text)
        if match:
            return False, f"Query violates security policy (prohibited pattern detected: '{match.group(0)}')."

        return True, ""


class SQLGuardrail:
    """
    Strict SQL Validator and Sanitizer:
    1. Enforces read-only operations (SELECT / WITH CTE only).
    2. Prohibits multi-statement query attacks (semicolon injection).
    3. Strictly blocks destructive/modifying keywords (DROP, DELETE, UPDATE, INSERT, ALTER, etc.).
    4. Enforces safety LIMIT clauses to prevent out-of-memory queries.
    """
    FORBIDDEN_KEYWORDS = [
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
        "CREATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA", "VACUUM",
        "REINDEX", "EXEC", "EXECUTE", "INTO", "GRANT", "REVOKE"
    ]

    def __init__(self, default_limit: int = 100, max_limit: int = 500):
        self.default_limit = default_limit
        self.max_limit = max_limit
        
        # Regex to match any forbidden keyword as a whole word
        pattern = r"\b(" + "|".join(self.FORBIDDEN_KEYWORDS) + r")\b"
        self.forbidden_regex = re.compile(pattern, re.IGNORECASE)

    def validate_and_sanitize(self, sql_query: str) -> Tuple[bool, str, str]:
        """
        Validates and sanitizes the generated SQL query.
        Returns:
            (is_safe: bool, sanitized_sql: str, reason_if_unsafe: str)
        """
        if not sql_query or not sql_query.strip():
            return False, "", "Empty SQL query received."

        clean_sql = sql_query.strip()

        # Remove surrounding markdown code block markers if present
        if clean_sql.startswith("```sql"):
            clean_sql = clean_sql[6:]
        elif clean_sql.startswith("```"):
            clean_sql = clean_sql[3:]
        if clean_sql.endswith("```"):
            clean_sql = clean_sql[:-3]
        clean_sql = clean_sql.strip()

        # Strip trailing semicolon
        clean_sql = clean_sql.rstrip(";").strip()

        # Check for multi-statement injection (embedded semicolon)
        if ";" in clean_sql:
            return False, "", "Multiple SQL statements detected (semicolon injection blocked)."

        # Check that query starts with allowed read-only statements
        upper_sql = clean_sql.upper()
        if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH") or upper_sql.startswith("EXPLAIN")):
            return False, "", "Only read-only SELECT and WITH (CTE) queries are permitted."

        # Scan for forbidden modifying keywords anywhere in the statement
        forbidden_match = self.forbidden_regex.search(clean_sql)
        if forbidden_match:
            keyword = forbidden_match.group(0)
            return False, "", f"Destructive SQL operation detected and blocked: keyword '{keyword}' is forbidden."

        # Enforce safety LIMIT clause
        # If no LIMIT is present in the query, append the default limit
        if not re.search(r"\bLIMIT\s+\d+\b", clean_sql, re.IGNORECASE):
            # If query does not aggregate into a single row, append LIMIT
            clean_sql = f"{clean_sql}\nLIMIT {self.default_limit}"
        else:
            # If a limit is present, ensure it does not exceed max_limit
            def limit_cap(match):
                val = int(match.group(1))
                if val > self.max_limit:
                    return f"LIMIT {self.max_limit}"
                return match.group(0)

            clean_sql = re.sub(r"\bLIMIT\s+(\d+)\b", limit_cap, clean_sql, flags=re.IGNORECASE)

        return True, clean_sql, ""
