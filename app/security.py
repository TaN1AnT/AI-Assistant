"""
Security Service — Authentication, authorization, rate limiting, and input sanitization.
"""
import re
import time
from typing import List, Optional, Dict
from collections import defaultdict
import logging

logger = logging.getLogger("security")


# ── Mock User Database ──
# In production, replace with a real Identity Provider (OAuth2, JWT, Okta, Auth0)

MOCK_USERS = {
    "token_admin_123": {
        "email": "admin@example.com",
        "role": "admin",
        "permissions": [
            "view_all_deals", "create_ticket", "update_deal",
            "delete_contact", "send_notification", "manage_tickets",
            "query_crm", "rag_search",
        ],
    },
    "token_sales_456": {
        "email": "sales@example.com",
        "role": "sales_rep",
        "permissions": [
            "view_own_deals", "create_ticket", "update_deal",
            "send_notification", "query_crm", "rag_search",
        ],
    },
    "token_guest_789": {
        "email": "guest@example.com",
        "role": "guest",
        "permissions": ["rag_search"],
    },
}


class SecurityService:
    """Authentication and authorization."""

    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        """
        Validates the token and returns the user profile if valid.
        Returns None if invalid.
        """
        if not token or not isinstance(token, str):
            return None
        return MOCK_USERS.get(token)

    @staticmethod
    def check_permission(user_role: str, user_permissions: List[str], required_permission: str) -> bool:
        """
        Checks if the user has the required permission.
        Admin role bypasses individual permission checks.
        """
        if user_role == "admin":
            return True
        return required_permission in user_permissions

    @staticmethod
    def sanitize_input(text: str) -> str:
        """
        Sanitizes user input to mitigate prompt injection attacks.
        Strips dangerous patterns while preserving the user's intent.
        """
        if not text:
            return text

        # Remove common prompt injection patterns
        injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"disregard\s+(all\s+)?prior\s+(instructions|prompts)",
            r"you\s+are\s+now\s+(?:a|an)\s+(?:new|different)",
            r"system\s*:\s*",
            r"<\s*system\s*>",
            r"<\s*/\s*system\s*>",
            r"\[INST\]",
            r"\[/INST\]",
        ]

        sanitized = text
        for pattern in injection_patterns:
            sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)

        # Truncate excessively long inputs (prevent context overflow attacks)
        max_length = 4000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "... [truncated]"
            logger.warning(f"Input truncated from {len(text)} to {max_length} characters")

        return sanitized


class RateLimiter:
    """
    Simple sliding-window rate limiter.
    Tracks request counts per user within a time window.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, user_id: str) -> bool:
        """Check if the user is within the rate limit."""
        now = time.time()
        cutoff = now - self.window_seconds

        # Remove expired timestamps
        self._requests[user_id] = [
            ts for ts in self._requests[user_id] if ts > cutoff
        ]

        if len(self._requests[user_id]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for user '{user_id}'")
            return False

        self._requests[user_id].append(now)
        return True
