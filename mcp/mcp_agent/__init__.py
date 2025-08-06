"""
Databricks MCP Agent - Authentication Utilities

Provides authentication utilities for Databricks MCP ResponsesAgent.
"""

from .auth import setup_workspace_client, get_current_user

__version__ = "0.1.0"
__all__ = [
    "setup_workspace_client",
    "get_current_user"
]