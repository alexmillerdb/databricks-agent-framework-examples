"""
Authentication utilities for Databricks MCP Agent.

Provides authentication methods for WorkspaceClient and user management.
"""

import os
import logging
from typing import Optional

from databricks.sdk import WorkspaceClient

# Optional imports for environment setup
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)


def setup_workspace_client(profile: Optional[str] = None, verbose: bool = True) -> WorkspaceClient:
    """
    Set up WorkspaceClient for authentication.
    
    Args:
        profile: Optional Databricks CLI profile name
        verbose: Whether to print authentication information
        
    Returns:
        Configured WorkspaceClient instance
    """
    # Load environment variables if available
    if load_dotenv:
        load_dotenv()
    
    # Check authentication method
    if os.getenv('DATABRICKS_TOKEN') and os.getenv('DATABRICKS_HOST'):
        if verbose:
            logger.info("🏠 Local Development Mode (Token Auth)")
            logger.info(f"✅ Host: {os.getenv('DATABRICKS_HOST')}")
            logger.info("✅ Using token authentication")
        # Token auth from environment
        return WorkspaceClient()
    elif profile:
        if verbose:
            logger.info(f"🔐 Using Databricks CLI profile: {profile}")
        return WorkspaceClient(profile=profile)
    else:
        if verbose:
            logger.info("☁️ Databricks Environment Mode")
            logger.info("ℹ️ Using default authentication")
        # Default auth (works in Databricks notebooks)
        return WorkspaceClient()


def get_current_user(workspace_client: WorkspaceClient) -> str:
    """Get current user from workspace."""
    try:
        if _is_databricks_notebook():
            # In notebook, try to use spark context if available
            try:
                return spark.sql("SELECT current_user()").collect()[0][0]  # type: ignore
            except NameError:
                # spark not available, fall back to workspace API
                pass
        
        # Use workspace API (works in all environments)
        current_user = workspace_client.current_user.me()
        return current_user.user_name
    except Exception as e:
        logger.warning(f"Could not get user: {e}")
        return "unknown_user"


def _is_databricks_notebook() -> bool:
    """Check if we're running in a Databricks notebook."""
    try:
        dbutils  # type: ignore
        return True
    except NameError:
        return False