"""
Shared authentication utilities for MCP examples.

This module provides consistent authentication setup across all MCP examples,
supporting OAuth/CLI profiles, token authentication, and notebook authentication.
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


def setup_mlflow_for_profile(workspace_client: WorkspaceClient, profile: Optional[str] = None):
    """
    Configure MLflow tracking and registry for the given profile/workspace.
    
    Args:
        workspace_client: Configured WorkspaceClient
        profile: Optional CLI profile name
    """
    import mlflow
    
    # Get current user for experiment setup
    current_user = get_current_user(workspace_client)
    
    if profile:
        # Use profile-based URIs
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri(f"databricks-uc://{profile}")
        
        # Set environment variable for other Databricks SDK components
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    else:
        # Use default URIs
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    
    return current_user


def get_mcp_server_url(
    workspace_client: WorkspaceClient, 
    server_type: str = "system/ai",
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
    genie_space_id: Optional[str] = None
) -> str:
    """
    Build MCP server URL for the workspace based on server type.
    
    Args:
        workspace_client: Configured WorkspaceClient
        server_type: MCP server type. Options:
            - "system/ai" (default): System AI functions
            - "vector-search": Vector Search MCP server
            - "functions": UC Functions MCP server
            - "genie": Genie Space MCP server
        catalog: Catalog name (required for vector-search and functions)
        schema: Schema name (required for vector-search and functions)
        genie_space_id: Genie space ID (required for genie)
        
    Returns:
        Full MCP server URL
        
    Raises:
        ValueError: If required parameters are missing for the server type
    """
    workspace_hostname = workspace_client.config.host
    
    if server_type == "system/ai":
        return f"{workspace_hostname}/api/2.0/mcp/functions/system/ai"
    
    elif server_type == "vector-search":
        if not catalog or not schema:
            raise ValueError("catalog and schema are required for vector-search server type")
        return f"{workspace_hostname}/api/2.0/mcp/vector-search/{catalog}/{schema}"
    
    elif server_type == "functions":
        if not catalog or not schema:
            raise ValueError("catalog and schema are required for functions server type")
        return f"{workspace_hostname}/api/2.0/mcp/functions/{catalog}/{schema}"
    
    elif server_type == "genie":
        if not genie_space_id:
            raise ValueError("genie_space_id is required for genie server type")
        return f"{workspace_hostname}/api/2.0/mcp/genie/{genie_space_id}"
    
    else:
        raise ValueError(f"Unsupported server_type: {server_type}. "
                        f"Supported types: 'system/ai', 'vector-search', 'functions', 'genie'")


def create_mcp_client_with_auth(
    profile: Optional[str] = None, 
    server_type: str = "system/ai",
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
    genie_space_id: Optional[str] = None
):
    """
    Create an authenticated MCP client.
    
    Args:
        profile: Optional Databricks CLI profile name
        server_type: MCP server type
        catalog: Catalog name (required for vector-search and functions)
        schema: Schema name (required for vector-search and functions)
        genie_space_id: Genie space ID (required for genie)
        
    Returns:
        Tuple of (DatabricksMCPClient, WorkspaceClient)
    """
    from databricks_mcp import DatabricksMCPClient
    
    # Set up authentication
    workspace_client = setup_workspace_client(profile)
    
    # Build server URL
    server_url = get_mcp_server_url(
        workspace_client, 
        server_type, 
        catalog=catalog, 
        schema=schema, 
        genie_space_id=genie_space_id
    )
    
    # Create and return MCP client
    mcp_client = DatabricksMCPClient(
        server_url=server_url,
        workspace_client=workspace_client
    )
    
    return mcp_client, workspace_client