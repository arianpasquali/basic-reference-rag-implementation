"""
Application settings and configuration management using Pydantic.

This module provides a centralized configuration system that loads settings
from environment variables with proper validation, type checking, and defaults.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables with fallback defaults.

    This class uses Pydantic BaseSettings to automatically load configuration
    from environment variables, with proper type validation and documentation.
    """

    # API Configuration
    OPENAI_API_KEY: str = Field(
        ..., description="OpenAI API key for accessing GPT models", env="OPENAI_API_KEY"
    )

    # Model Configuration
    DEFAULT_MODEL: str = Field(
        default="openai/gpt-4.1-mini",
        description="Default language model to use for AI operations",
        env="DEFAULT_MODEL",
    )

    JUDGE_MODEL: str = Field(
        default="openai/gpt-4.1-mini",
        description="Default language model to use as LLM-as-a-Judge operations",
        env="JUDGE_MODEL",
    )

    # Database Configuration
    DEFAULT_DB_PATH: Path = Field(
        default=Path("./lancedb"),
        description="Default path for LanceDB vector database",
        env="DEFAULT_DB_PATH",
    )

    DEFAULT_SQLITE_PATH: Path = Field(
        default=Path("./toyota_sales.db"),
        description="Default path for SQLite database",
        env="DEFAULT_SQLITE_PATH",
    )

    # Optional Observability Configuration
    LANGSMITH_PROJECT: Optional[str] = Field(
        default=None,
        description="LangSmith project name for observability",
        env="LANGSMITH_PROJECT",
    )

    LANGCHAIN_TRACING_V2: bool = Field(
        default=False, description="Enable LangChain tracing v2", env="LANGCHAIN_TRACING_V2"
    )

    LANGCHAIN_PROJECT: Optional[str] = Field(
        default=None, description="LangChain project name", env="LANGCHAIN_PROJECT"
    )

    LANGSMITH_API_KEY: Optional[str] = Field(
        default=None, description="LangSmith API key for observability", env="LANGSMITH_API_KEY"
    )

    # Application Configuration
    APP_NAME: str = Field(
        default="Toyota RAG Assistant",
        description="Application name for display purposes",
        env="APP_NAME",
    )

    ENVIRONMENT: str = Field(
        default="development",
        description="Application environment (development, staging, production)",
        env="ENVIRONMENT",
    )

    DEBUG: bool = Field(default=False, description="Enable debug mode", env="DEBUG")

    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host address", env="HOST")

    PORT: int = Field(default=8000, description="Server port number", env="PORT")

    @field_validator("DEFAULT_DB_PATH", "DEFAULT_SQLITE_PATH", mode="before")
    @classmethod
    def convert_path_strings(cls, value):
        """Convert string paths to Path objects."""
        if isinstance(value, str):
            return Path(value)
        return value

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, value):
        """Validate environment is one of the allowed values."""
        allowed_environments = {"development", "staging", "production", "test"}
        if value.lower() not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return value.lower()

    model_config = {
        "env_file": ".env",
    }


# Global settings instance
# This is the main way to access settings throughout the application
settings = Settings()


# def get_settings() -> Settings:
#     """
#     Get the global settings instance.

#     This function provides a way to access settings that can be easily
#     mocked in tests or overridden in specific contexts.

#     Returns:
#         Settings: The global settings instance
#     """
#     return settings


# def reload_settings() -> Settings:
#     """
#     Reload settings from environment variables.

#     This is useful for testing or when environment variables change
#     during runtime.

#     Returns:
#         Settings: A new settings instance with current environment values
#     """
#     global settings
#     settings = Settings()
#     return settings


# # Convenience function for common database paths
# def get_project_root() -> Path:
#     """
#     Get the project root directory.

#     Returns:
#         Path: The project root directory
#     """
#     current_file = Path(__file__)
#     # Go up from src/core/settings.py to project root
#     return current_file.parent.parent.parent


# def get_absolute_db_path() -> Path:
#     """
#     Get the absolute path to the default database relative to project root.

#     Returns:
#         Path: Absolute path to the database
#     """
#     if settings.DEFAULT_DB_PATH.is_absolute():
#         return settings.DEFAULT_DB_PATH
#     return get_project_root() / settings.DEFAULT_DB_PATH


# def get_absolute_sqlite_path() -> Path:
#     """
#     Get the absolute path to the SQLite database relative to project root.

#     Returns:
#         Path: Absolute path to the SQLite database
#     """
#     if settings.DEFAULT_SQLITE_PATH.is_absolute():
#         return settings.DEFAULT_SQLITE_PATH
#     return get_project_root() / settings.DEFAULT_SQLITE_PATH
