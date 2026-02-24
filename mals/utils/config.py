"""
Configuration management for MALS.

Supports loading configuration from:
1. YAML configuration file (mals.yaml)
2. Environment variables (prefixed with MALS_)
3. Constructor arguments (highest priority)

Configuration hierarchy (highest to lowest priority):
    Constructor args > Environment variables > YAML file > Defaults
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("mals.config")


@dataclass
class LLMConfig:
    """Configuration for the LLM backend."""
    model: str = "gpt-4.1-mini"
    conductor_model: str = "gpt-4.1-nano"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.3
    max_retries: int = 3


@dataclass
class BlackboardConfig:
    """Configuration for the Dynamic Blackboard."""
    backend: str = "memory"  # "memory" or "redis"
    redis_url: str = "redis://localhost:6379/0"


@dataclass
class ConductorConfig:
    """Configuration for the Conductor Agent."""
    max_steps: int = 50
    consensus_max_iterations: int = 3


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


@dataclass
class MALSConfig:
    """Top-level configuration for the MALS system."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    blackboard: BlackboardConfig = field(default_factory=BlackboardConfig)
    conductor: ConductorConfig = field(default_factory=ConductorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "MALSConfig":
        """
        Load configuration from file and environment variables.

        Args:
            config_path: Path to a YAML configuration file.
                         If None, looks for 'mals.yaml' in the current directory.

        Returns:
            A fully resolved MALSConfig instance.
        """
        config = cls()

        # Step 1: Load from YAML file
        if config_path is None:
            config_path = Path("mals.yaml")
        else:
            config_path = Path(config_path)

        if config_path.exists():
            config = _load_yaml(config_path, config)
            logger.info("Loaded config from %s", config_path)

        # Step 2: Override with environment variables
        config = _apply_env_overrides(config)

        return config


def _load_yaml(path: Path, config: MALSConfig) -> MALSConfig:
    """Load configuration from a YAML file."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed. Skipping YAML config file.")
        return config

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # LLM config
    llm_data = data.get("llm", {})
    if llm_data:
        config.llm.model = llm_data.get("model", config.llm.model)
        config.llm.conductor_model = llm_data.get("conductor_model", config.llm.conductor_model)
        config.llm.api_key = llm_data.get("api_key", config.llm.api_key)
        config.llm.base_url = llm_data.get("base_url", config.llm.base_url)
        config.llm.temperature = llm_data.get("temperature", config.llm.temperature)

    # Blackboard config
    bb_data = data.get("blackboard", {})
    if bb_data:
        config.blackboard.backend = bb_data.get("backend", config.blackboard.backend)
        config.blackboard.redis_url = bb_data.get("redis_url", config.blackboard.redis_url)

    # Conductor config
    cond_data = data.get("conductor", {})
    if cond_data:
        config.conductor.max_steps = cond_data.get("max_steps", config.conductor.max_steps)
        config.conductor.consensus_max_iterations = cond_data.get(
            "consensus_max_iterations", config.conductor.consensus_max_iterations
        )

    # Logging config
    log_data = data.get("logging", {})
    if log_data:
        config.logging.level = log_data.get("level", config.logging.level)

    return config


def _apply_env_overrides(config: MALSConfig) -> MALSConfig:
    """Override configuration with environment variables."""
    # LLM
    if val := os.environ.get("MALS_LLM_MODEL"):
        config.llm.model = val
    if val := os.environ.get("MALS_CONDUCTOR_MODEL"):
        config.llm.conductor_model = val
    if val := os.environ.get("OPENAI_API_KEY"):
        config.llm.api_key = val
    if val := os.environ.get("OPENAI_BASE_URL"):
        config.llm.base_url = val

    # Blackboard
    if val := os.environ.get("MALS_BLACKBOARD_BACKEND"):
        config.blackboard.backend = val
    if val := os.environ.get("MALS_REDIS_URL"):
        config.blackboard.redis_url = val

    # Conductor
    if val := os.environ.get("MALS_MAX_STEPS"):
        config.conductor.max_steps = int(val)

    # Logging
    if val := os.environ.get("MALS_LOG_LEVEL"):
        config.logging.level = val

    return config
