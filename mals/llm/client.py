"""
LLM Client — OpenAI-compatible interface layer.

Provides a unified async interface for calling LLMs. Supports any provider
that exposes an OpenAI-compatible API (OpenAI, Azure, Anthropic via proxy,
local models via Ollama/vLLM, etc.).

Usage:
    client = LLMClient(model="gpt-4.1-mini")
    response = await client.complete(
        system_prompt="You are a helpful assistant.",
        user_prompt="Hello!",
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("mals.llm")


@dataclass
class LLMUsage:
    """Token usage statistics from a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Structured response from an LLM call, including content and usage."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class LLMClient:
    """
    Async LLM client with OpenAI-compatible API support.

    Supports configuration via environment variables or constructor arguments:
    - OPENAI_API_KEY: API key (required)
    - OPENAI_BASE_URL: Base URL for the API (optional, for custom endpoints)
    - MALS_LLM_MODEL: Default model name (optional)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "LLMClient requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from e

        self.model = model or os.environ.get("MALS_LLM_MODEL", "gpt-4.1-mini")
        self.default_temperature = temperature

        # Build client kwargs
        client_kwargs: dict = {"max_retries": max_retries}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)
        self._total_usage = LLMUsage()
        self._last_usage = LLMUsage()

        logger.info("LLMClient initialized: model=%s", self.model)

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int = 2000,
        temperature: float | None = None,
    ) -> str:
        """
        Send a completion request to the LLM.

        Returns:
            The LLM's response text.
        """
        resp = await self.complete_with_usage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.content

    async def complete_with_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int = 2000,
        temperature: float | None = None,
    ) -> LLMResponse:
        """
        Send a completion request and return both content and token usage.

        Args:
            system_prompt: The system message that sets the agent's behavior.
            user_prompt: The user message containing the task or question.
            model: Override the default model for this call.
            max_tokens: Maximum tokens in the response.
            temperature: Override the default temperature for this call.

        Returns:
            An LLMResponse with content and per-call token usage.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature if temperature is not None else self.default_temperature,
            )

            input_tok = response.usage.prompt_tokens if response.usage else 0
            output_tok = response.usage.completion_tokens if response.usage else 0
            total_tok = response.usage.total_tokens if response.usage else 0

            # Track cumulative usage
            self._total_usage.input_tokens += input_tok
            self._total_usage.output_tokens += output_tok
            self._total_usage.total_tokens += total_tok

            # Store last call usage for easy access
            self._last_usage = LLMUsage(
                input_tokens=input_tok,
                output_tokens=output_tok,
                total_tokens=total_tok,
            )

            content = response.choices[0].message.content or ""
            logger.debug(
                "LLM call: model=%s, input=%d, output=%d tokens",
                model or self.model, input_tok, output_tok,
            )
            return LLMResponse(
                content=content,
                input_tokens=input_tok,
                output_tokens=output_tok,
                total_tokens=total_tok,
            )

        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise

    @property
    def total_usage(self) -> LLMUsage:
        """Return cumulative token usage across all calls."""
        return self._total_usage

    def reset_usage(self) -> None:
        """Reset the cumulative token usage counter."""
        self._total_usage = LLMUsage()
