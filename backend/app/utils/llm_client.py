"""
LLM client wrapper.
Uses the Claude Agent SDK, authenticated via the user's Claude subscription.
"""

import asyncio
import json
import re
from typing import Optional, Dict, Any, List

import os

from claude_agent_sdk import query, ClaudeAgentOptions

from .logger import log_llm_interaction

# Default model, configurable via CLAUDE_MODEL env variable
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")


def _run_async(coro):
    """Run an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If we're already in an async context, create a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class LLMClient:
    """LLM client powered by Claude Agent SDK."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or DEFAULT_MODEL

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        should_log: bool = True
    ) -> str:
        """
        Send a chat request via Claude Agent SDK.

        Args:
            messages: Message list (system/user/assistant roles)
            temperature: Temperature parameter (best-effort)
            max_tokens: Maximum token count (best-effort)
            response_format: Response format hint (best-effort)
            should_log: Whether to save request/response to log file

        Returns:
            Model response text
        """
        system_prompt, prompt = self._convert_messages(messages)

        options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=system_prompt if system_prompt else None,
        )

        if response_format and response_format.get("type") == "json_object":
            options.output_format = {
                "type": "json_schema",
                "schema": {"type": "object"},
            }

        async def _run():
            result_text = ""
            async for message in query(prompt=prompt, options=options):
                if hasattr(message, "result") and message.result:
                    result_text = message.result
            return result_text

        content_raw = _run_async(_run())

        # Strip <think> tags (some models include reasoning traces)
        content_cleaned = re.sub(r'<think>[\s\S]*?</think>', '', content_raw).strip()

        if should_log:
            log_llm_interaction(
                source_file="llm_client.py",
                messages=messages,
                response_text=content_cleaned,
            )

        return content_cleaned

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send a chat request and return JSON.

        Args:
            messages: Message list
            temperature: Temperature parameter (best-effort)
            max_tokens: Maximum token count (best-effort)

        Returns:
            Parsed JSON object
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            should_log=False,
        )
        # Clean markdown code fence markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            parsed_json = json.loads(cleaned_response)
            log_llm_interaction(
                source_file="llm_client.py",
                messages=messages,
                response_text=cleaned_response,
            )
            return parsed_json
        except json.JSONDecodeError:
            log_llm_interaction(
                source_file="llm_client.py",
                messages=messages,
                response_text=cleaned_response,
            )
            raise ValueError(f"Invalid JSON returned by LLM: {cleaned_response}")

    @staticmethod
    def _convert_messages(messages: List[Dict[str, str]]) -> tuple:
        """
        Convert a messages list to (system_prompt, user_prompt) for the Agent SDK.
        """
        system_parts = []
        conversation_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                conversation_parts.append(f"[Assistant]: {content}")
            else:
                conversation_parts.append(content)

        system_prompt = "\n\n".join(system_parts) if system_parts else ""
        prompt = "\n\n".join(conversation_parts) if conversation_parts else ""

        return system_prompt, prompt
