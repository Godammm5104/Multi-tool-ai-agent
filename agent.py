from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

DEFAULT_BASE_URL = "https://api.intelligence.io.solutions/api/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning"

SYSTEM_PROMPT = """You are a careful multi-tool assistant.
Use a tool when it will improve accuracy or save the user time.
Never invent tool results. If a tool fails, explain the failure briefly.
Keep the final answer concise and practical."""


@dataclass
class ToolCallLog:
    step: int
    name: str
    arguments: dict[str, Any]
    result: str
    error: str | None = None


class MultiToolAgent:
    """A small tool-using agent with a bounded reasoning loop."""

    def __init__(
        self,
        client=None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_steps: int = 4,
        temperature: float = 0.2,
    ) -> None:
        self.model = model or os.getenv("IO_MODEL", DEFAULT_MODEL)
        self.max_steps = max(1, max_steps)
        self.temperature = temperature
        self._client = client or OpenAI(
            base_url=base_url or os.getenv("IO_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key or os.getenv("IO_API_KEY"),
        )
        self.trace: list[ToolCallLog] = []

    def reset(self) -> None:
        self.trace = []

    def _create(self, messages: list[dict[str, Any]]):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5**attempt)
        raise RuntimeError(f"LLM request failed: {last_error}") from last_error

    def run(self, user_message: str) -> str:
        if not user_message.strip():
            raise ValueError("Message cannot be empty.")

        self.reset()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        for step in range(1, self.max_steps + 1):
            response = self._create(messages)
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                return message.content or "No response."

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )

            for call in tool_calls:
                name = call.function.name
                raw_arguments = call.function.arguments or "{}"

                try:
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    result = f"Error: invalid arguments for {name}: {exc}"
                    self.trace.append(
                        ToolCallLog(
                            step=step,
                            name=name,
                            arguments={},
                            result=result,
                            error=str(exc),
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        }
                    )
                    continue

                function = TOOL_FUNCTIONS.get(name)
                if function is None:
                    result = f"Error: unknown tool '{name}'"
                    self.trace.append(
                        ToolCallLog(
                            step=step,
                            name=name,
                            arguments=arguments,
                            result=result,
                            error="unknown tool",
                        )
                    )
                else:
                    try:
                        result = str(function(**arguments))
                        self.trace.append(
                            ToolCallLog(
                                step=step,
                                name=name,
                                arguments=arguments,
                                result=result,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        result = f"Error while running {name}: {exc}"
                        self.trace.append(
                            ToolCallLog(
                                step=step,
                                name=name,
                                arguments=arguments,
                                result=result,
                                error=str(exc),
                            )
                        )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        return "I hit the step limit before finishing. Try narrowing the request."
