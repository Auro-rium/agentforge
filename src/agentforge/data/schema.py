"""Canonical training-row schema shared by every dataset normalizer.

This is the shape TRL's SFTTrainer expects for tool-calling conversational
datasets: a `messages` list (with `tool_calls`/`tool`-role turns) plus a
`tools` column of JSON-schema function definitions. Every normalizer maps its
source dataset's native format into a `Row`, so `train.py` can load
`manifest.jsonl` straight via `datasets.load_dataset("json", ...)` with no
adapter layer at train time.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Source = Literal["glaive", "hermes", "toolace", "xlam", "agent_flan"]
Role = Literal["system", "user", "assistant", "tool"]


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON-encoded string, OpenAI convention (not a dict)

    @field_validator("arguments")
    @classmethod
    def _arguments_must_be_json(cls, v: str) -> str:
        json.loads(v)  # raises ValueError if not valid JSON text
        return v


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _role_field_constraints(self) -> "Message":
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls is only valid on role='assistant'")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("role='tool' messages must set tool_call_id")
        if self.role != "tool" and self.name is not None:
            raise ValueError("name is only valid on role='tool'")
        return self


class ToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: dict  # {name, description, parameters: JSONSchema}

    @field_validator("function")
    @classmethod
    def _function_must_have_name(cls, v: dict) -> dict:
        if not v.get("name"):
            raise ValueError("tool function spec must include a non-empty 'name'")
        return v


class Row(BaseModel):
    id: str
    source: Source
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_row_shape(self) -> "Row":
        if not self.messages:
            raise ValueError("row must contain at least one message")
        if self.messages[0].role not in ("system", "user"):
            raise ValueError("first message must be role='system' or role='user'")

        open_call_ids: set[str] = set()
        for msg in self.messages:
            if msg.role == "assistant" and msg.tool_calls:
                open_call_ids.update(tc.id for tc in msg.tool_calls)
            elif msg.role == "tool":
                if msg.tool_call_id not in open_call_ids:
                    # Soft check: some source datasets have imperfect tool_call_id
                    # linkage. Never drop the row for this alone — just tag it so
                    # manifest_stats.json can report how common it is.
                    self.meta = {**self.meta, "orphan_tool_response": True}
        return self
