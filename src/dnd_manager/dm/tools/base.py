"""Base classes for DM tools.

Tools are defined by Python and called by the LLM.
The LLM provides arguments, Python executes.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field

from dnd_manager.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Tool Base Classes
# =============================================================================


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_name: str
    success: bool
    result: str
    data: dict[str, Any] = Field(default_factory=dict)


class DMTool:
    """A tool that the DM can invoke.
    
    Tools are defined by Python and called by the LLM.
    The LLM provides arguments, Python executes.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., ToolResult],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""
        try:
            return self.handler(**kwargs)
        except Exception as exc:
            logger.exception(f"Tool {self.name} failed")
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=f"Error: {exc}",
            )
