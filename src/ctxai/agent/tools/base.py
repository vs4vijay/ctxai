"""
Base tool classes for agent.

This module defines the abstract interface that all tools must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolParameterType(str, Enum):
    """Tool parameter types."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


@dataclass
class ToolParameter:
    """Represents a tool parameter."""
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None
    properties: Optional[Dict[str, "ToolParameter"]] = None  # For object type
    items: Optional["ToolParameter"] = None  # For array type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        param_dict = {
            "type": self.type.value,
            "description": self.description,
        }

        if self.enum:
            param_dict["enum"] = self.enum

        if self.default is not None:
            param_dict["default"] = self.default

        if self.properties:
            param_dict["properties"] = {
                name: prop.to_dict() for name, prop in self.properties.items()
            }

        if self.items:
            param_dict["items"] = self.items.to_dict()

        return param_dict


@dataclass
class ToolSchema:
    """Schema definition for a tool."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function schema format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_dict()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool schema format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_dict()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to generic dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    **p.to_dict(),
                    "required": p.required,
                }
                for p in self.parameters
            ],
        }


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Tools are capabilities that the agent can use to interact with
    the environment (files, bash, web, etc.).
    """

    def __init__(self):
        """Initialize the tool."""
        self.name = self._get_tool_name()

    def _get_tool_name(self) -> str:
        """
        Get tool name from class name.

        Converts 'ReadFileTool' -> 'read_file'
        """
        class_name = self.__class__.__name__
        # Remove 'Tool' suffix
        if class_name.endswith('Tool'):
            class_name = class_name[:-4]

        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

        return name

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """
        Get the tool schema.

        Returns:
            ToolSchema describing the tool and its parameters
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool.

        Args:
            **kwargs: Tool parameters

        Returns:
            Dictionary with execution results. Should include:
            - success: bool - Whether execution succeeded
            - result: Any - The result data
            - error: Optional[str] - Error message if failed

        Raises:
            Exception: If execution fails critically
        """
        pass

    def validate_parameters(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate parameters before execution.

        Args:
            **kwargs: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        schema = self.get_schema()

        # Check required parameters
        for param in schema.parameters:
            if param.required and param.name not in kwargs:
                return False, f"Missing required parameter: {param.name}"

            # Check enum values
            if param.enum and param.name in kwargs:
                value = kwargs[param.name]
                if value not in param.enum:
                    return False, f"Invalid value for {param.name}. Must be one of: {param.enum}"

        return True, None

    async def safe_execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute tool with parameter validation and error handling.

        Args:
            **kwargs: Tool parameters

        Returns:
            Execution result dictionary
        """
        # Validate parameters
        is_valid, error = self.validate_parameters(**kwargs)
        if not is_valid:
            return {
                "success": False,
                "result": None,
                "error": error,
            }

        try:
            result = await self.execute(**kwargs)
            return result
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"Tool execution failed: {str(e)}",
            }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"
