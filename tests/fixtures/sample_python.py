"""Sample Python code for testing."""


def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers."""
    return a + b


def calculate_product(a: int, b: int) -> int:
    """Calculate product of two numbers."""
    return a * b


class Calculator:
    """Simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, x: float, y: float) -> float:
        """Add two numbers."""
        result = x + y
        self.history.append(f"{x} + {y} = {result}")
        return result

    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers."""
        result = x * y
        self.history.append(f"{x} * {y} = {result}")
        return result

    def get_history(self) -> list:
        """Get calculation history."""
        return self.history.copy()
