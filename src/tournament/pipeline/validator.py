"""Code validation before sandbox execution."""

import ast
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of code validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)


class CodeValidator:
    """Validates miner training code before execution.

    Checks:
    1. Syntax validity (can be parsed)
    2. Required function exists (inner_steps)
    3. No forbidden imports (optional, configurable)
    """

    REQUIRED_FUNCTIONS = {"inner_steps"}

    # Imports that are explicitly forbidden (security risk)
    FORBIDDEN_IMPORTS = {
        "os",
        "subprocess",
        "socket",
        "http",
        "urllib",
        "requests",
        "ftplib",
        "telnetlib",
        "smtplib",
        "poplib",
        "imaplib",
        "ctypes",
        "multiprocessing",
        "signal",
        "pty",
        "fcntl",
        "resource",
        "pickle",
        "marshal",
    }

    def __init__(self, check_imports: bool = True):
        self.check_imports = check_imports

    def validate(self, code: str) -> ValidationResult:
        """Validate code string."""
        errors: list[str] = []

        # Check syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                errors=[f"Syntax error at line {e.lineno}: {e.msg}"],
            )

        # Check required functions
        defined_functions = self._get_defined_functions(tree)
        missing = self.REQUIRED_FUNCTIONS - defined_functions
        if missing:
            errors.append(f"Missing required functions: {', '.join(sorted(missing))}")

        # Check forbidden imports
        if self.check_imports:
            forbidden = self._get_forbidden_imports(tree)
            if forbidden:
                errors.append(f"Forbidden imports: {', '.join(sorted(forbidden))}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
        )

    def validate_file(self, path: str) -> ValidationResult:
        """Validate code from file path."""
        try:
            with open(path) as f:
                code = f.read()
        except FileNotFoundError:
            return ValidationResult(valid=False, errors=[f"File not found: {path}"])
        except Exception as e:
            return ValidationResult(valid=False, errors=[f"Error reading file: {e}"])

        return self.validate(code)

    def _get_defined_functions(self, tree: ast.AST) -> set[str]:
        """Extract top-level function names from AST."""
        functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.add(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.add(node.name)
        return functions

    def _get_forbidden_imports(self, tree: ast.AST) -> set[str]:
        """Find any forbidden imports in the AST."""
        forbidden_found = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in self.FORBIDDEN_IMPORTS:
                        forbidden_found.add(module)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in self.FORBIDDEN_IMPORTS:
                        forbidden_found.add(module)

        return forbidden_found
