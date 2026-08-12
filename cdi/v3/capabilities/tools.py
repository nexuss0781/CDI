"""Typed local dry-run tool registry for bounded Stage F diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
from typing import Any, Callable, Dict, Mapping

from .audit import AuditTrail


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]
    timeout_ms: int
    permission: str
    mode: str = "dry_run"
    version: str = "v1"


@dataclass(frozen=True)
class ToolResult:
    status: str
    output: Dict[str, Any]
    reason: str | None = None


class ToolRegistry:
    """Registry that refuses external/mutating execution under its dry-run policy."""

    ALLOWED_PERMISSIONS = {"read_only", "simulated"}
    DECLARED_PERMISSIONS = {"read_only", "simulated", "mutating"}

    def __init__(self, audit: AuditTrail | None = None, dry_run: bool = True) -> None:
        self.audit = audit or AuditTrail()
        self.dry_run = dry_run
        self._definitions: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[[Mapping[str, Any]], Dict[str, Any]]] = {}

    def register(self, definition: ToolDefinition, handler: Callable[[Mapping[str, Any]], Dict[str, Any]]) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Tool {definition.name!r} is already registered.")
        if definition.permission not in self.DECLARED_PERMISSIONS:
            raise ValueError("Tool permission must be a declared local diagnostic permission.")
        if definition.mode != "dry_run" or not self.dry_run:
            raise ValueError("Stage F tool registration is dry-run only.")
        if definition.timeout_ms <= 0:
            raise ValueError("Tool timeout must be positive.")
        self._definitions[definition.name] = definition
        self._handlers[definition.name] = handler
        self.audit.append("tool_register", {"definition": asdict(definition)})

    def validate(self, name: str, payload: Mapping[str, Any]) -> ToolResult:
        definition = self._definitions.get(name)
        if definition is None:
            result = ToolResult("REJECTED", {}, "unregistered_tool")
        elif definition.permission not in self.ALLOWED_PERMISSIONS or definition.mode != "dry_run":
            result = ToolResult("REJECTED", {}, "permission_or_mode_denied")
        elif set(payload).difference({"simulated_delay_ms"}) != set(definition.input_schema):
            result = ToolResult("REJECTED", {}, "schema_keys_mismatch")
        elif any(not isinstance(payload[key], _type_for(schema)) for key, schema in definition.input_schema.items()) or not isinstance(payload.get("simulated_delay_ms", 0), int):
            result = ToolResult("REJECTED", {}, "schema_type_mismatch")
        elif int(payload.get("simulated_delay_ms", 0)) > definition.timeout_ms:
            result = ToolResult("TIMEOUT", {}, "simulated_timeout")
        else:
            result = ToolResult("VALID", {})
        self.audit.append("tool_validate", {"name": name, "payload": dict(payload), "status": result.status, "reason": result.reason})
        return result

    def invoke(self, name: str, payload: Mapping[str, Any]) -> ToolResult:
        validation = self.validate(name, payload)
        if validation.status != "VALID":
            self.audit.append("tool_invoke_blocked", {"name": name, "status": validation.status, "reason": validation.reason})
            return validation
        try:
            output = self._handlers[name](payload)
        except Exception as exc:
            result = ToolResult("ERROR", {}, f"handler_error:{type(exc).__name__}")
        else:
            result = ToolResult("OK", output)
        self.audit.append("tool_invoke", {"name": name, "payload": dict(payload), "status": result.status, "output": result.output, "reason": result.reason})
        return result

    def definitions(self) -> Dict[str, ToolDefinition]:
        return dict(self._definitions)


def _type_for(name: str):
    return {"str": str, "int": int, "float": (int, float), "bool": bool}.get(name, object)


def _safe_arithmetic(expression: str) -> float:
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.USub, ast.UAdd, ast.Constant, ast.Pow)
    tree = ast.parse(expression, mode="eval")
    if not all(isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError("unsupported_arithmetic_expression")
    return float(_evaluate(tree.body))


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _evaluate(node.operand)
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.FloorDiv): return left // right
        if isinstance(node.op, ast.Pow): return left ** right
    raise ValueError("unsupported_arithmetic_expression")


def register_default_tools(registry: ToolRegistry, lookup: Callable[[str], Dict[str, Any]]) -> ToolRegistry:
    registry.register(ToolDefinition("arithmetic", {"expression": "str"}, {"value": "float"}, timeout_ms=100, permission="read_only"), lambda payload: {"value": _safe_arithmetic(str(payload["expression"]))})
    registry.register(ToolDefinition("local_lookup", {"query": "str"}, {"result": "str"}, timeout_ms=100, permission="read_only"), lambda payload: lookup(str(payload["query"])))
    return registry
