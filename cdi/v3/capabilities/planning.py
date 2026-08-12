"""Typed bounded planning and dry-run execution interfaces for Stage F.

The implementation lives with the composition shell to keep every execution
path visibly constrained to the same audit and dry-run policy.
"""
from .orchestrator import Executor, Plan, PlanAction, Planner

__all__ = ["Executor", "Plan", "PlanAction", "Planner"]
