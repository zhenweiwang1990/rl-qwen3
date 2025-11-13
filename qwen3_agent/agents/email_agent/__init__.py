"""Email search agent implementation."""

from .agent import EmailAgent
from .tasks import EmailTask
from .evaluator import EmailEvaluator, EmailRubric

__all__ = ["EmailAgent", "EmailTask", "EmailEvaluator", "EmailRubric"]

