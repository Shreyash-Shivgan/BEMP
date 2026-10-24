"""Typed exception hierarchy for StrikeShield.

All custom errors inherit from StrikeShieldError and support uniform serialization
of context payloads for JSON manifests and SQLite audit records.
"""

from typing import Any, Dict, Optional


class StrikeShieldError(Exception):
    """Base exception for all StrikeShield errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize error details for run manifests and audit tables."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
        }

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | Context: {self.context}"
        return self.message


class JudgeSchemaError(StrikeShieldError):
    """Raised when Judge LLM output fails schema validation or is malformed."""
    pass


class EvidenceCheckError(StrikeShieldError):
    """Raised or logged when quoted evidence does not appear in the model response."""
    pass


class JudgeUnvalidatedWarning(UserWarning):
    """Warning raised when Cohen's kappa < 0.6 between judge and human labels."""
    pass


class PatcherValidationError(StrikeShieldError):
    """Raised when a candidate patch fails structural or domain validation."""
    pass


class PatcherTooWeakError(StrikeShieldError):
    """Raised when Patcher validation fails repeatedly, triggering model escalation."""
    pass


class GateRejectedError(StrikeShieldError):
    """Raised when a candidate policy fails to satisfy gate acceptance criteria."""
    pass


class VRAMStillAllocatedError(StrikeShieldError):
    """Raised when a model fails to unload from VRAM, violating the single-model invariant."""
    pass


class MockMissError(StrikeShieldError):
    """Raised in mock mode when a required fixture is missing."""
    pass


class PolicyTreeDirtyError(StrikeShieldError):
    """Raised when Git working tree is dirty before policy operations."""
    pass


class HeldoutLeakError(StrikeShieldError):
    """Raised if a held-out attack sample is passed to the Patcher."""
    pass


class PolicyPathError(StrikeShieldError):
    """Raised when an invalid or disallowed policy path (e.g. from policies/candidates/) is loaded."""
    pass


class MixedCorpusError(StrikeShieldError):
    """Raised when an operation or dashboard view attempts to mix smoke and real corpus data."""
    pass

