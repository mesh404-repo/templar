"""Custom exceptions for templar-crusades."""

from enum import StrEnum


class EvaluationErrorCode(StrEnum):
    """Structured error codes for evaluation failures.

    Use these codes instead of string matching for robust error handling.
    """

    # Code validation errors
    NO_CODE = "no_code"
    SYNTAX_ERROR = "syntax_error"
    MISSING_INNER_STEPS = "missing_inner_steps"
    INVALID_RETURN_TYPE = "invalid_return_type"

    # Verification failures (anti-cheat)
    INSUFFICIENT_TRAINABLE_PARAMS = "insufficient_trainable_params"
    INSUFFICIENT_PARAMS_CHANGED = "insufficient_params_changed"
    GRADIENT_COVERAGE_FAILED = "gradient_coverage_failed"
    GRADIENT_NORM_RATIO_FAILED = "gradient_norm_ratio_failed"
    GRADIENT_COSINE_FAILED = "gradient_cosine_failed"
    LOSS_MISMATCH = "loss_mismatch"
    TOKEN_COUNT_MISMATCH = "token_count_mismatch"

    # Runtime errors
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    WARMUP_FAILED = "warmup_failed"
    EXECUTION_FAILED = "execution_failed"

    # Infrastructure errors
    MODEL_LOAD_FAILED = "model_load_failed"
    DATA_LOAD_FAILED = "data_load_failed"
    DOCKER_FAILED = "docker_failed"

    # Unknown
    UNKNOWN = "unknown"

    @classmethod
    def is_verification_failure(cls, code: "EvaluationErrorCode") -> bool:
        """Check if error code indicates a verification/anti-cheat failure."""
        return code in {
            cls.INSUFFICIENT_TRAINABLE_PARAMS,
            cls.INSUFFICIENT_PARAMS_CHANGED,
            cls.GRADIENT_COVERAGE_FAILED,
            cls.GRADIENT_NORM_RATIO_FAILED,
            cls.GRADIENT_COSINE_FAILED,
            cls.LOSS_MISMATCH,
            cls.TOKEN_COUNT_MISMATCH,
        }

    @classmethod
    def is_miner_fault(cls, code: "EvaluationErrorCode") -> bool:
        """Check if error is likely the miner's fault (vs infrastructure)."""
        return code not in {
            cls.MODEL_LOAD_FAILED,
            cls.DATA_LOAD_FAILED,
            cls.DOCKER_FAILED,
            cls.TIMEOUT,  # Could be either
        }


class CrusadesError(Exception):
    """Base exception for all crusades errors."""

    pass


class SandboxError(CrusadesError):
    """Error during sandbox execution."""

    pass


class SandboxTimeoutError(SandboxError):
    """Sandbox execution timed out."""

    pass


class SandboxCrashError(SandboxError):
    """Code crashed during sandbox execution."""

    pass


class EvaluationError(CrusadesError):
    """Error during evaluation with structured error code."""

    def __init__(self, message: str, code: EvaluationErrorCode = EvaluationErrorCode.UNKNOWN):
        super().__init__(message)
        self.code = code
        self.message = message


class ValidationError(CrusadesError):
    """Code validation failed."""

    pass


class StorageError(CrusadesError):
    """Error accessing storage (database)."""

    pass


class ChainError(CrusadesError):
    """Error interacting with Bittensor chain."""

    pass
