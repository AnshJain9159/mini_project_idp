"""Custom exception classes for the diabetes prediction system."""


class DiabetesPredictorError(Exception):
    """Base exception for diabetes predictor errors."""
    pass


class DataValidationError(DiabetesPredictorError):
    """Raised when data validation fails."""
    pass


class ModelError(DiabetesPredictorError):
    """Raised when model-related errors occur."""
    pass


class PDFExtractionError(DiabetesPredictorError):
    """Raised when PDF extraction fails."""
    pass


class ConfigurationError(DiabetesPredictorError):
    """Raised when configuration errors occur."""
    pass


class DatabaseError(DiabetesPredictorError):
    """Raised when database operations fail."""
    pass


class APIError(DiabetesPredictorError):
    """Raised when API errors occur."""
    pass
