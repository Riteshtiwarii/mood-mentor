"""Custom domain exceptions for Mood Mentor Milestone 1."""


class MoodMentorError(Exception):
    """Base exception for all Mood Mentor application errors."""

    pass


class IngestionError(MoodMentorError):
    """Raised when an input file, format, or direct text fails validation or ingestion."""

    pass


class PreprocessingError(MoodMentorError):
    """Raised when text preprocessing fails or receives invalid input structure."""

    pass


class SentimentAnalysisError(MoodMentorError):
    """Raised when sentiment calculation or lexicon initialization fails."""

    pass


class ReportGenerationError(MoodMentorError):
    """Raised when generating tabular, summary, or dataframe reports fails."""

    pass
