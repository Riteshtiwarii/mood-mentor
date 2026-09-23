"""Custom domain exceptions for Mood Mentor Milestone 2."""


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


class EmotionModelError(MoodMentorError):
    """Raised when a Transformer emotion model fails to load, initialize, or run inference."""

    pass


class TrainingError(MoodMentorError):
    """Raised when training or fine-tuning of BERT/DistilBERT fails."""

    pass


class EvaluationError(MoodMentorError):
    """Raised when computing accuracy, precision, recall, or F1 metrics fails."""

    pass


class BenchmarkError(MoodMentorError):
    """Raised when running or parsing the ISEAR benchmark validation fails."""

    pass


class ReportGenerationError(MoodMentorError):
    """Raised when generating tabular, summary, or dataframe reports fails."""

    pass
