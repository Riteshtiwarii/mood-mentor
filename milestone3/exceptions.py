"""Custom domain exceptions for Mood Mentor Milestone 3."""


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


class IntensityAnalysisError(MoodMentorError):
    """Raised when emotion intensity or severity calculation fails (Task 1)."""

    pass


class UserProfileError(MoodMentorError):
    """Raised when user preference or profile tracking fails (Task 2)."""

    pass


class RecommendationError(MoodMentorError):
    """Raised when hybrid recommendation or dynamic ranking fails (Task 3 & 4)."""

    pass


class WellnessCatalogError(MoodMentorError):
    """Raised when wellness content catalog parsing or indexing fails (Task 5)."""

    pass


class SemanticMatchingError(MoodMentorError):
    """Raised when semantic embedding generation or cosine matching fails (Task 5)."""

    pass


class FeedbackError(MoodMentorError):
    """Raised when processing or recording user feedback fails (Task 7)."""

    pass


class ExplainabilityError(MoodMentorError):
    """Raised when generating recommendation explanations fails (Task 8)."""

    pass


class RecommenderEvaluationError(MoodMentorError):
    """Raised when evaluating recommendation metrics (Precision@K, NDCG) fails (Task 9)."""

    pass


class ReportGenerationError(MoodMentorError):
    """Raised when generating tabular, summary, or dataframe reports fails."""

    pass
