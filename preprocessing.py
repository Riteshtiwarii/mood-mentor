"""Text Preprocessing Module for Mood Mentor (Milestone 1).

Applies noise filtering, tokenization, sentiment-negation-preserving stopword
filtering, and lemmatization with clean dataclass representations and error handling.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import logging
import re
from typing import Sequence, Set

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from config import NEGATION_WORDS, REQUIRED_NLTK_PACKAGES
from exceptions import PreprocessingError

logger = logging.getLogger(__name__)

# Compile regex patterns once for high-performance noise filtering
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<.*?>")
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
SPECIAL_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9\s\'-]")
REPEATED_SPACES_PATTERN = re.compile(r"\s+")
PUNCTUATION_STRIP_PATTERN = re.compile(r"^[\'\-]+|[\'\-]+$")


def _ensure_nltk_resources() -> None:
    """Download required NLTK corpora if not already available locally."""
    for resource in REQUIRED_NLTK_PACKAGES:
        try:
            # Check availability
            if resource == "vader_lexicon":
                nltk.data.find("sentiment/vader_lexicon.zip")
            elif resource == "stopwords":
                nltk.data.find("corpora/stopwords.zip")
            elif resource == "wordnet":
                nltk.data.find("corpora/wordnet.zip")
            elif resource in ("punkt", "punkt_tab"):
                nltk.data.find(f"tokenizers/{resource}")
        except (LookupError, FileNotFoundError):
            logger.info("Downloading required NLTK resource '%s'...", resource)
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:
                logger.warning(
                    "Could not auto-download NLTK resource '%s': %s", resource, exc
                )


_ensure_nltk_resources()


@dataclass(frozen=True)
class PreprocessedResult:
    """Immutable container holding the stages of preprocessed text."""

    raw_text: str
    cleaned_text: str
    tokens: list[str]
    lemmas: list[str]
    token_count: int

    def reconstructed_text(self) -> str:
        """Return the preprocessed lemmatized tokens as a single coherent string."""
        return " ".join(self.lemmas)


class TextPreprocessor:
    """Thread-safe, modular text preprocessor for Employee Sentiment Analysis."""

    def __init__(
        self,
        negation_whitelist: Set[str] = NEGATION_WORDS,
        preserve_negations: bool = True,
    ) -> None:
        """Initialize preprocessor with stopword lists and lemmatizer."""
        self.preserve_negations = preserve_negations
        self.negation_whitelist = {w.lower() for w in negation_whitelist}
        self.lemmatizer = WordNetLemmatizer()

        try:
            raw_stopwords = set(stopwords.words("english"))
        except Exception:
            # Fallback standard English stopword set if NLTK data unavailable
            raw_stopwords = {
                "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
                "your", "yours", "yourself", "yourselves", "he", "him", "his",
                "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                "they", "them", "their", "theirs", "themselves", "what", "which",
                "who", "whom", "this", "that", "these", "those", "am", "is", "are",
                "was", "were", "be", "been", "being", "have", "has", "had", "having",
                "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
                "or", "because", "as", "until", "while", "of", "at", "by", "for",
                "with", "about", "against", "between", "into", "through", "during",
                "before", "after", "above", "below", "to", "from", "up", "down",
                "in", "out", "on", "off", "over", "under", "again", "further", "then",
                "once", "here", "there", "when", "where", "why", "how", "all", "any",
                "both", "each", "few", "more", "most", "other", "some", "such",
                "no", "nor", "not", "only", "own", "same", "so", "than", "too",
                "very", "s", "t", "can", "will", "just", "don", "should", "now"
            }

        if self.preserve_negations:
            # Remove negation modifiers from stopwords to preserve sentiment integrity
            self.filtered_stopwords = raw_stopwords - self.negation_whitelist
        else:
            self.filtered_stopwords = raw_stopwords

    def clean_noise(self, text: str) -> str:
        """Strip URLs, HTML tags, emails, special symbols, and excess spaces.

        Args:
            text: Raw input text.

        Returns:
            Normalized and cleaned string in lower case.
        """
        if not text:
            return ""

        # Unescape HTML entities (e.g. &amp; -> &)
        text = html.unescape(text)

        # Remove URLs and email addresses
        text = URL_PATTERN.sub(" ", text)
        text = EMAIL_PATTERN.sub(" ", text)

        # Remove HTML tags
        text = HTML_TAG_PATTERN.sub(" ", text)

        # Remove unwanted special characters while preserving contractions and hyphens
        text = SPECIAL_CHARS_PATTERN.sub(" ", text)

        # Lowercase and normalize whitespace
        text = text.lower()
        text = REPEATED_SPACES_PATTERN.sub(" ", text).strip()

        return text

    def tokenize(self, text: str) -> list[str]:
        """Tokenize cleaned text into word tokens.

        Args:
            text: Cleaned text string.

        Returns:
            List of word tokens.
        """
        if not text:
            return []

        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback regex-based word tokenization if punkt tokenizer fails
            tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)

        cleaned_tokens: list[str] = []
        for t in tokens:
            stripped = PUNCTUATION_STRIP_PATTERN.sub("", t.strip())
            if stripped:
                cleaned_tokens.append(stripped)

        return cleaned_tokens

    def filter_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove general stopwords while preserving sentiment negation modifiers.

        Args:
            tokens: Raw token list.

        Returns:
            Filtered token list.
        """
        return [token for token in tokens if token not in self.filtered_stopwords]

    def _get_wordnet_pos(self, word: str) -> str:
        """Map POS tag to WordNet POS format for accurate lemmatization."""
        # Fast rule-based heuristics for verb/adjective/noun lemmatization
        if word.endswith(("ing", "ed")):
            return wordnet.VERB
        if word.endswith(("er", "est")):
            return wordnet.ADJ
        if word.endswith("ly"):
            return wordnet.ADV
        return wordnet.NOUN

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Apply WordNet lemmatization with heuristic POS tagging to tokens.

        Args:
            tokens: Tokenized words.

        Returns:
            List of lemmatized words.
        """
        lemmatized: list[str] = []
        for token in tokens:
            try:
                pos = self._get_wordnet_pos(token)
                lemma = self.lemmatizer.lemmatize(token, pos=pos)
                # If unchanged and word is noun-like, attempt noun lemmatization
                if lemma == token and pos != wordnet.NOUN:
                    lemma = self.lemmatizer.lemmatize(token, pos=wordnet.NOUN)
                lemmatized.append(lemma)
            except Exception:
                # If lemmatizer is unavailable, retain token unchanged
                lemmatized.append(token)
        return lemmatized

    def process(self, text: str) -> PreprocessedResult:
        """Execute full preprocessing pipeline on a single text string.

        Args:
            text: Raw input string.

        Returns:
            PreprocessedResult containing all intermediate and final structures.

        Raises:
            PreprocessingError: If input is not a string or processing fails.
        """
        if not isinstance(text, str):
            raise PreprocessingError(
                f"Expected string input for preprocessing, got '{type(text).__name__}'."
            )

        try:
            cleaned = self.clean_noise(text)
            tokens = self.tokenize(cleaned)
            filtered_tokens = self.filter_stopwords(tokens)
            lemmas = self.lemmatize(filtered_tokens)

            return PreprocessedResult(
                raw_text=text,
                cleaned_text=cleaned,
                tokens=filtered_tokens,
                lemmas=lemmas,
                token_count=len(lemmas),
            )
        except Exception as exc:
            raise PreprocessingError(f"Failed to preprocess text: {exc}") from exc


# Module-level default preprocessor singleton
_DEFAULT_PREPROCESSOR = TextPreprocessor()


def preprocess_text(text: str) -> PreprocessedResult:
    """Preprocess a single text string using default settings."""
    return _DEFAULT_PREPROCESSOR.process(text)


def preprocess_corpus(texts: Sequence[str]) -> list[PreprocessedResult]:
    """Preprocess a sequence/corpus of raw texts.

    Args:
        texts: Sequence of raw text strings.

    Returns:
        List of PreprocessedResult instances.
    """
    if not isinstance(texts, (list, tuple)):
        raise PreprocessingError(
            f"Expected list/tuple of strings, received '{type(texts).__name__}'."
        )

    results: list[PreprocessedResult] = []
    for idx, item in enumerate(texts):
        if not isinstance(item, str):
            raise PreprocessingError(
                f"Item at index {idx} is not a string (type: {type(item).__name__})."
            )
        results.append(_DEFAULT_PREPROCESSOR.process(item))

    logger.info("Successfully preprocessed corpus of %d item(s).", len(results))
    return results
