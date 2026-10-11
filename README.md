# WellSync — Employee Wellness & Sentiment Intelligence Platform

## NLP & Ingestion Module

This module processes employee feedback text and converts unstructured text into structured NLP and sentiment information.

## Pipeline

Raw Employee Text
->
PII Cleaning
->
Text Normalization
->
Tokenization & Lemmatization
->
Negation Handling
->
VADER Sentiment Analysis
->
Structured Output

## Features

- Email masking
- Phone number masking
- URL masking
- Person-name detection and masking
- Text normalization
- Tokenization
- Stopword removal
- Lemmatization
- Negation preservation
- Negation-aware text representation
- VADER sentiment analysis
- Positive, negative and neutral scores
- Compound sentiment score
- Sentiment classification

## Project Structure

wellsync/
+-- nlp_ingestion/
¦   +-- __init__.py
¦   +-- main.py
¦   +-- pii_cleaner.py
¦   +-- preprocessing.py
¦   +-- negation.py
¦   +-- sentiment.py
¦   +-- tests/
¦       +-- __init__.py
¦       +-- test_pii.py
¦       +-- test_preprocessing.py
¦       +-- test_negation.py
¦       +-- test_sentiment.py
+-- requirements.txt
+-- .gitignore
+-- README.md

## Installation

Create and activate the virtual environment:

python -m venv venv
.\venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Install the spaCy English model:

python -m spacy download en_core_web_sm

## Run the NLP Pipeline

python -m nlp_ingestion.main

## Run Tests

python -m pytest nlp_ingestion\tests -v

## Testing Status

The current NLP module contains 20 automated tests covering PII cleaning, preprocessing, negation handling, and sentiment analysis.

Current result: 20/20 tests passed.

## Output

The pipeline produces raw text, PII-cleaned text, normalized text, tokens, processed text, negation-aware text, VADER sentiment scores, and a sentiment label.

## Technologies

- Python
- spaCy
- NLTK
- VADER Sentiment
- pytest

## Module Responsibility

Member 2 — NLP & Ingestion Engineer

Responsibilities:
- Text preprocessing
- PII cleaning
- Negation handling
- VADER sentiment scoring
- Structured NLP output for downstream modules

