import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline


def load_spacy_model():
    for model_name in ("en_core_web_md", "en_core_web_sm"):
        try:
            return spacy.load(model_name)
        except OSError:
            continue

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def load_sentiment_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def load_emotion_classifier():
    try:
        return pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
        )
    except Exception:
        def neutral_emotion_classifier(text):
            return [[{"label": "neutral", "score": 0.0}]]

        return neutral_emotion_classifier


def load_models():
    nlp = load_spacy_model()
    sia = load_sentiment_analyzer()
    emotion_classifier = load_emotion_classifier()

    return nlp, sia, emotion_classifier
