import os

import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer


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
    if os.getenv("USE_TRANSFORMERS_EMOTION", "").lower() not in {"1", "true", "yes"}:
        return LightweightEmotionClassifier()

    try:
        from transformers import pipeline

        return pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
        )
    except Exception:
        def neutral_emotion_classifier(text):
            return [[{"label": "neutral", "score": 0.0}]]

        return neutral_emotion_classifier


class LightweightEmotionClassifier:
    anger_words = {
        "accuse",
        "accusing",
        "angry",
        "mad",
        "ridiculous",
        "liar",
        "stupid",
        "idiot",
        "shut",
        "nonsense",
        "wrong",
    }
    fear_words = {
        "afraid",
        "scared",
        "fear",
        "worried",
        "panic",
        "nervous",
        "anxious",
    }
    sadness_words = {
        "sad",
        "sorry",
        "upset",
        "hurt",
        "cry",
        "regret",
    }

    def __call__(self, text):
        words = {word.strip(".,!?;:\"'()[]{}").lower() for word in text.split()}
        scores = {
            "anger": len(words & self.anger_words),
            "fear": len(words & self.fear_words),
            "sadness": len(words & self.sadness_words),
        }
        label, count = max(scores.items(), key=lambda item: item[1])
        if count == 0:
            return [[{"label": "neutral", "score": 0.0}]]
        return [[{"label": label, "score": min(0.65 + count * 0.15, 0.95)}]]


def load_models():
    nlp = load_spacy_model()
    sia = load_sentiment_analyzer()
    emotion_classifier = load_emotion_classifier()

    return nlp, sia, emotion_classifier
