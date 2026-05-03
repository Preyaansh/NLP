import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

def load_models():
    nlp = spacy.load("en_core_web_md")
    sia = SentimentIntensityAnalyzer()

    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )

    return nlp, sia, emotion_classifier