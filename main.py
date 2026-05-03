import html
import json
import logging
import re
from datetime import datetime
from io import BytesIO

import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from config import CONTRADICTION_THRESHOLD, SHIFT_THRESHOLD, WEIGHTS
from features.contradiction import contradiction_strength
from features.emotion import emotion_strength
from features.sentiment import normalize_shift
from models.loader import load_models
from output.formatter import interpret_score
from output.summary import generate_summary
from utils.text_utils import (
    get_level,
    get_main_verb,
    get_object,
    has_negation,
    has_semantic_negation,
)


logging.getLogger("transformers").setLevel(logging.ERROR)

PLACEHOLDER = '"I did not go to the party.\nActually I went there later."'
PRONOUNS = {"i", "me", "you", "he", "she", "it", "they", "we"}

DENIAL_PHRASES = [
    "i didn't",
    "i did not",
    "i never",
    "never went",
    "never been",
    "never met",
    "no idea",
    "no clue",
    "nothing happened",
    "didn't do",
    "did not do",
    "didn't change",
    "did not change",
    "wasn't involved",
    "was not involved",
    "wasn't part",
    "was not part",
    "didn't do anything",
    "did not do anything",
]

UNCERTAINTY_PHRASES = [
    "i mean",
    "well",
    "i think",
    "maybe",
    "might have",
    "could have",
    "not sure",
    "i guess",
    "probably",
    "kind of",
    "sort of",
]

MINIMIZATION_PHRASES = [
    "nothing important",
    "not important",
    "wasn't important",
    "was not important",
    "not a big deal",
    "nothing happened",
    "nothing serious",
    "it's fine",
    "its fine",
    "doesn't matter",
    "does not matter",
    "just to check",
    "only checked",
    "only opened",
    "opened it once",
    "once just",
]

VAGUE_PHRASES = ["something", "stuff", "things", "a bit", "somewhere", "for a while", "just", "once"]
REASSURANCE_PHRASES = ["trust me", "you can trust", "honestly"]
DEFENSIVE_PHRASES = ["why are you", "accusing me", "what you're talking about", "i didn't do anything wrong"]
OFFENSIVE_PHRASES = ["shut up", "liar", "stupid", "idiot", "nonsense", "ridiculous"]
CHALLENGE_PHRASES = [
    "why would i",
    "why are you",
    "how dare you",
    "are you serious",
    "what are you talking about",
    "stop accusing",
    "don't accuse",
    "do not accuse",
]
JUSTIFICATION_PHRASES = [
    "just to",
    "only",
    "i only",
    "i just",
    "to check",
    "by mistake",
    "accidentally",
    "not my fault",
    "i didn't mean",
    "i did not mean",
]
DISTANCING_PHRASES = [
    "that thing",
    "that place",
    "the file",
    "that",
    "it",
    "anything",
    "something",
]
CORRECTION_MARKERS = ["actually", "to be honest", "truth is"]
OUTSIDE_MARKERS = ["step out", "stepped out", "went out", "left home", "was out"]
HOME_MARKERS = ["was home", "at home", "went home", "home most"]
STRONG_HOME_MARKERS = ["whole evening", "all evening", "entire evening", "the whole night"]
ABSENT_PLACE_PATTERNS = [
    r"\bnever\s+(?:went|been|visited|entered)\b",
    r"\bdid(?:n['’]t| not)\s+(?:go|visit|enter)\b",
    r"\bwas(?:n['’]t| not)\s+(?:there|at|in)\b",
]
PRESENT_PLACE_PATTERNS = [
    r"\b(?:went|visited|entered|came|arrived)\b",
    r"\bwas\s+(?:there|at|in)\b",
    r"\bwent\s+inside\b",
]
PARTIAL_PLACE_PATTERNS = [
    r"\bpassed\s+by\b",
    r"\bnear(?:by)?\b",
    r"\boutside\b",
    r"\baround\s+there\b",
]
DENIED_FILE_CONTACT_PATTERNS = [
    r"\bnever\s+(?:touched|opened|accessed|used)\b",
    r"\bdid(?:n['’]t| not)\s+(?:touch|open|access|use)\b",
]
FILE_ACCESS_PATTERNS = [
    r"\b(?:opened|accessed|used|checked|looked\s+at|viewed)\b",
]
FILE_CHANGE_DENIAL_PATTERNS = [
    r"\bdid(?:n['’]t| not)\s+(?:change|edit|modify|delete|move|copy)\b",
    r"\bnever\s+(?:changed|edited|modified|deleted|moved|copied)\b",
]
INVOLVEMENT_DENIAL_PATTERNS = [
    r"\bwas(?:n['â€™]t| not)\s+involved\b",
    r"\bwas(?:n['â€™]t| not)\s+part\s+of\b",
    r"\bdid(?:n['â€™]t| not)\s+(?:do|take part|participate)\b",
    r"\bnever\s+(?:participated|helped|joined)\b",
]
KNOWLEDGE_ADMISSION_PATTERNS = [
    r"\bknew\s+about\b",
    r"\bheard\s+about\b",
    r"\bwas\s+aware\b",
]
PRESENCE_ADMISSION_PATTERNS = [
    r"\bi\s+was\s+there\b",
    r"\bwas\s+there\b",
    r"\bi\s+showed\s+up\b",
    r"\bi\s+was\s+around\b",
]
ACTION_DENIAL_PATTERNS = [
    r"\bdid(?:n['â€™]t| not)\s+do\s+anything\b",
    r"\bdid(?:n['â€™]t| not)\s+(?:touch|change|help|participate|take part)\b",
    r"\bnothing\s+to\s+do\s+with\b",
]


st.set_page_config(
    page_title="Suspicious Conversation Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1120px; }
    h1 { margin-bottom: 0.25rem; }
    .sentence-card {
        border: 1px solid #303747;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin: 0.75rem 0;
        background: #171b24;
        color: #f8fafc;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.24);
    }
    .sentence-card h4,
    .sentence-card p,
    .sentence-card li,
    .sentence-card strong {
        color: #f8fafc;
    }
    .sentence-card h4 {
        margin: 0 0 0.7rem;
    }
    .metric-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin: 0.55rem 0 0.25rem;
    }
    .metric-pill {
        border: 1px solid #3f4758;
        border-radius: 999px;
        padding: 0.28rem 0.6rem;
        background: #111827;
        color: #f8fafc;
        font-size: 0.9rem;
    }
    .metric-pill strong { color: #cbd5e1; }
    .highlighted {
        background: #fff1a8;
        color: #111827;
        padding: 0.05rem 0.22rem;
        border-radius: 4px;
        font-weight: 700;
    }
    .disclaimer {
        border-left: 4px solid #f59e0b;
        background: #2a2110;
        color: #fde68a;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin: 0.65rem 0 1rem;
    }
    .disclaimer strong { color: #fbbf24; }
    .final-panel {
        border: 1px solid #303747;
        border-radius: 8px;
        padding: 1.2rem;
        background: #171b24;
        color: #f8fafc;
        margin-top: 1rem;
    }
    .final-panel h3,
    .final-panel h4,
    .final-panel p,
    .final-panel li {
        color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading NLP and emotion models...")
def get_cached_models():
    return load_models()


def safe_llm_explanation(sentence, signals):
    try:
        from llm.explainer import generate_explanation

        return generate_explanation(sentence, signals)
    except Exception:
        reasons = signals.get("reasons", [])
        if any("Denial-to-correction" in reason for reason in reasons):
            return "The sentence revises or softens an earlier denial, which creates a clear consistency concern."
        if any("File access contradiction" in reason for reason in reasons):
            return "The statement changes the earlier claim about touching or accessing the file, which is a strong consistency concern."
        if any("involvement" in reason.lower() or "responsibility" in reason.lower() for reason in reasons):
            return "The wording narrows or revises the speaker's role, moving from no involvement toward knowledge or presence while denying responsibility."
        if any("Damage-control" in reason for reason in reasons) or any("Containment denial" in reason for reason in reasons):
            return "The wording narrows or downplays responsibility after a concerning admission, which raises suspicion."
        if any("defensive" in reason.lower() or "reassurance" in reason.lower() for reason in reasons):
            return "The wording shifts into a protective or persuasive posture, which can signal discomfort or an attempt to control the interpretation."
        if any("Soft revision" in reason for reason in reasons):
            return "The wording softens an earlier absolute denial, which makes the account less stable."
        if any("place contradiction" in reason.lower() for reason in reasons):
            return "The statement changes the earlier position about being at or near the place, which raises a consistency concern."
        if any("timeline" in reason.lower() for reason in reasons):
            return "The wording introduces a timeline detail that does not sit cleanly with an earlier statement."
        if signals.get("behavioral", 0) > 0:
            return "The wording uses behavioral cues such as uncertainty, minimization, denial, or reassurance, which can indicate evasiveness."
        if signals["contradiction"] > CONTRADICTION_THRESHOLD:
            return "This statement appears inconsistent with an earlier claim, which raises the suspicion level."
        if signals["shift"] > SHIFT_THRESHOLD:
            return "The tone changes noticeably here, suggesting a possible shift in emotional posture."
        if signals["emotion"] > 0:
            return "The wording carries heightened emotion, which may indicate discomfort or defensiveness."
        return "The sentence appears relatively consistent and does not show strong suspicious signals."


def safe_final_summary(conversation, summary):
    try:
        from llm.summary import generate_final_summary

        return generate_final_summary(conversation, summary)
    except Exception:
        level = interpret_score(summary["avg_score"]).lower()
        observations = generate_summary(summary)
        if not observations:
            return f"The conversation shows {level}, with mostly stable tone and limited signs of inconsistency."
        return f"The conversation shows {level}. The main signals are: {', '.join(observations).lower()}."


def level_badge(level):
    if level == "High":
        return "🔴 High"
    if level == "Moderate":
        return "🟡 Moderate"
    return "🟢 Low"


def normalize_text(text):
    normalized = (
        text.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("â€™", "'")
    )
    normalized = re.sub(r"\bn\?t\b", "n't", normalized)
    normalized = re.sub(r"([a-z])\?t\b", r"\1't", normalized)
    return normalized


def has_any_phrase(text, phrases):
    text = normalize_text(text)
    return any(phrase in text for phrase in phrases)


def has_any_pattern(text, patterns):
    text = normalize_text(text)
    return any(re.search(pattern, text) for pattern in patterns)


def extract_claims(sentence_text):
    text = normalize_text(sentence_text)
    claims = set()

    if has_any_pattern(text, ABSENT_PLACE_PATTERNS):
        claims.add("absent_from_place")
    if has_any_pattern(text, PRESENT_PLACE_PATTERNS):
        claims.add("present_at_place")
    if has_any_pattern(text, PARTIAL_PLACE_PATTERNS):
        claims.add("partial_presence_near_place")
    if has_any_phrase(text, HOME_MARKERS):
        claims.add("home")
    if has_any_phrase(text, STRONG_HOME_MARKERS):
        claims.add("strong_home_claim")
    if has_any_phrase(text, OUTSIDE_MARKERS):
        claims.add("outside_home")
    if has_any_pattern(text, DENIED_FILE_CONTACT_PATTERNS):
        claims.add("denied_file_contact")
    if has_any_pattern(text, FILE_ACCESS_PATTERNS):
        claims.add("file_access")
    if has_any_pattern(text, FILE_CHANGE_DENIAL_PATTERNS):
        claims.add("denied_file_change")
    if has_any_pattern(text, INVOLVEMENT_DENIAL_PATTERNS):
        claims.add("denied_involvement")
    if has_any_pattern(text, KNOWLEDGE_ADMISSION_PATTERNS):
        claims.add("knowledge_admission")
    if has_any_pattern(text, PRESENCE_ADMISSION_PATTERNS):
        claims.add("presence_admission")
    if has_any_pattern(text, ACTION_DENIAL_PATTERNS):
        claims.add("action_denial")

    return claims


def detect_behavioral_signals(sentence_text, context):
    text = normalize_text(sentence_text)
    score = 0.0
    reasons = []

    if has_any_phrase(text, DENIAL_PHRASES):
        score += 0.12
        reasons.append("Denial language detected")

    if has_any_pattern(text, ABSENT_PLACE_PATTERNS) or "never" in text:
        score += 0.06
        reasons.append("Absolute denial detected")

    if has_any_phrase(text, UNCERTAINTY_PHRASES):
        score += 0.15
        reasons.append("Uncertainty in statement")

    if has_any_phrase(text, MINIMIZATION_PHRASES):
        score += 0.18
        reasons.append("Downplaying behavior detected")

    if has_any_phrase(text, VAGUE_PHRASES):
        score += 0.10
        reasons.append("Vague language detected")

    if has_any_phrase(text, REASSURANCE_PHRASES):
        score += 0.15
        reasons.append("Reassurance phrase detected")

    if has_any_phrase(text, DEFENSIVE_PHRASES):
        score += 0.12
        reasons.append("Defensive wording detected")

    if has_any_phrase(text, OFFENSIVE_PHRASES):
        score += 0.18
        reasons.append("Offensive wording detected")

    if text.startswith("anyway") or " anyway" in text:
        score += 0.06
        reasons.append("Topic-shifting language detected")

    if text.startswith("well") or " well" in text:
        score += 0.08
        reasons.append("Hesitation or qualification detected")

    if "but" in text and (context["denial_seen"] or context["absolute_denial_seen"]):
        score += 0.08
        reasons.append("Qualified revision after denial")

    if context["contradiction_seen"] and (
        has_any_phrase(text, MINIMIZATION_PHRASES)
        or has_any_pattern(text, FILE_CHANGE_DENIAL_PATTERNS)
        or text.startswith("anyway")
    ):
        score += 0.18
        reasons.append("Damage-control language after contradiction")

    if context["file_access_seen"] and has_any_pattern(text, FILE_CHANGE_DENIAL_PATTERNS):
        score += 0.22
        reasons.append("Containment denial after admitted access")

    if context["absolute_denial_seen"] and (
        text.startswith("well")
        or has_any_phrase(text, UNCERTAINTY_PHRASES)
        or has_any_pattern(text, PARTIAL_PLACE_PATTERNS)
    ):
        score += 0.12
        reasons.append("Soft revision after absolute denial")

    if context["involvement_denial_seen"] and (
        has_any_pattern(text, KNOWLEDGE_ADMISSION_PATTERNS)
        or has_any_pattern(text, PRESENCE_ADMISSION_PATTERNS)
    ):
        score += 0.24
        reasons.append("Involvement story changed")

    if context["presence_admitted"] and has_any_pattern(text, ACTION_DENIAL_PATTERNS):
        score += 0.22
        reasons.append("Action denial after presence admission")

    if context["denial_seen"] and has_any_phrase(text, CORRECTION_MARKERS):
        score += 0.45
        reasons.append("Denial-to-correction pattern detected")

    if context["home_seen"] and has_any_phrase(text, OUTSIDE_MARKERS):
        score += 0.12
        reasons.append("Inconsistent timeline detail")

    return min(score, 0.75), reasons


def detect_defensiveness_signals(sentence_text, context):
    text = normalize_text(sentence_text)
    score = 0.0
    reasons = []

    if has_any_phrase(text, CHALLENGE_PHRASES):
        score += 0.30
        reasons.append("Defensive challenge detected")

    if has_any_phrase(text, DENIAL_PHRASES) or has_any_pattern(text, FILE_CHANGE_DENIAL_PATTERNS):
        score += 0.14
        reasons.append("Self-protective denial detected")

    if has_any_phrase(text, REASSURANCE_PHRASES):
        score += 0.18
        reasons.append("Persuasive reassurance detected")

    if has_any_phrase(text, JUSTIFICATION_PHRASES):
        score += 0.16
        reasons.append("Justifying language detected")

    if has_any_phrase(text, OFFENSIVE_PHRASES):
        score += 0.32
        reasons.append("Offensive reaction detected")

    if has_any_phrase(text, DISTANCING_PHRASES) and (
        context["denial_seen"] or context["contradiction_seen"]
    ):
        score += 0.10
        reasons.append("Distancing language detected")

    if context["file_access_seen"] and has_any_pattern(text, FILE_CHANGE_DENIAL_PATTERNS):
        score += 0.28
        reasons.append("Responsibility containment detected")

    if context["contradiction_seen"] and (
        text.startswith("but")
        or text.startswith("anyway")
        or has_any_phrase(text, MINIMIZATION_PHRASES)
    ):
        score += 0.18
        reasons.append("Post-contradiction defensive framing")

    if context["denial_count"] >= 1 and (
        has_any_phrase(text, DENIAL_PHRASES)
        or has_any_phrase(text, REASSURANCE_PHRASES)
        or has_any_phrase(text, JUSTIFICATION_PHRASES)
    ):
        score += 0.10
        reasons.append("Repeated defensive posture")

    if context["involvement_denial_seen"] and (
        has_any_pattern(text, KNOWLEDGE_ADMISSION_PATTERNS)
        or has_any_pattern(text, PRESENCE_ADMISSION_PATTERNS)
        or has_any_pattern(text, ACTION_DENIAL_PATTERNS)
    ):
        score += 0.26
        reasons.append("Defensive narrowing of involvement")

    if context["presence_admitted"] and has_any_pattern(text, ACTION_DENIAL_PATTERNS):
        score += 0.24
        reasons.append("Responsibility denial after presence admission")

    return min(score, 0.85), reasons


def detect_semantic_contradiction(sentence_text, context):
    text = normalize_text(sentence_text)
    claims = extract_claims(sentence_text)

    if "absent_from_place" in context["claims"] and "present_at_place" in claims:
        return 0.86, "Place contradiction detected"

    if "absent_from_place" in context["claims"] and "partial_presence_near_place" in claims:
        return 0.74, "Soft place contradiction detected"

    if "present_at_place" in context["claims"] and "absent_from_place" in claims:
        return 0.86, "Place contradiction detected"

    if "partial_presence_near_place" in context["claims"] and "absent_from_place" in claims:
        return 0.72, "Soft place contradiction detected"

    if "denied_file_contact" in context["claims"] and "file_access" in claims:
        return 0.82, "File access contradiction detected"

    if "file_access" in context["claims"] and "denied_file_contact" in claims:
        return 0.82, "File access contradiction detected"

    if "denied_involvement" in context["claims"] and "presence_admission" in claims:
        return 0.82, "Involvement contradiction detected"

    if "denied_involvement" in context["claims"] and "knowledge_admission" in claims:
        return 0.70, "Soft involvement contradiction detected"

    if "presence_admission" in context["claims"] and "action_denial" in claims:
        return 0.68, "Responsibility narrowing detected"

    if context["home_seen"] and has_any_phrase(text, OUTSIDE_MARKERS):
        if context["strong_home_claim"]:
            return 0.82, "Location contradiction detected"
        return 0.7, "Location contradiction detected"

    if context["outside_seen"] and has_any_phrase(text, HOME_MARKERS):
        return 0.7, "Location contradiction detected"

    return 0.0, None


def update_behavioral_context(sentence_text, context):
    text = normalize_text(sentence_text)
    claims = extract_claims(sentence_text)
    context["claims"].update(claims)
    if has_any_phrase(text, DENIAL_PHRASES):
        context["denial_seen"] = True
        context["denial_count"] += 1
    if has_any_pattern(text, FILE_CHANGE_DENIAL_PATTERNS):
        context["denial_seen"] = True
        context["denial_count"] += 1
    if has_any_phrase(text, ["never", "whole evening", "entire evening", "all evening"]):
        context["absolute_denial_seen"] = True
    if has_any_phrase(text, HOME_MARKERS):
        context["home_seen"] = True
    if has_any_phrase(text, STRONG_HOME_MARKERS):
        context["strong_home_claim"] = True
    if has_any_phrase(text, OUTSIDE_MARKERS):
        context["outside_seen"] = True
    if "file_access" in claims:
        context["file_access_seen"] = True
    if "denied_involvement" in claims:
        context["involvement_denial_seen"] = True
    if "knowledge_admission" in claims:
        context["knowledge_admitted"] = True
    if "presence_admission" in claims:
        context["presence_admitted"] = True


def append_observation_once(observations, observation):
    normalized = observation.lower()
    if not any(existing.lower() == normalized for existing in observations):
        observations.append(observation)


def split_sentences_with_analysis(text):
    nlp, sia, emotion_classifier = get_cached_models()
    doc = nlp(text)

    previous_sentiment = 0.0
    memory = []
    behavioral_context = {
        "denial_seen": False,
        "absolute_denial_seen": False,
        "home_seen": False,
        "outside_seen": False,
        "strong_home_claim": False,
        "contradiction_seen": False,
        "file_access_seen": False,
        "involvement_denial_seen": False,
        "knowledge_admitted": False,
        "presence_admitted": False,
        "denial_count": 0,
        "claims": set(),
    }
    sentence_results = []

    for index, sent in enumerate(doc.sents, start=1):
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue

        score = 0.0
        reasons = []

        emotion_result = emotion_classifier(sentence_text)[0][0]
        emotion_label = emotion_result["label"]
        emotion_score = emotion_result["score"]
        emotion_signal = emotion_strength(emotion_label, emotion_score)

        if emotion_signal > 0:
            score += WEIGHTS["emotion"] * emotion_signal
            reasons.append("High emotional intensity")

        polarity = sia.polarity_scores(sentence_text)["compound"]
        shift = abs(polarity - previous_sentiment)
        shift_signal = normalize_shift(shift)

        if memory and shift_signal > SHIFT_THRESHOLD:
            score += WEIGHTS["shift"] * shift_signal
            reasons.append("Sudden tone shift")

        previous_sentiment = polarity

        verb = get_main_verb(sent)
        negation = has_negation(sent) or has_semantic_negation(sent)
        obj = get_object(sent)

        if obj and (obj.lower() in PRONOUNS or obj == "-PRON-"):
            obj = None

        contradiction_signal = 0.0
        comparable_contradiction = False
        for past in memory:
            current_contra = contradiction_strength(past, verb, obj, negation, nlp)
            if current_contra > contradiction_signal:
                contradiction_signal = current_contra
                comparable_contradiction = bool(
                    past.get("verb")
                    and verb
                    and (past.get("object") or obj)
                    and past.get("negation") != negation
                )

        semantic_contra, semantic_reason = detect_semantic_contradiction(
            sentence_text,
            behavioral_context,
        )
        if semantic_contra > contradiction_signal:
            contradiction_signal = semantic_contra
            comparable_contradiction = True

        contradiction_triggered = comparable_contradiction and contradiction_signal > CONTRADICTION_THRESHOLD
        if contradiction_triggered:
            score += WEIGHTS["contradiction"] * contradiction_signal
            reasons.append(semantic_reason or "Contradiction detected")

        behavioral_signal, behavioral_reasons = detect_behavioral_signals(
            sentence_text,
            behavioral_context,
        )
        if behavioral_signal > 0:
            score += behavioral_signal
            reasons.extend(behavioral_reasons)

        defensiveness_signal, defensiveness_reasons = detect_defensiveness_signals(
            sentence_text,
            behavioral_context,
        )
        if defensiveness_signal > 0:
            score += defensiveness_signal * 0.6
            reasons.extend(defensiveness_reasons)

        if defensiveness_signal >= 0.30 and behavioral_signal == 0 and not contradiction_triggered:
            score += 0.08
            reasons.append("Standalone defensive reaction")

        if contradiction_triggered and defensiveness_signal >= 0.25:
            score += 0.12
            reasons.append("Contradiction paired with defensive framing")

        if behavioral_signal >= 0.35 and defensiveness_signal >= 0.30:
            score += 0.10
            reasons.append("Multiple suspicious behavior signals clustered")

        final_score = round(min(score, 1.0) * 100, 1)
        signals = {
            "emotion": round(emotion_signal, 2),
            "shift": round(shift_signal, 2),
            "contradiction": round(contradiction_signal if contradiction_triggered else 0.0, 2),
            "behavioral": round(behavioral_signal, 2),
            "defensiveness": round(defensiveness_signal, 2),
            "reasons": reasons,
            "level": get_level(final_score),
            "score": final_score,
        }

        sentence_results.append(
            {
                "index": index,
                "sentence": sentence_text,
                "score": final_score,
                "level": signals["level"],
                "reasons": reasons or ["No major suspicious signal detected"],
                "emotion_label": emotion_label.title(),
                "emotion_score": round(emotion_score, 2),
                "polarity": round(polarity, 2),
                "shift": round(shift, 2) if memory else None,
                "signals": signals,
                "explanation": safe_llm_explanation(sentence_text, signals),
                "highlighted": highlight_sentence(sentence_text, reasons),
            }
        )

        memory.append({"verb": verb, "negation": negation, "object": obj})
        if contradiction_triggered:
            behavioral_context["contradiction_seen"] = True
        update_behavioral_context(sentence_text, behavioral_context)

    return sentence_results


def highlight_sentence(sentence, reasons):
    escaped = html.escape(sentence)
    suspicious_terms = [
        "accusing me",
        "actually",
        "well",
        "but",
        "never",
        "involved",
        "knew about",
        "wasn't part",
        "was not part",
        "was there",
        "didn't do anything",
        "did not do anything",
        "i mean",
        "passed by",
        "nothing happened",
        "nothing serious",
        "wasn't important",
        "was not important",
        "just to check",
        "opened it once",
        "touched",
        "opened",
        "change anything",
        "trust me",
        "you can trust",
        "did not",
        "didn't",
        "i think",
        "maybe",
        "not sure",
        "not important",
        "not a big deal",
        "might have",
        "could have",
        "a bit",
        "step out",
        "stepped out",
        "at home",
        "that place",
        "whole evening",
        "no idea",
        "honestly",
        "shut up",
        "liar",
        "stupid",
        "idiot",
    ]
    if reasons == ["No major suspicious signal detected"]:
        return escaped

    highlighted = escaped
    for term in suspicious_terms:
        highlighted = highlighted.replace(
            html.escape(term),
            f"<span class='highlighted'>{html.escape(term)}</span>",
        )
        highlighted = highlighted.replace(
            html.escape(term.title()),
            f"<span class='highlighted'>{html.escape(term.title())}</span>",
        )
    return highlighted


def build_summary(text, sentence_results):
    scores = [item["score"] for item in sentence_results]
    emotions = [item["signals"]["emotion"] for item in sentence_results]
    shifts = [item["signals"]["shift"] for item in sentence_results]
    contradictions = [item["signals"]["contradiction"] for item in sentence_results]
    behavioral = [item["signals"]["behavioral"] for item in sentence_results]
    defensiveness = [item["signals"]["defensiveness"] for item in sentence_results]
    all_reasons = [
        reason
        for item in sentence_results
        for reason in item["reasons"]
    ]

    summary = {
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "emotion_mean": float(np.mean(emotions)) if emotions else 0.0,
        "emotion_max": float(np.max(emotions)) if emotions else 0.0,
        "shift_mean": float(np.mean(shifts)) if shifts else 0.0,
        "shift_max": float(np.max(shifts)) if shifts else 0.0,
        "contradiction_max": float(np.max(contradictions)) if contradictions else 0.0,
        "behavioral_mean": float(np.mean(behavioral)) if behavioral else 0.0,
        "behavioral_max": float(np.max(behavioral)) if behavioral else 0.0,
        "defensiveness_mean": float(np.mean(defensiveness)) if defensiveness else 0.0,
        "defensiveness_max": float(np.max(defensiveness)) if defensiveness else 0.0,
    }

    observations = generate_summary(summary)
    if summary["behavioral_max"] >= 0.15:
        append_observation_once(observations, "Evasive or defensive language patterns detected")
    if summary["defensiveness_max"] >= 0.25:
        append_observation_once(observations, "Defensive posture detected")
    if any("Denial-to-correction" in reason for reason in all_reasons):
        append_observation_once(observations, "A denial is later revised or corrected")
    if any("timeline" in reason.lower() for reason in all_reasons):
        append_observation_once(observations, "Timeline language shifts across the conversation")
    if not observations:
        observations = ["No strong contradiction, defensive tone, or emotional spike was detected"]

    inconsistent = (
        summary["contradiction_max"] > CONTRADICTION_THRESHOLD
        or any("Denial-to-correction" in reason for reason in all_reasons)
        or any("timeline" in reason.lower() for reason in all_reasons)
    )
    consistency = (
        "The speaker shows signs of inconsistency."
        if inconsistent
        else "The speaker appears broadly consistent based on the available text."
    )

    return {
        "signals": summary,
        "assessment": safe_final_summary(text, summary),
        "observations": observations,
        "consistency": consistency,
        "overall_level": interpret_score(summary["avg_score"]),
    }


def get_pdf_font(size, bold=False):
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    font_path = f"C:/Windows/Fonts/{font_name}"
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        return ImageFont.load_default()


def sanitize_pdf_text(value):
    return str(value).replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')


def wrap_pdf_text(draw, text, font, max_width):
    words = sanitize_pdf_text(text).split()
    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)
    return lines or [""]


def build_pdf_report(text, sentence_results, final_analysis):
    page_width, page_height = 1240, 1754
    margin = 80
    content_width = page_width - margin * 2
    bg = "white"
    ink = "#111827"
    muted = "#4b5563"
    accent = "#1d4ed8"

    title_font = get_pdf_font(34, bold=True)
    h_font = get_pdf_font(22, bold=True)
    body_font = get_pdf_font(17)
    small_font = get_pdf_font(14)
    bold_font = get_pdf_font(17, bold=True)

    pages = []
    image = Image.new("RGB", (page_width, page_height), bg)
    draw = ImageDraw.Draw(image)
    y = margin

    def new_page():
        nonlocal image, draw, y
        pages.append(image)
        image = Image.new("RGB", (page_width, page_height), bg)
        draw = ImageDraw.Draw(image)
        y = margin

    def ensure_space(height):
        if y + height > page_height - margin:
            new_page()

    def add_wrapped(text_value, font=body_font, color=ink, gap=10, prefix=None):
        nonlocal y
        lines = wrap_pdf_text(draw, text_value, font, content_width - (28 if prefix else 0))
        line_height = getattr(font, "size", 16) + 9
        ensure_space(line_height * len(lines) + gap)
        for idx, line in enumerate(lines):
            x = margin + (28 if prefix else 0)
            bullet = prefix if idx == 0 else None
            if bullet:
                draw.text((margin, y), bullet, fill=color, font=font)
            draw.text((x, y), line, fill=color, font=font)
            y += line_height
        y += gap

    def add_heading(text_value):
        nonlocal y
        ensure_space(46)
        draw.text((margin, y), sanitize_pdf_text(text_value), fill=accent, font=h_font)
        y += 42

    draw.text((margin, y), "Suspicious Conversation Analyzer Report", fill=ink, font=title_font)
    y += 52
    add_wrapped(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small_font, muted, gap=6)
    add_wrapped("Disclaimer: This report detects behavioral patterns, not factual truth.", small_font, "#92400e", gap=24)

    add_heading("Overall Assessment")
    add_wrapped(final_analysis["assessment"])
    add_wrapped(f"Overall Level: {final_analysis['overall_level']}", bold_font)
    add_wrapped(f"Consistency Judgment: {final_analysis['consistency']}", bold_font)

    add_heading("Key Observations")
    for observation in final_analysis["observations"]:
        add_wrapped(observation, body_font, ink, gap=4, prefix="-")

    add_heading("Conversation")
    for line in text.splitlines():
        add_wrapped(line, body_font, ink, gap=2)

    add_heading("Sentence Analysis")
    for item in sentence_results:
        ensure_space(180)
        draw.rectangle((margin, y, page_width - margin, y + 2), fill="#e5e7eb")
        y += 18
        add_wrapped(f"Sentence {item['index']}: {item['sentence']}", bold_font, ink, gap=8)
        add_wrapped(
            f"Score: {item['score']} / 100 | Level: {item['level']} | Emotion: {item['emotion_label']} ({item['emotion_score']})",
            small_font,
            muted,
            gap=8,
        )
        add_wrapped(
            f"Defensiveness: {round(item['signals']['defensiveness'] * 100, 1)} / 100 | Behavioral/Evasive: {round(item['signals']['behavioral'] * 100, 1)} / 100 | Contradiction: {round(item['signals']['contradiction'] * 100, 1)} / 100",
            small_font,
            muted,
            gap=8,
        )
        add_wrapped("Signals Detected:", bold_font, ink, gap=4)
        for reason in item["reasons"]:
            add_wrapped(reason, small_font, ink, gap=2, prefix="-")
        add_wrapped("Explanation:", bold_font, ink, gap=4)
        add_wrapped(item["explanation"], body_font, ink, gap=18)

    pages.append(image)
    output = BytesIO()
    pages[0].save(output, format="PDF", save_all=True, append_images=pages[1:], resolution=150)
    return output.getvalue()


def render_sentence_card(item):
    reasons_html = "".join(f"<li>{html.escape(reason)}</li>" for reason in item["reasons"])
    st.markdown(
        f"""
        <div class="sentence-card">
            <h4>Sentence {item["index"]}: {html.escape(item["sentence"])}</h4>
            <div class="metric-row">
                <span class="metric-pill"><strong>Score:</strong> {item["score"]} / 100</span>
                <span class="metric-pill"><strong>Level:</strong> {level_badge(item["level"])}</span>
                <span class="metric-pill"><strong>Emotion:</strong> {item["emotion_label"]} ({item["emotion_score"]})</span>
                <span class="metric-pill"><strong>Defensive:</strong> {round(item["signals"]["defensiveness"] * 100, 1)} / 100</span>
            </div>
            <p><strong>Signals Detected</strong></p>
            <ul>{reasons_html}</ul>
            <p><strong>Explanation</strong></p>
            <p>{html.escape(item["explanation"])}</p>
            <p><strong>Highlighted Sentence</strong></p>
            <p>{item["highlighted"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.header("About")
    st.write(
        "This app analyzes linguistic patterns such as tone shifts, contradictions, and emotional signals."
    )
    st.header("How it works")
    st.markdown(
        """
        - NLP feature extraction
        - Suspicion scoring
        - LLM-based explanations
        """
    )
    st.header("Disclaimer")
    st.warning("Not a lie detector. For behavioral insights only.")

st.title("Suspicious Conversation Analyzer")
st.write(
    "Analyze conversations for linguistic signals of deception, inconsistency, and emotional patterns."
)
st.markdown(
    "<div class='disclaimer'><strong>Disclaimer:</strong> This system detects behavioral patterns, not factual truth.</div>",
    unsafe_allow_html=True,
)

if "conversation_text" not in st.session_state:
    st.session_state.conversation_text = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = None

conversation_text = st.text_area(
    "Conversation input",
    key="conversation_text",
    height=220,
    placeholder=f"Paste your conversation here...\n\nExample:\n{PLACEHOLDER}",
    label_visibility="collapsed",
)

analyze_clicked = st.button("🔍 Analyze Conversation", type="primary", use_container_width=True)

if analyze_clicked:
    if not conversation_text.strip():
        st.error("Paste a conversation or load a sample case before analyzing.")
    else:
        with st.spinner("Analyzing conversation..."):
            sentence_results = split_sentences_with_analysis(conversation_text)
            final_analysis = build_summary(conversation_text, sentence_results)
            st.session_state.analysis = {
                "sentence_results": sentence_results,
                "final_analysis": final_analysis,
                "conversation": conversation_text,
            }

if st.session_state.analysis:
    sentence_results = st.session_state.analysis["sentence_results"]
    final_analysis = st.session_state.analysis["final_analysis"]

    st.subheader("Sentence-by-Sentence Analysis")
    for item in sentence_results:
        render_sentence_card(item)

    st.subheader("Timeline View")
    timeline = pd.DataFrame(
        {
            "Sentence": [item["index"] for item in sentence_results],
            "Suspicion Score": [item["score"] for item in sentence_results],
            "Emotion Intensity": [round(item["signals"]["emotion"] * 100, 1) for item in sentence_results],
            "Contradiction Score": [
                round(item["signals"]["contradiction"] * 100, 1)
                for item in sentence_results
            ],
            "Defensiveness": [
                round(item["signals"]["defensiveness"] * 100, 1)
                for item in sentence_results
            ],
        }
    )
    chart_data = timeline.melt(
        "Sentence",
        var_name="Signal",
        value_name="Score",
    )
    selected_signals = st.multiselect(
        "Signals shown",
        ["Suspicion Score", "Defensiveness", "Contradiction Score", "Emotion Intensity"],
        default=["Suspicion Score", "Defensiveness", "Contradiction Score"],
    )
    chart_data = chart_data[chart_data["Signal"].isin(selected_signals)]
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Sentence:O", title="Sentence", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Signal:N", legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[
                alt.Tooltip("Sentence:O", title="Sentence"),
                alt.Tooltip("Signal:N", title="Signal"),
                alt.Tooltip("Score:Q", title="Score", format=".1f"),
            ],
        )
        .properties(height=340)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Tip: use the selector above to hide noisy lines. Hover points for exact values.")
    spikes = [item["index"] for item in sentence_results if item["score"] >= 50]
    if spikes:
        st.caption(f"Spikes highlighted at sentence(s): {', '.join(map(str, spikes))}")

    st.markdown(
        f"""
        <div class="final-panel">
            <h3>Overall Assessment</h3>
            <p>{html.escape(final_analysis["assessment"])}</p>
            <h4>Key Observations</h4>
            <ul>{"".join(f"<li>{html.escape(obs)}</li>" for obs in final_analysis["observations"])}</ul>
            <h4>Consistency Judgment</h4>
            <p>{html.escape(final_analysis["consistency"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    report = build_pdf_report(
        st.session_state.analysis["conversation"],
        sentence_results,
        final_analysis,
    )
    st.download_button(
        "⬇️ Download PDF Report",
        data=report,
        file_name="suspicious_conversation_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
