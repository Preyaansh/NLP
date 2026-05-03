from config import EMOTION_THRESHOLD

def emotion_strength(label, score):
    if score < EMOTION_THRESHOLD:
        return 0.0

    if label in ["anger", "fear"]:
        return score
    elif label == "sadness":
        return 0.6 * score
    return 0.0