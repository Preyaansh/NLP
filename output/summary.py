def generate_summary(summary):
    observations = []

    if summary["contradiction_max"] > 0.65:
        observations.append("Contradiction detected in conversation")

    if summary.get("behavioral_max", 0) > 0.15:
        observations.append("Evasive or defensive language patterns detected")

    if summary.get("defensiveness_max", 0) > 0.25:
        observations.append("Defensive posture detected")

    if summary["emotion_mean"] > 0.3 or summary["emotion_max"] > 0.7:
        observations.append("Emotionally intense responses observed")

    if summary["shift_mean"] > 0.3 or summary["shift_max"] > 0.5:
        observations.append("Notable tone shifts detected")

    return observations
