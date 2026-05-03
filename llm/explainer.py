from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_explanation(sentence, signals):
    prompt = f"""
    You are explaining the reasoning of an AI system that detects suspicious patterns in conversations.

    Sentence: "{sentence}"

    System Analysis:
    - Suspicion Level: {signals['level']}
    - Emotion intensity: {signals['emotion']}
    - Tone shift: {signals['shift']}
    - Contradiction: {signals['contradiction']}
    - Behavioral/evasive language: {signals.get('behavioral', 0)}
    - Defensiveness: {signals.get('defensiveness', 0)}
    - Detected reasons: {signals.get('reasons', [])}

    Task:
    Explain the reasoning in a natural, concise, and confident way.

    Rules:
    - Do NOT mention "the system"
    - Do NOT mention numbers
    - Do NOT repeat the sentence
    - Keep it short (1–2 sentences max)
    - Stay consistent with the detected reasons
    - Do NOT assume contradiction unless "Contradiction detected", "Location contradiction detected", "Denial-to-correction pattern detected", or "Inconsistent timeline detail" appears in detected reasons
    - If behavioral/evasive language is present, explain that cue directly
    - If defensiveness is present, describe the defensive posture directly
    - If LOW → explain why it seems normal
    - If MODERATE/HIGH → clearly state what feels suspicious
    - Avoid filler phrases (e.g., "overall assessment", "contributes to")
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
