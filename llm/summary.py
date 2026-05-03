# llm/summary.py

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_final_summary(conversation, signals_summary):
    prompt = f"""
Analyze this conversation and provide a final assessment.

Conversation:
{conversation}

Summary Signals:
- Avg suspicion: {signals_summary['avg_score']}
- Emotion level: {signals_summary['emotion_mean']}
- Tone shifts: {signals_summary['shift_mean']}
- Contradictions: {signals_summary['contradiction_max']}
- Behavioral/evasive patterns: {signals_summary.get('behavioral_mean', 0)}

Give:
1. Overall assessment
2. Key behavioral observations
3. Whether the speaker seems consistent or not
4. Do NOT mention numeric values. Focus on behavior.

Rules:
- Treat denial-to-correction and timeline shifts as consistency concerns.
- Treat uncertainty, vagueness, minimization, reassurance, and defensive questioning as behavioral cues, not factual proof.
- Do not call the speaker deceptive. Describe linguistic patterns only.

Keep it concise.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
