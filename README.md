# Suspicious Conversation Analyzer

Streamlit app for analyzing conversation text for linguistic signals such as contradiction, defensiveness, minimization, uncertainty, emotional intensity, and tone shifts.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Deploy on Render

This repo includes Render-ready files:

- `render.yaml`
- `Procfile`
- `runtime.txt`
- `.streamlit/config.toml`

Render start command:

```bash
streamlit run main.py --server.address=0.0.0.0 --server.port=$PORT
```

Set this environment variable in Render:

```text
GROQ_API_KEY=your_groq_key
```

The app still runs without `GROQ_API_KEY`, but it will use local fallback explanations instead of LLM-generated explanations.

The hosted version uses a lightweight built-in emotion detector to keep Render free-tier memory usage stable. To use the heavier Hugging Face emotion model locally, install `transformers` and `torch`, then set:

```text
USE_TRANSFORMERS_EMOTION=true
```

## Notes

This is not a lie detector. It reports linguistic and behavioral patterns, not factual truth.
