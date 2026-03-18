# app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Sentinox — AI Sentiment Analyzer", layout="wide")


# --- Optional: small CSS fallback to forcibly hide the sidebar if present ---
# (This is a CSS hack. Removing sidebar usage is the proper fix; this is just extra safety.)
HIDE_SIDEBAR_CSS = """
<style>
/* Hide Streamlit sidebar container */
aside {display: none !important;}

/* Make main content use full width */
.css-1d391kg {display:none !important;}
.main .block-container {
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
"""
st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)
# -----------------------------------------------------------------------------


@st.cache_resource
def load_model(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"):
    """Load tokenizer and model once and cache it (Streamlit cache)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    # Model has 3 classes: negative, neutral, positive
    id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    return tokenizer, model, device, id2label


def predict_batch(tokenizer, model, device, id2label, texts, max_length=128):
    """Return list of dicts: text, label, score, probs"""
    if not texts:
        return []

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for i, p in enumerate(probs):
        label_id = int(np.argmax(p))
        label = id2label.get(label_id, "UNKNOWN")
        results.append(
            {
                "text": texts[i],
                "label": label,
                "score": round(float(p[label_id]), 4),
                "prob_negative": round(float(p[0]), 4),
                "prob_neutral": round(float(p[1]), 4),
                "prob_positive": round(float(p[2]), 4),
            }
        )

    return results


# --- UI ---
st.title("Sentinox - AI Sentiment Analyzer")
st.write(
    "Write your text and check the inner feelings. Uses a 3-class sentiment model (NEGATIVE / NEUTRAL / POSITIVE)."
)

# Put options in the main page using an expander (no sidebar)
with st.expander("Options (model & settings)", expanded=False):
    model_name = st.text_input(
        "HuggingFace model id or local path",
        value="cardiffnlp/twitter-roberta-base-sentiment",
        help="Change to a different HF model id or a local path to your checkpoint.",
    )
    max_length = st.slider(
        "Max token length", min_value=32, max_value=512, value=128, step=32
    )
    st.write(f"Device: **{'cuda' if torch.cuda.is_available() else 'cpu'}**")
    st.write("Model is cached — change the model id and then refresh to reload.")

# Load / show model info
with st.spinner("Loading model... (cached after first load)"):
    tokenizer, model, device, id2label = load_model(model_name)

st.markdown("---")

# Input area - single text
st.subheader("Try single text")
single_text = st.text_input(
    "Enter text here", value="This phone battery drains fast"
)
if st.button("Predict (single)"):
    results = predict_batch(
        tokenizer, model, device, id2label, [single_text], max_length=max_length
    )
    if results:
        r = results[0]
        st.json(r)
        st.metric("Label", r["label"], delta=None)
        # progress expects 0-100; convert score (0-1) to percent
        try:
            st.progress(int(r["score"] * 100))
        except Exception:
            st.progress(0)
        st.write("Probabilities:")
        st.write(f"- Negative: {r['prob_negative']}")
        st.write(f"- Neutral: {r['prob_neutral']}")
        st.write(f"- Positive: {r['prob_positive']}")

st.markdown("---")

# Batch input area
st.subheader("Batch input (one sentence per line)")
batch_texts = st.text_area(
    "Paste multiple lines of text (or leave empty)", height=120
)
if st.button("Predict (batch)"):
    texts = [t.strip() for t in batch_texts.splitlines() if t.strip()]
    if not texts:
        st.warning("No texts provided.")
    else:
        results = predict_batch(
            tokenizer, model, device, id2label, texts, max_length=max_length
        )
        df = pd.DataFrame(results)
        st.dataframe(df)
        # show probabilities chart (negative / neutral / positive)
        st.bar_chart(df[["prob_negative", "prob_neutral", "prob_positive"]])

st.markdown("---")

# CSV upload area
st.subheader("Upload CSV (column: text)")
uploaded = st.file_uploader("Upload CSV with a `text` column", type=["csv"])
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        if "text" not in df_in.columns:
            st.error("CSV must have a column named `text`.")
        else:
            st.info(f"Predicting {len(df_in)} rows...")
            texts = df_in["text"].astype(str).tolist()
            results = predict_batch(
                tokenizer, model, device, id2label, texts, max_length=max_length
            )
            out_df = pd.DataFrame(results)
            st.dataframe(out_df)
            csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

st.markdown("---")
st.write(
    "Tip: first run will download model weights and tokenizer to your HF cache (~ a few 10s MBs). Subsequent runs are faster."
)