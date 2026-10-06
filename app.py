# ============================================================
# NLP-Based Sentiment Classification
# European Money Market Text
# Custom FastText Embeddings + BiLSTM
# ============================================================

import streamlit as st
import numpy as np
import pickle
import json
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="European Money Market Sentiment Analysis",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_trained_model():
    """
    Loads the trained BiLSTM model.

    The FastText embedding weights are already stored
    inside the trained model, so the FastText model
    itself is not required during prediction.
    """

    return load_model("model_bi_lstm_model.h5")
import os, hashlib
p = "model_bi_lstm_model.h5"
st.write("exists:", os.path.exists(p))
if os.path.exists(p):
    st.write("size:", os.path.getsize(p))
    st.write("header:", open(p, "rb").read(8))
    st.write("md5:", hashlib.md5(open(p, "rb").read()).hexdigest())
model = load_trained_model()


# ============================================================
# LOAD TOKENIZER
# ============================================================

@st.cache_resource
def load_tokenizer():
    """
    Loads the tokenizer that was fitted on the training data.
    """

    with open("tokenizer.pkl", "rb") as file:
        return pickle.load(file)


tokenizer = load_tokenizer()


# ============================================================
# LOAD MAXIMUM SEQUENCE LENGTH
# ============================================================

@st.cache_resource
def load_max_length():
    """
    Loads the maximum sequence length used during training.
    """

    with open("max_len.pkl", "rb") as file:
        return pickle.load(file)


max_len = load_max_length()


# ============================================================
# LOAD LABEL MAP
# ============================================================

@st.cache_resource
def load_label_map():
    """
    Loads the mapping between class IDs and sentiment labels.

    Example:
        0 -> Negative
        1 -> Neutral
        2 -> Positive
    """

    with open("label_map.json", "r") as file:
        return json.load(file)


label_map = load_label_map()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def noramalize(text):
    """
    Preprocesses the input text.

    IMPORTANT:
    This preprocessing must be exactly the same as the
    preprocessing used during model training.
    """

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces
    return text.strip()


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

def predict_text(text):
    """
    Predicts the sentiment of the given input text.

    Returns:
        predicted_label
        prediction_probabilities
    """

    # Apply the same preprocessing used during training
    cleaned_text = noramalize(text)

    # Convert text into integer sequence using the saved tokenizer
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Pad the sequence to the same length used during training
    padded_sequence = pad_sequences(
        sequence,
        maxlen=max_len
    )

    # Generate prediction probabilities
    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    # Get the class having the highest probability
    class_id = int(np.argmax(prediction, axis=1)[0])

    # Convert class ID into sentiment label
    predicted_label = label_map[str(class_id)]

    return predicted_label, prediction[0]


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("📊 European Money Market Sentiment Analysis")

st.write(
    "NLP-based sentiment classification using "
    "Custom FastText Embeddings and BiLSTM."
)

st.divider()


# ============================================================
# TEXT INPUT
# ============================================================

user_input = st.text_area(
    "Enter European money market or financial text:",
    placeholder=(
        "Example: The European money market showed strong "
        "recovery as liquidity increased and short-term "
        "interest rates stabilized."
    ),
    height=150
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("Predict Sentiment", type="primary"):

    # Check whether the user entered text
    if not user_input.strip():

        st.warning(
            "Please enter some text before making a prediction."
        )

    else:

        # Generate prediction
        predicted_label, probabilities = predict_text(
            user_input
        )

        # ====================================================
        # DISPLAY PREDICTION
        # ====================================================

        st.subheader("Prediction")

        if predicted_label == "Positive":

            st.success(
                f"Sentiment: {predicted_label}"
            )

        elif predicted_label == "Negative":

            st.error(
                f"Sentiment: {predicted_label}"
            )

        else:

            st.info(
                f"Sentiment: {predicted_label}"
            )


        # ====================================================
        # DISPLAY PREDICTION CONFIDENCE
        # ====================================================

        class_id = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[class_id]
        )

        st.write(
            f"**Prediction Confidence: "
            f"{confidence * 100:.2f}%**"
        )


        # ====================================================
        # DISPLAY CLASS PROBABILITIES
        # ====================================================

        st.subheader("Confidence Scores")

        negative_probability = float(
            probabilities[0]
        )

        neutral_probability = float(
            probabilities[1]
        )

        positive_probability = float(
            probabilities[2]
        )


        # Negative probability
        st.write(
            f"**Negative:** "
            f"{negative_probability * 100:.2f}%"
        )

        st.progress(
            negative_probability
        )


        # Neutral probability
        st.write(
            f"**Neutral:** "
            f"{neutral_probability * 100:.2f}%"
        )

        st.progress(
            neutral_probability
        )


        # Positive probability
        st.write(
            f"**Positive:** "
            f"{positive_probability * 100:.2f}%"
        )

        st.progress(
            positive_probability
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NLP-Based Sentiment Classification of European Money "
    "Market Text using Custom FastText Embeddings and BiLSTM."
)
