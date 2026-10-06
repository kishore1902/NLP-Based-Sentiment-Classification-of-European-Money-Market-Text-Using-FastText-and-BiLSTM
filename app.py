# Import Streamlit for building the web application.
import streamlit as st

# Import NumPy for numerical operations and finding the predicted class.
import numpy as np

# Import pickle for loading the saved tokenizer and maximum sequence length.
import pickle

# Import JSON for loading the saved label mapping.
import json

# Import re for text preprocessing using regular expressions.
import re

# Import os for checking whether required files exist.
import os

# Import h5py for checking whether the H5 model file is a valid HDF5 file.
import h5py

# Import Keras model loader for loading the trained BiLSTM model.
from tensorflow.keras.models import load_model

# Import pad_sequences for converting token sequences into fixed-length sequences.
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

# Configure the Streamlit page.
st.set_page_config(
    page_title="European Money Market Sentiment Analysis",
    page_icon="📊",
    layout="centered"
)


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

# Name of the trained BiLSTM model.
MODEL_PATH = "model_bi_lstm_model.h5"

# Name of the saved tokenizer.
TOKENIZER_PATH = "tokenizer.pkl"

# Name of the saved maximum sequence length.
MAX_LEN_PATH = "max_len.pkl"

# Name of the saved label mapping.
LABEL_MAP_PATH = "label_map.json"


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_trained_model():

    # Check whether the model file exists.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' was not found in the application directory."
        )

    try:

        # Open the H5 file in read-only mode to verify that it is a valid HDF5 file.
        with h5py.File(MODEL_PATH, "r"):

            # If no exception occurs, the file can be opened as HDF5.
            pass

        # Load the trained Keras model.
        model = load_model(
            MODEL_PATH,
            compile=False
        )

        # Return the successfully loaded model.
        return model

    except Exception as e:

        # Display a useful error message instead of hiding the actual problem.
        st.error(
            "The model file could not be loaded."
        )

        st.error(
            f"Model file: {MODEL_PATH}"
        )

        st.error(
            f"Error: {str(e)}"
        )

        # Stop execution because prediction cannot work without the model.
        st.stop()


# Load the trained model.
model = load_trained_model()


# ---------------------------------------------------------
# LOAD TOKENIZER
# ---------------------------------------------------------

@st.cache_resource
def load_tokenizer():

    # Check whether tokenizer file exists.
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer file '{TOKENIZER_PATH}' was not found."
        )

    # Open the tokenizer file in binary read mode.
    with open(TOKENIZER_PATH, "rb") as file:

        # Load the tokenizer using pickle.
        tokenizer = pickle.load(file)

    # Return the tokenizer.
    return tokenizer


# Load the tokenizer.
tokenizer = load_tokenizer()


# ---------------------------------------------------------
# LOAD MAXIMUM SEQUENCE LENGTH
# ---------------------------------------------------------

@st.cache_resource
def load_max_length():

    # Check whether max length file exists.
    if not os.path.exists(MAX_LEN_PATH):
        raise FileNotFoundError(
            f"Maximum sequence length file '{MAX_LEN_PATH}' was not found."
        )

    # Open the file in binary read mode.
    with open(MAX_LEN_PATH, "rb") as file:

        # Load the saved maximum sequence length.
        max_len = pickle.load(file)

    # Return the maximum sequence length.
    return max_len


# Load maximum sequence length.
max_len = load_max_length()


# ---------------------------------------------------------
# LOAD LABEL MAP
# ---------------------------------------------------------

@st.cache_resource
def load_label_map():

    # Check whether label map exists.
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(
            f"Label map file '{LABEL_MAP_PATH}' was not found."
        )

    # Open the JSON file.
    with open(LABEL_MAP_PATH, "r") as file:

        # Load the label mapping.
        label_map = json.load(file)

    # Return the label mapping.
    return label_map


# Load the label map.
label_map = load_label_map()


# ---------------------------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------------------------

def noramalize(text):

    # Convert the entire text to lowercase.
    text = text.lower()

    # Remove URLs.
    text = re.sub(r"http\S+|www\S", " ", text)

    # Remove numbers.
    text = re.sub(r"\d+", "", text)

    # Remove punctuation and special characters.
    text = re.sub(r"[^a-z\s]", " ", text)

    # Replace multiple spaces with a single space.
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces.
    return text.strip()


# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------

def predict_text(text):

    # Apply exactly the same preprocessing used during training.
    cleaned_text = noramalize(text)

    # Convert the cleaned text into tokenizer word indices.
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Convert the sequence into the fixed length expected by the model.
    padded_sequence = pad_sequences(
        sequence,
        maxlen=max_len
    )

    # Generate prediction probabilities using the trained model.
    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    # Find the class with the highest probability.
    class_id = int(np.argmax(prediction, axis=1)[0])

    # Convert the class ID into the corresponding sentiment label.
    predicted_label = label_map[str(class_id)]

    # Return the predicted label and all class probabilities.
    return predicted_label, prediction[0]


# ---------------------------------------------------------
# STREAMLIT USER INTERFACE
# ---------------------------------------------------------

# Display the application title.
st.title("📊 European Money Market Sentiment Analysis")

# Display a short description of the project.
st.write(
    "NLP-based sentiment classification using Custom FastText Embeddings and BiLSTM."
)

# Add a horizontal divider.
st.divider()


# ---------------------------------------------------------
# TEXT INPUT
# ---------------------------------------------------------

# Create a text area where the user can enter financial text.
user_input = st.text_area(
    "Enter European money market or financial text:",

    # Provide an example inside the input box.
    placeholder=(
        "Example: The European money market showed strong recovery "
        "as liquidity increased and short-term interest rates stabilized."
    ),

    # Set the height of the input area.
    height=150
)


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

# Create the prediction button.
if st.button("Predict Sentiment", type="primary"):

    # Check whether the user entered any text.
    if not user_input.strip():

        # Display a warning if the input is empty.
        st.warning(
            "Please enter some text before making a prediction."
        )

    else:

        # Run the sentiment prediction.
        predicted_label, probabilities = predict_text(user_input)

        # Display the prediction heading.
        st.subheader("Prediction")

        # Display the result using different Streamlit message types.
        if predicted_label == "Positive":

            # Positive sentiment.
            st.success(
                f"Sentiment: {predicted_label}"
            )

        elif predicted_label == "Negative":

            # Negative sentiment.
            st.error(
                f"Sentiment: {predicted_label}"
            )

        else:

            # Neutral sentiment.
            st.info(
                f"Sentiment: {predicted_label}"
            )


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        # Find the predicted class.
        class_id = int(np.argmax(probabilities))

        # Get the probability of the predicted class.
        confidence = float(probabilities[class_id])

        # Display prediction confidence.
        st.write(
            f"**Prediction Confidence: {confidence * 100:.2f}%**"
        )


        # -------------------------------------------------
        # INDIVIDUAL CLASS PROBABILITIES
        # -------------------------------------------------

        # Display the confidence scores heading.
        st.subheader("Confidence Scores")


        # Get probability for Negative class.
        negative_probability = float(probabilities[0])

        # Display Negative probability.
        st.write(
            f"**Negative:** {negative_probability * 100:.2f}%"
        )

        # Display Negative probability bar.
        st.progress(negative_probability)


        # Get probability for Neutral class.
        neutral_probability = float(probabilities[1])

        # Display Neutral probability.
        st.write(
            f"**Neutral:** {neutral_probability * 100:.2f}%"
        )

        # Display Neutral probability bar.
        st.progress(neutral_probability)


        # Get probability for Positive class.
        positive_probability = float(probabilities[2])

        # Display Positive probability.
        st.write(
            f"**Positive:** {positive_probability * 100:.2f}%"
        )

        # Display Positive probability bar.
        st.progress(positive_probability)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

# Add a horizontal divider.
st.divider()

# Display project information at the bottom of the application.
st.caption(
    "NLP-Based Sentiment Classification of European Money Market Text "
    "Using Custom FastText Embeddings and BiLSTM."
)
