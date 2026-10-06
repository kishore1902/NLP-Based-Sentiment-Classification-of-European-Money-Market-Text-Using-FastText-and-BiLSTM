# NLP-Based Sentiment Classification of European Money Market Text

An NLP and deep learning-based sentiment classification system for analyzing textual information related to the European money market. The project explores custom-trained FastText word embeddings combined with recurrent neural network architectures for classifying financial text into Negative, Neutral, and Positive sentiment categories.

## Overview

Financial markets generate large volumes of textual information through reports, announcements, economic updates, and market commentary. Extracting sentiment from this text can help identify the overall tone of financial information.

This project applies Natural Language Processing (NLP) and Deep Learning techniques to classify European money market text into three sentiment categories:

- **Negative**
- **Neutral**
- **Positive**

The primary focus of the project is the use of **custom-trained FastText embeddings** for representing financial text, followed by deep learning models such as LSTM, BiLSTM, and GRU.

A BERT-based model was also implemented for comparison with the custom FastText-based approaches.

---

## Objectives

- Perform text preprocessing and normalization for financial text.
- Train custom FastText word embeddings on the training corpus.
- Convert textual data into numerical representations using FastText embeddings.
- Develop deep learning models for sentiment classification.
- Compare LSTM, BiLSTM, and GRU architectures.
- Evaluate the models using accuracy, precision, recall, and F1-score.
- Analyze model overfitting using training and validation performance.
- Compare custom FastText-based models with pretrained FastText and BERT.
- Deploy the selected sentiment classification model using Streamlit.

---

## NLP Pipeline

```text
Raw Financial Text
        ↓
Text Preprocessing
        ↓
Tokenization
        ↓
Custom FastText Embeddings
        ↓
Embedding Matrix
        ↓
Sequence Padding
        ↓
Deep Learning Model
        ↓
Sentiment Classification
        ↓
Negative / Neutral / Positive