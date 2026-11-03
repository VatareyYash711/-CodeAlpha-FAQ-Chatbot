# Task 2 — FAQ Chatbot

A Streamlit chatbot that matches a user's question to the closest stored FAQ using TF-IDF and cosine similarity.

## Main features
- Sample FAQ knowledge base
- Text preprocessing and TF-IDF vectorization
- Cosine-similarity matching
- Chat-style user interface
- Low-confidence fallback response

## Run
From the main project folder:
```bash
streamlit run task2_faq_chatbot/app.py
```

## Method
Questions in the FAQ list are transformed into TF-IDF vectors. The user's question is transformed in the same way. Cosine similarity compares the vectors, and the answer for the closest FAQ is displayed when the similarity score is high enough.
