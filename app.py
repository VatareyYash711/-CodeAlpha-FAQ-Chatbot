import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")

st.title("💬 FAQ Chatbot")
st.write("Ask a question about the sample topic: CodeAlpha AI internship.")

# FAQ dataset
FAQS = [
    ("What is CodeAlpha?", "CodeAlpha is a software development company that offers internship opportunities and project-based learning."),
    ("How many tasks must I complete?", "The task sheet says interns must complete at least two or three tasks from the four listed tasks. Confirm the exact requirement with your coordinator."),
    ("Where should I upload my source code?", "The task sheet instructs interns to upload complete source code to GitHub in a repository named CodeAlpha_ProjectName."),
    ("How do I submit my project?", "The task sheet says a submission form will be shared in the respective WhatsApp group. Submit completed work through that form."),
    ("Do I need to post a project video?", "Yes. The task sheet asks interns to post a video explanation on LinkedIn with the GitHub repository link."),
    ("What is a language translation tool?", "It is an application that accepts text and translates it from a selected source language into a target language."),
    ("What is NLP?", "Natural Language Processing is a field of AI that helps computers process and understand human language."),
    ("What is object detection?", "Object detection identifies objects in images or video frames and typically draws boxes and labels around them."),
    ("What is an FAQ chatbot?", "An FAQ chatbot compares a user's question with a set of stored frequently asked questions and returns the closest matching answer."),
    ("What is machine learning?", "Machine learning is a branch of AI in which systems learn patterns from data to make predictions or decisions."),
]

# Prepare FAQ questions and answers
questions = [q for q, _ in FAQS]
answers = [a for _, a in FAQS]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

faq_matrix = vectorizer.fit_transform(questions)

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display previous messages
for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

# User input
user_question = st.chat_input("Type your question...")

if user_question:
    st.session_state.history.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    # Find the closest FAQ
    query_vector = vectorizer.transform([user_question])
    scores = cosine_similarity(query_vector, faq_matrix)[0]

    best_idx = int(scores.argmax())
    score = float(scores[best_idx])

    # Generate response
    if score < 0.12:
        response = (
            "Sorry, I couldn't find a close FAQ match. "
            "Try asking about submissions, GitHub, internship tasks, NLP, or object detection."
        )
    else:
        response = answers[best_idx]
        response += f"\n\n_Matched FAQ: {questions[best_idx]}_"

    # Save and display response
    st.session_state.history.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.markdown(response)

# Sidebar
with st.sidebar:
    st.header("How it works")
    st.write("1. FAQ questions are stored in a list.")
    st.write("2. TF-IDF converts questions into numerical vectors.")
    st.write("3. Cosine similarity finds the closest FAQ.")
    st.write("4. The chatbot displays the matching answer.")

    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

# Display sample FAQs
with st.expander("View sample FAQs"):
    for q, a in FAQS:
        st.markdown(f"**Q: {q}**")
        st.write(f"**A:** {a}")