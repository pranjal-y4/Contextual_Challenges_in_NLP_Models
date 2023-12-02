import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_conversation(file_path, encoding="utf-8"):
    df = pd.read_csv(file_path, encoding=encoding)
    return df

def train_chatbot(conversation_df):
    corpus = conversation_df['question'] + ' ' + conversation_df['answer']
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return vectorizer, similarity_matrix

def simple_chatbot(conversation_df, user_input, vectorizer, similarity_matrix):
    if user_input:
        user_input_tfidf = vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(user_input_tfidf, vectorizer.transform(conversation_df['question'])).flatten()
        most_similar_index = similarity_scores.argmax()
        bot_response = conversation_df['answer'][most_similar_index]
        st.text_area("Bot:", bot_response, key="bot_response")

def main():
    st.title("Simple Chatbot: Lack of Contextual Understanding")
    st.subheader("Exploring the Impact on Conversational Scenarios")

    st.write(
        "Natural Language Processing (NLP) models, while powerful, face challenges in understanding the context "
        "in which words or phrases are used. This lack of contextual understanding can lead to inappropriate or "
        "biased responses in various conversational scenarios, as the models may not consider the broader context."
    )

    conversation_path = '/Users/pranjalyadav/Desktop/Conversation.csv'
    conversation_encoding = "utf-8"
    conversation_df = load_conversation(conversation_path, encoding=conversation_encoding)

    vectorizer, similarity_matrix = train_chatbot(conversation_df)

    

    user_input = st.text_area("Enter your message:", "")
    simple_chatbot(conversation_df, user_input, vectorizer, similarity_matrix)

if __name__ == "__main__":
    main()





