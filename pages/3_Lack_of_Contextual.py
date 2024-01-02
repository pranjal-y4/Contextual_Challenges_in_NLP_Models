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
    st.subheader("Introduction")

    st.write(
        "A simple chatbot with a lack of contextual understanding struggles to grasp the nuances and details of a conversation. It may misinterpret user inputs or fail to respond appropriately to complex queries, as it doesn't have the ability to remember past interactions or understand the broader context of a conversation."
    )
    
    st.markdown(
        f"<div style='text-align: left;'><b> Aim : In this segment, we'll utilize a straightforward dataset containing responses to commonly asked questions typically posed to chatbots. Our objective is to train a model on this dataset and then use it to predict responses for given sentences.</div>",
        unsafe_allow_html=True
    )

    conversation_path = '/Users/pranjalyadav/Desktop/Conversation.csv'
    conversation_encoding = "utf-8"
    conversation_df = load_conversation(conversation_path, encoding=conversation_encoding)

    vectorizer, similarity_matrix = train_chatbot(conversation_df)

    

    user_input = st.text_area("Enter your message:", "")
    simple_chatbot(conversation_df, user_input, vectorizer, similarity_matrix)

if __name__ == "__main__":
    main()





