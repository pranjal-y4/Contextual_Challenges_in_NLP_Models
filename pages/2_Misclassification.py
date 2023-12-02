import streamlit as st
from textblob import TextBlob
import pandas as pd

# Function to load dataset
def load_dataset(file_path, encoding="utf-8", nrows=None):
    df = pd.read_csv(file_path, encoding=encoding, nrows=nrows)
    return df

# Function for sentiment analysis
def perform_sentiment_analysis(user_input):
    analysis = TextBlob(user_input)
    sentiment = analysis.sentiment.polarity

    # Classify sentiment
    if sentiment > 0:
        return "positive"
    elif sentiment < 0:
        return "negative"
    else:
        return "neutral"

def main():
    # Set the title and subheading
    st.title("Sentiment Analysis: Misclassification and Unintended Consequences")
    st.subheader("Exploring the Challenges of Text Classification")

    # Display introductory text
    st.write(
        "Natural Language Processing (NLP) models, while powerful, are not without their challenges. "
        "Misclassification and unintended consequences can arise, leading to potential harm. "
        "This Streamlit app aims to explore and discuss these challenges."
    )

    # Create an input text box for user input
    user_input = st.text_area("Enter a statement:", "")

    # Load the dataset
    dataset_path = '/Users/pranjalyadav/Desktop/target.csv'
    dataset_encoding = "ISO-8859-1"  # Replace with the correct encoding

    # Add pagination to display a subset of the dataset
    page_size = st.slider("Select number of rows to display:", 1, 100, 10)
    current_page = st.number_input("Enter page number:", 1, value=1)

    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size

    dataset = load_dataset(dataset_path, encoding=dataset_encoding, nrows=end_idx)

    # Display the original dataset with pagination
    st.subheader("Original Data")
    st.dataframe(dataset[start_idx:end_idx])

    # Button to trigger sentiment analysis
    if st.button("Analyze"):
        # Perform sentiment analysis
        result = perform_sentiment_analysis(user_input)

        # Display sentiment analysis result
        st.subheader("Sentiment Analysis Result:")
        st.write(f"Text: {user_input}")
        st.write(f"Predicted Sentiment: {result}")

    # Display cautionary note
    st.subheader("Caution:")
    st.write(
        "This is a simple example app for demonstration purposes. It uses a basic sentiment analysis approach. "
        "In a real-world scenario, careful consideration, testing, and monitoring are required to mitigate risks."
    )

if __name__ == "__main__":
    main()


