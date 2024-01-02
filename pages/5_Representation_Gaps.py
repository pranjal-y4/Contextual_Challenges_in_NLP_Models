import streamlit as st
import pandas as pd
import difflib

def load_translation_data(file_path, encoding="utf-8"):
    df = pd.read_csv(file_path, encoding=encoding)
    return df

def get_similarity_ratio(word1, word2):
    return difflib.SequenceMatcher(None, word1, word2).ratio()

def get_translation(word, translation_df):
    # Convert the input word to lowercase for case-insensitive matching
    word_lower = word.lower()

    # Calculate similarity scores for the 'hword' column (Hindi word)
    hindi_matches = translation_df[translation_df['hword'].notna()]
    hindi_matches['similarity'] = hindi_matches['hword'].apply(lambda x: get_similarity_ratio(word_lower, x.lower()))

    # Get the word with the highest similarity score
    best_hindi_match = hindi_matches.loc[hindi_matches['similarity'].idxmax(), 'hword']
    best_english_translation = hindi_matches.loc[hindi_matches['similarity'].idxmax(), 'eword']

    # Calculate similarity scores for the 'eword' column (English word)
    english_matches = translation_df[translation_df['eword'].notna()]
    english_matches['similarity'] = english_matches['eword'].apply(lambda x: get_similarity_ratio(word_lower, str(x).lower()))

    # Get the word with the highest similarity score
    best_english_match = english_matches.loc[english_matches['similarity'].idxmax(), 'eword']
    best_hindi_translation = english_matches.loc[english_matches['similarity'].idxmax(), 'hword']

    # Return the best matches
    return {
        'best_hindi_match': best_hindi_match,
        'best_english_translation': best_english_translation,
        'best_english_match': best_english_match,
        'best_hindi_translation': best_hindi_translation
    }

def main():
    # Set the title and subheading
    st.title("Challenges in NLP: Representation Gaps")
    st.subheader("Introduction")
    # Display introductory text
    st.write(
        "Representation gaps in Natural Language Processing (NLP) occur when the data used to train language models does not adequately cover all languages and dialects."
    )
    st.markdown(
        f"<div style='text-align: left;'><b> Aim : In this segment, we will employ an English-to-Hindi translation approach and aim to train our model to proficiently facilitate the translation between Hindi and English in both directions.</div>",
        unsafe_allow_html=True
    )

    # Load the translation data
    translation_path = 'English-Hindi.csv'
    translation_encoding = "utf-8"
    translation_df = load_translation_data(translation_path, encoding=translation_encoding)

    # Create an input text box for user input
    user_input = st.text_input("Enter a word (English or Hindi):", "")

    # Check if the user has entered a word
    if user_input:
        # Get the translation
        translation = get_translation(user_input, translation_df)
        st.write("Best Hindi Match:", translation['best_hindi_match'])
        st.write("Best English Translation:", translation['best_english_translation'])
        st.write("Best English Match:", translation['best_english_match'])
        st.write("Best Hindi Translation:", translation['best_hindi_translation'])

if __name__ == "__main__":
    main()






