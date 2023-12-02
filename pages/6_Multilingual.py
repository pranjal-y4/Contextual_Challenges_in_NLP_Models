import streamlit as st
from googletrans import Translator

def translate_text(input_text, target_language="hi"):
    translator = Translator()

    # Translate the input text to the target language
    translated_text = translator.translate(input_text, dest=target_language).text

    return translated_text

def main():
    # Set the title and subheading
    st.title("Multilingual Model")
    st.subheader("Understanding the Impact of Underrepresented Languages")

    # Display introductory text
    st.write(
        "Representation Gaps in Natural Language Processing (NLP) stem from the underrepresentation of certain languages "
        "and dialects. This underrepresentation can lead to biased behavior and poor performance in specific linguistic contexts."
    )

    # Create an input text box for user input
    user_input = st.text_input("Enter a sentence (English):", "")

    # Select target language
    target_language = st.selectbox("Select target language:", ["hi", "es", "mr", "gu"])  # Add more languages as needed

    # Check if the user has entered a sentence
    if user_input:
        # Get the translation
        translated_text = translate_text(user_input, target_language)
        st.write("Input Text:", user_input)
        st.write(f"Translated Text ({target_language}):", translated_text)

if __name__ == "__main__":
    main()












