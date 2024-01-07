import os
import streamlit as st
from translate import Translator as GoogleTranslator
import openai
import typing

# Set your OpenAI API key from the environment variable
openai.api_key = os.getenv("my_api_key")


class CustomTranslator(GoogleTranslator):
    def __init__(self, from_lang='en', to_lang='en'):
        super().__init__(from_lang=from_lang, to_lang=to_lang)

def translate_text(input_text, target_language="hi"):
    translator = CustomTranslator(to_lang=target_language)

    # Translate the input text to the target language
    translated_text = translator.translate(input_text)

    return translated_text

def detect_bias(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100
    )

    return response.choices[0].text.strip()

def main():
    # Set the title and subheading
    st.title("Multilingual Model")
    st.subheader("Introduction")

    # Display introductory text
    st.write(
        "A multilingual model is like a smart computer program that can understand and work with different languages. When some languages or dialects are not given enough attention or included in the program, it can cause problems. These problems may include the program behaving unfairly or not working well when dealing with specific language situations."
    )
    st.markdown(
        f"<div style='text-align: left;'><b> Aim : This code implements a simple multilingual model using Streamlit and the Google Translate API. The goal is to demonstrate a basic translation capability between English and Hindi, as well as other languages. The model uses the Googletrans library for translation.</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div style='text-align: left;'><br>[ Reference : hi = Hindi-India , es = Español-Spanish , mr = Marathi-India , gu = Gujrati-India ]<br> <br></div>",
        unsafe_allow_html=True
    )

    # Create an input text box for user input
    user_input = st.text_input("Enter a sentence (English):", "")

    # Check if the user has entered a sentence
    if user_input:
        # Get the translation for each language
        translations = {
            "Hindi": translate_text(user_input, "hi"),
            "Marathi": translate_text(user_input, "mr"),
            "Gujarati": translate_text(user_input, "gu"),
            "Spanish": translate_text(user_input, "es"),
        }

        # Display the translations
        st.write("Input Text:", user_input)
        for lang, translated_text in translations.items():
            st.write(f"Translated Text ({lang}): {translated_text}")

            # Detect bias in the translated sentence
            translated_prompt = f"Detect bias in this {lang} sentence in NLP context: '{translated_text}'"
            bias_detection = detect_bias(translated_prompt)
            st.write(f"Bias Detection ({lang}): {bias_detection}")

if __name__ == "__main__":
    main()
