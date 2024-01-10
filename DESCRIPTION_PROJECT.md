# Bias Detection (NLP)

## Introduction

This initiative represents one of our endeavors to comprehensively comprehend the phenomenon of bias in Natural Language Processing (NLP). We have meticulously curated bespoke datasets designed to elucidate diverse manifestations of bias within standard English sentences. The integration of various graphical representations enhances the interpretability of our data. Moreover, our project features innovative implementations of NLP concepts, leveraging different facets of bias in NLP models.


## Features

- App.py :The original dataset is loaded, and its structure is displayed.The dataset undergoes preprocessing, including handling missing values, removing invalid entries, and label encoding.Users can choose between Logistic Regression and SVM models for bias detection.The selected model is trained, and bias is detected, providing accuracy, precision, recall, and F1 score metrics.Prediction: Users can enter a sentence, and the model predicts gender bias and bias type.

- Stereotyping and Discrimination : This section focuses on detecting Stereotyping and Discrimination Bias in sentences using machine learning techniques. The application utilizes a Random Forest classifier to identify bias types present in the provided dataset. The key functionalities include data loading, preprocessing, model training, and user input classification.

- Misclassification using Sentiment Analysis : This section of application delves into the realm of sentiment analysis, a process aimed at determining the emotional tone expressed in a piece of text. Specifically, it explores the challenges of misclassification and the unintended consequences associated with inaccurate predictions in sentiment analysis models.Users can enter statements for sentiment analysis, and the application utilizes the TextBlob library to predict whether the text conveys a positive, negative, or neutral sentiment.

- Lack of Contextual Data - Explore the limitations of a basic chatbot with this Streamlit application. The chatbot presented here lacks contextual understanding, making it challenging to comprehend the intricacies and nuances of a conversation. As a result, it may struggle to interpret user inputs accurately or provide appropriate responses to complex queries. This limitation arises from the bot's inability to remember past interactions or understand the broader context of a conversation.

- Handling Insufficient Training Data: This delves into the challenges posed by scenarios with limited training data, emphasizing the impact on model accuracy and precision. The primary focus is to explore strategies for addressing the constraints imposed by a scarcity of training examples. The application demonstrates the implementation of these strategies and evaluates their effectiveness.Data Preprocessing involves handling missing values by dropping rows with null values in the 'Gender_Bias' and 'Type' columns and shuffling data for a balanced distribution. Both Dataset 1 and Dataset 2 undergo similar preprocessing steps, including missing value handling and data shuffling. The label encoding process combines both datasets and encodes the 'Type' column to facilitate model training. Model Training and Evaluation employ a Support Vector Machine (SVM) algorithm for both datasets, assessing performance using accuracy, precision, recall, and F1 Score metrics, providing insights into the model's effectiveness and generalization capabilities.

- Representation Gaps : Explores the issues arising from inadequate language representation in Natural Language Processing models. This section focuses on addressing these gaps by implementing an English-to-Hindi translation model, aiming to facilitate bidirectional translation between Hindi and English. Users can input a word in either language to observe the model's best matches and translations, highlighting the significance of comprehensive language coverage in NLP.The provided code for English-to-Hindi translation does not explicitly mention a specific algorithm for translation. Instead, it utilizes the difflib library, particularly the SequenceMatcher class, to calculate similarity ratios between words. This library is commonly employed for comparing sequences, such as strings, and determining their similarity.


- Multilingual : The provided code implements a multilingual model using Streamlit and the Google Translator API. This model translates a user-entered English sentence into Hindi, Marathi, Gujarati, and Spanish, demonstrating basic translation capabilities. Additionally, it incorporates OpenAI's GPT-3.5-turbo-instruct engine to detect bias in the translated sentences. The goal is to showcase a simple multilingual application that highlights translation and bias detection functionalities. Users can input an English sentence, and the code will provide translations into multiple languages, along with a bias detection analysis using the GPT-3.5-turbo-instruct engine.


## Prerequisites

1. Make sure you have Python installed on your machine. If not, you can download it from python.org.

2. Install Streamlit by running the following command in your terminal or command prompt: `pip install streamlit`

3. Install required Python packages by running: `pip install pandas difflib scikit-learn matplotlib seaborn`

4. If you haven't obtained your OpenAI API key, sign up on the OpenAI website and follow the instructions to get your API key. Set up an environment variable named my_api_key and assign your OpenAI API key as its value.


## Getting Started
1. Download the necessary language models and data from repo. 

2. Run the following command to start the Streamlit app: `streamlit run app.py`

3. After executing the command, Streamlit will provide a local server address (usually starting with http://localhost). Open your web browser and access this address to view and interact with the Streamlit app.

4. In the Streamlit app, you'll find input fields and buttons corresponding to the functionalities of each project. Enter relevant inputs and follow the instructions displayed on the Streamlit app to observe the results.

Note: Ensure that you have an active internet connection as some components, such as the OpenAI API, require internet access.



## License

## Authors 

- Dr.Siddharth Hariharan
- Ved T
- Pranjal Y
- Shyam S


