import streamlit as st
import pandas as pd
import plotly.express as px  # Import Plotly Express for plotting
from sklearn.model_selection import train_test_split, GridSearchCV
import plotly.graph_objects as go  # Import Plotly Graph Objects for additional chart customization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import chardet
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np


def load_dataset(file_path):
    detected_encoding = detect_encoding(file_path)
    st.write(f'Detected Encoding: {detected_encoding}')
    df = pd.read_csv(file_path, encoding=detected_encoding)
    return df

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def preprocess_data(df):
    # Display general information about the dataset
    st.subheader("Dataset Information:")
    # Center the subheading
    st.markdown(
        f"<div style='text-align: left;'>Let's try to explore the dataset and find important insights.</div>",
        unsafe_allow_html=True
    )
    
    # Check for invalid values in 'Gender_Bias' column
    invalid_values_gender_bias = df.loc[~df['Gender_Bias'].isin([0, 1])]

    # Display invalid values in 'Gender_Bias' column
    st.subheader(f"Invalid Values in 'Gender_Bias' column:")
    
    # Center the subheading
    st.markdown(
        f"<div style='text-align: left;'>1. Invalid Values in gender bias are anything other than 0 or 1. "
        f"<br> 2. This data can include none because according to our problem statement it will also be invalid. "
        f"<br>Aim: Find these invalid values and try to deal with it to improve the performance of the system.</div>",
        unsafe_allow_html=True
    )
    
    # Check for missing values in 'Type' column
    missing_values_type = df[df['Type'].isnull()]

    # Display missing values in 'Type' column
    st.subheader("Missing Values in 'Type' column:")
    if not missing_values_type.empty:
        st.dataframe(missing_values_type)
        st.write(f"Number of entries with missing values in 'Type' column: {len(missing_values_type)}")

        # Replace missing values with "Not Applicable"
        df['Type'].fillna("Not Applicable", inplace=True)
    else:
        st.write("No missing values found in 'Type' column.")
        
    # Find rows with invalid values in 'Gender_Bias' column
    invalid_values_gender_bias = df[~df['Gender_Bias'].isin([0, 1])]

    # Check if there are invalid values
    if not invalid_values_gender_bias.empty:
        st.subheader(f"Invalid Values in 'Gender_Bias' column:")
        st.dataframe(invalid_values_gender_bias)
        st.write(f"Number of entries with invalid values in 'Gender_Bias' column: {len(invalid_values_gender_bias)}")
        
        # Drop rows with less than 5 entries if the count is less than 5
        if len(invalid_values_gender_bias) < 5:
            st.subheader("Dropping Rows with Invalid Values:")
            st.write(f"Number of entries with invalid values in 'Gender_Bias' column is less than 5. Dropping those rows.")
            df = df[df['Gender_Bias'].isin([0, 1])]
        else:
            st.subheader("Replacing Invalid Values with Mode:")
            # Replace invalid values with the mode of 'Gender_Bias'
            mode_gender_bias = df['Gender_Bias'].mode().values[0]
            df['Gender_Bias'].replace(to_replace=~df['Gender_Bias'].isin([0, 1]), value=mode_gender_bias, inplace=True)
    else:
        st.write("No invalid values found in 'Gender_Bias' column.")
        # Shuffle the rows in the dataset
        df = df.sample(frac=1).reset_index(drop=True)

    # Create a duplicate DataFrame
    df1 = df.copy()

    # Display the shuffled and processed data
    st.subheader("Shuffled and Processed Data:")
    df = df.sample(frac=1).reset_index(drop=True)

    # Center the subheading
    st.markdown(
        f"<div style='text-align: left;'>1. Rows have been shuffled <br> 2. Missing values in gender bias column are not found, missing values in type have been replaced with - Not Applicable. "
        f"<br>3. Invalid values in 'Gender_Bias' column are removed.<br> 4. Label Encoding for Type column.</div>",
        unsafe_allow_html=True
    )

    label_encoder = LabelEncoder()
    df['Type'] = label_encoder.fit_transform(df['Type'])

    return df


def tfidf_vectorization(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def logistic_regression(X_train_tfidf, y_gender_train, y_type_train, best_C_gender=None, best_C_type=None):
    # Logistic regression for gender prediction
    if best_C_gender is None:
        param_grid_gender = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        grid_search_gender = GridSearchCV(LogisticRegression(), param_grid_gender, cv=5, scoring='accuracy')
        grid_search_gender.fit(X_train_tfidf, y_gender_train)

        best_C_gender = grid_search_gender.best_params_['C']

    model_gender = LogisticRegression(C=best_C_gender)
    model_gender.fit(X_train_tfidf, y_gender_train)

    # Logistic regression for type prediction
    if best_C_type is None:
        param_grid_type = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        grid_search_type = GridSearchCV(LogisticRegression(), param_grid_type, cv=5, scoring='accuracy')
        grid_search_type.fit(X_train_tfidf, y_type_train)

        best_C_type = grid_search_type.best_params_['C']

    model_type = LogisticRegression(C=best_C_type)
    model_type.fit(X_train_tfidf, y_type_train)

    return model_gender, model_type, best_C_gender, best_C_type

def detect_bias(dataset, model_name, use_tfidf=True, use_gridsearch=True):
    # Extract the 'Sentence' column for bias detection
    X = dataset['Sentence']
    y_gender = dataset['Gender_Bias']
    y_type = dataset['Type']

    # Replace NaN values in 'Sentence' column with a dummy string
    X = X.fillna("NaN")

    # Split the dataset
    X_train, X_test, y_gender_train, y_gender_test, y_type_train, y_type_test = train_test_split(
        X,
        y_gender,
        y_type,
        test_size=0.2,
        random_state=42
    )

    # TF-IDF Vectorization
    global tfidf_vectorizer  # Declare tfidf_vectorizer as a global variable
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_vectorization(X_train, X_test)

    # Choose the model based on the user's radio button selection
    if model_name == 'Logistic Regression':
        model_gender, model_type, best_C_gender, best_C_type = logistic_regression(X_train_tfidf, y_gender_train, y_type_train)
    elif model_name == 'SVM':
        model_gender, model_type, best_C_gender, best_C_type = svm(X_train_tfidf, y_gender_train, y_type_train)
        pass
    elif model_name == 'Random Forest':
        model_gender, model_type, best_C_gender, best_C_type = random_forest(X_train_tfidf, y_gender_train, y_type_train)
        pass
    else:
        st.warning("Please select a model before checking gender bias.")
        return None, None, None, None, None  # Return None for all metrics if model_name is invalid

    # Make predictions for Gender_Bias
    predictions_gender = model_gender.predict(X_test_tfidf)

    # Make predictions for Type
    predictions_type = model_type.predict(X_test_tfidf)

    # Convert 'Gender_Bias' to integers
    y_gender_test = y_gender_test.astype(int)

    # Calculate metrics for Gender_Bias
    accuracy_gender = accuracy_score(y_gender_test, predictions_gender)
    precision_gender = precision_score(y_gender_test, predictions_gender)
    recall_gender = recall_score(y_gender_test, predictions_gender)
    f1_gender = f1_score(y_gender_test, predictions_gender)

    # Provide an if-else clause to interpret accuracy for Gender_Bias
    if accuracy_gender >= 0.8:
        interpretation_gender = "High accuracy, model is performing well for Gender_Bias."
    elif 0.6 <= accuracy_gender < 0.8:
        interpretation_gender = "Moderate accuracy, some improvements may be needed for Gender_Bias."
    else:
        interpretation_gender = "Low accuracy, model may need significant improvements for Gender_Bias."

    # Calculate metrics for Type
    accuracy_type = accuracy_score(y_type_test, predictions_type)
    precision_type = precision_score(y_type_test, predictions_type, average='weighted')
    recall_type = recall_score(y_type_test, predictions_type, average='weighted')
    f1_type = f1_score(y_type_test, predictions_type, average='weighted')

    # Provide an if-else clause to interpret accuracy for Type
    if accuracy_type >= 0.8:
        interpretation_type = "High accuracy, model is performing well for Type."
    elif 0.6 <= accuracy_type < 0.8:
        interpretation_type = "Moderate accuracy, some improvements may be needed for Type."
    else:
        interpretation_type = "Low accuracy, model may need significant improvements for Type."

    return (
        accuracy_gender, precision_gender, recall_gender, f1_gender, interpretation_gender,
        accuracy_type, precision_type, recall_type, f1_type, interpretation_type,
        model_name  # Include the model name as the last element
    )


def svm(X_train_tfidf, y_train_gender, y_train_type, best_C_gender=None, best_C_type=None):
    # Initialize models with default parameters
    model_gender = SVC(probability=True)  # Set probability to True for decision_function to work
    model_type = SVC(probability=True)

    # SVM for gender prediction
    if best_C_gender is None:
        best_C_gender = 1.0  # Default value if hyperparameter tuning is not performed

    # Update the model with the best hyperparameter
    model_gender = SVC(C=best_C_gender, probability=True)

    # Train SVM model for gender prediction
    model_gender.fit(X_train_tfidf, y_train_gender)

    # SVM for type prediction
    if best_C_type is None:
        best_C_type = 1.0  # Default value if hyperparameter tuning is not performed

    # Update the model with the best hyperparameter
    model_type = SVC(C=best_C_type, probability=True)

    # Train SVM model for type prediction
    model_type.fit(X_train_tfidf, y_train_type)

    return model_gender, model_type, best_C_gender, best_C_type



def remove_bias(dataset):
    # Replace gender-specific words with gender-neutral words
    gender_neutral_mapping = {
        r'\b(?:he|him|his)\b': 'they',
        r'\b(?:she|her)\b': 'they'
        # Add more mappings as needed
    }

    # Apply the replacement to the 'Sentence' column with capitalized first words
    for pattern, replacement in gender_neutral_mapping.items():
        dataset['Sentence'] = dataset['Sentence'].apply(lambda x: re.sub(pattern, replacement.capitalize(), x, flags=re.IGNORECASE))
    # Set 'Gender_Bias' column to 0 for all rows
    dataset['Gender_Bias'] = 0
    # Return the processed dataset
    return dataset

# Define the model variables outside the main function
model = None
tfidf_vectorizer = None

def predict_bias_and_type(sentence, model, tfidf_vectorizer, threshold=0.5):
    if model is None or tfidf_vectorizer is None:
        st.warning("Please select a model before making predictions.")
        return None, None

    # Tokenize the sentence and vectorize it using the same TF-IDF vectorizer
    sentence_tfidf = tfidf_vectorizer.transform([sentence])

    # Get the decision scores for the positive class (gender bias)
    decision_scores_gender = model[0].decision_function(sentence_tfidf)
    probability_gender = 1 / (1 + np.exp(-decision_scores_gender))  # Convert decision scores to probabilities
    prediction_gender = int(probability_gender > threshold)

    # Predict the type
    prediction_type = model[1].predict(sentence_tfidf)

    # Map numerical type predictions to names
    type_names = {
        0: "Not Applicable",
        1: "Appearance and Beauty Bias",
        2: "Double Standards",
        3: "---- ",
        4: "Language Bias",
        5: "Maternal Bias",
        6: "Not Applicable",
        7: "Occupational Bias",
        8: "Parental Leave Bias",
        9: "Role Stereotyping",
    }

    # Get the corresponding name for the type prediction
    type_name = type_names.get(prediction_type[0], "Unknown")
    
    # Display the gender bias prediction based on the value
    gender_bias_result = "Detected" if prediction_gender == 1 else "Not Detected"


    return gender_bias_result, type_name



def main():
    global model, tfidf_vectorizer  # Declare model and tfidf_vectorizer as global variables
    # st.title("Bias Detection Showcase")

    # Get the HTML element corresponding to the title
    title_html = """
        <div style="text-align: center; font-size: 52px;">
            <b>Bias Amplification: Gender Bias Detection</b>
        </div>
    """

            
    # Apply the HTML style to the title
    st.markdown(title_html, unsafe_allow_html=True)

        # Load the dataset
    st.subheader("Introduction")
    # Center the subheading
    st.markdown(
        f"<div style='text-align: left;'>Gender bias refers to the unequal treatment or discrimination based on an individual's gender, often favoring one gender over the other. This bias can manifest in various aspects of life, including education, employment, social interactions, and more. It involves preconceived notions, stereotypes, and expectations associated with being male or female.</div>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"<div style='text-align: left;'><br><b> Aim :In this segment, our aim is to predict whether a given sentence exhibits gender bias and, if so, identify its specific type based on the dataset crafted by the authors of this project.<br> </div>",
        unsafe_allow_html=True
    )
    # Load the dataset
    st.subheader("Original Dataset")

    dataset_path = 'gender-bias_1.csv'
    dataset = load_dataset(dataset_path)

    # Display the dataset
    st.dataframe(dataset)

    # Data Preprocessing
    st.subheader("Data Preprocessing")
    processed_dataset = preprocess_data(dataset.copy())
    st.dataframe(processed_dataset)
    
    
    # Create a duplicate DataFrame
    df1 = dataset.copy()
    
    # Create a columns layout for side-by-side charts
    col1, col2 = st.columns(2)

    # Custom colors in VIBGYOR combination
    custom_colors = ['#8e44ad', '#3498db', '#1f3a93', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    
    # Calculate types of biases for df1
    bias_types_df1 = df1['Type'].value_counts()

    # Pie chart showing the distribution of bias types for df1
    with col1:
        st.subheader("Distribution of Bias Types for Dataset")
        fig_bias_types_df1 = px.pie(
            names=bias_types_df1.index,
            values=bias_types_df1.values,
            title='Bias Types Distribution for df1',
            color_discrete_sequence=custom_colors
        )
        st.plotly_chart(fig_bias_types_df1, use_container_width=True)


    # Bar chart showing the comparison of Gender_Bias values
    with col2:
        st.subheader("Gender Bias Comparison")
        gender_bias_distribution_after = processed_dataset['Gender_Bias'].value_counts()
        fig_gender_bias_comparison = go.Figure()
        fig_gender_bias_comparison.add_trace(go.Bar(
            x=gender_bias_distribution_after.index,
            y=gender_bias_distribution_after.values,
            marker_color=['#e74c3c', '#2ecc71']
        ))
        fig_gender_bias_comparison.update_layout(
            title='Gender Bias Comparison',
            xaxis_title='Gender Bias',
            yaxis_title='Count',
            showlegend=False
        )
        st.plotly_chart(fig_gender_bias_comparison, use_container_width=True)

    st.subheader("Our Own Model for Implementation")
    st.markdown(
        f"<div style='text-align: left;'>1. Choose the model with which you'd like to start. The predictions presented in subsequent scenarios will be dependent on the selected model. "
        f"<br> 2. Evaluate different performance metrics and select the appropriate model for training. "
        f"<br>Aim: Once you've chosen the model, we will commence with the implementation.</div><br>",
        unsafe_allow_html=True
    )

    # Select model using radio buttons
    model_name = st.radio("Select Model", ["Logistic Regression", "SVM"])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        processed_dataset['Sentence'],
        processed_dataset['Gender_Bias'],
        test_size=0.2,
        random_state=42
    )

    # TF-IDF Vectorization
    global tfidf_vectorizer  # Declare tfidf_vectorizer as a global variable
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_vectorization(X_train, X_test)
    
    
    # Detect bias and measure metrics
    if st.button("Calculate"):
        (
            accuracy_gender, precision_gender, recall_gender, f1_gender, interpretation_gender,
            accuracy_type, precision_type, recall_type, f1_type, interpretation_type,
            selected_model  # Include the model name as the last element
        ) = detect_bias(processed_dataset, model_name, tfidf_vectorizer)

        st.write(f"{selected_model} Metrics:")
        st.write(f"Selected Model: {selected_model}")
        st.write(f"1. Accuracy: {accuracy_gender:.2%}")
        st.write(f"2. Precision: {precision_gender:.2%}")
        st.write(f"3. Recall: {recall_gender:.2%}")
        st.write(f"4. F1 Score: {f1_gender:.2%}")
        st.write(f"Interpretation: {interpretation_gender}")

        
    # Add a text input box for the user to enter a sentence
    user_sentence = st.text_area("Enter a sentence:")

    if st.button("Predict"):
        # Load the selected model

        if model_name == 'Logistic Regression':
            # Split the dataset to obtain training data
            X_train, X_test, y_gender_train, y_gender_test, y_type_train, y_type_test = train_test_split(
                processed_dataset['Sentence'],
                processed_dataset['Gender_Bias'],
                processed_dataset['Type'],
                test_size=0.2,
                random_state=42
            )
            
            # TF-IDF Vectorization
            X_train_tfidf, _, _ = tfidf_vectorization(X_train, X_test)

            model_gender, model_type, _, _ = logistic_regression(X_train_tfidf, y_gender_train, y_type_train)
        elif model_name == 'SVM':
            # Split the dataset to obtain training data
            X_train, X_test, y_gender_train, y_gender_test, y_type_train, y_type_test = train_test_split(
                processed_dataset['Sentence'],
                processed_dataset['Gender_Bias'],
                processed_dataset['Type'],
                test_size=0.2,
                random_state=42
            )
            
            # TF-IDF Vectorization
            X_train_tfidf, _, _ = tfidf_vectorization(X_train, X_test)

            model_gender, model_type, _, _ = svm(X_train_tfidf, y_gender_train, y_type_train)
        else:
            st.warning("Please select a model before predicting.")
            return

        # Check if the input sentence exhibits gender bias and predict the type
        if user_sentence:
            prediction_gender, prediction_type = predict_bias_and_type(user_sentence, (model_gender, model_type), tfidf_vectorizer)
            st.write(f"Gender Bias Prediction: {prediction_gender}")
            st.write(f"Type Prediction: {prediction_type}")
        else:
            st.warning("Please enter a sentence to make predictions.")

        

    # # Automatically remove bias when selecting a model
    # if st.button("Remove Bias"):
    #     processed_dataset = remove_bias(processed_dataset)
    #     # Display the processed dataset after bias removal
    #     st.subheader("Processed Dataset After Bias Removal")
    #     st.dataframe(processed_dataset)

    #     # Retrain the model after removing bias
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         processed_dataset['Sentence'],
    #         processed_dataset['Gender_Bias'],
    #         test_size=0.2,
    #         random_state=42
    #     )

    #     # TF-IDF Vectorization on the updated dataset
    #     X_train_tfidf, X_test_tfidf, _ = tfidf_vectorization(X_train, X_test)

    # # Detect bias and measure metrics after bias removal
    # if st.button("Calculate After Bias Removal"):
    #     accuracy_after, precision_after, recall_after, f1_after, interpretation_after, selected_model_after = detect_bias(processed_dataset, model_name, tfidf_vectorizer)
    #     st.write(f"{selected_model_after} Metrics After Bias Removal:")
    #     st.write(f"Accuracy: {accuracy_after:.2%}")
    #     st.write(f"Precision: {precision_after:.2%}")
    #     st.write(f"Recall: {recall_after:.2%}")
    #     st.write(f"F1 Score: {f1_after:.2%}")
    #     st.write(f"Interpretation: {interpretation_after}")

if __name__ == "__main__":
    main()
