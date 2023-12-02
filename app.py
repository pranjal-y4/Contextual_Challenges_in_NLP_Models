import streamlit as st
import pandas as pd
import plotly.express as px  # Import Plotly Express for plotting
from sklearn.model_selection import train_test_split, GridSearchCV
import plotly.graph_objects as go  # Import Plotly Graph Objects for additional chart customization
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import chardet
import re

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
    target_column = 'Gender_Bias'

    # Check for missing values
    missing_values = df.isnull().sum()

    # Check for values other than 0 and 1 in 'gender_bias' column
    invalid_values = df[target_column].loc[~df[target_column].isin([0, 1])]

    # Create a columns layout for side-by-side display
    col1, col2 = st.columns(2)

    # Display missing values in the first column
    with col1:
        st.subheader("Missing Values:")
        st.write(missing_values)

    # Display invalid values in the second column
    with col2:
        st.subheader(f"Invalid Values in '{target_column}' column:")
        st.write(invalid_values)

    # Replace missing values and invalid values with the mode of 'gender_bias'
    mode_gender_bias = df[target_column].mode().values[0]
    df[target_column].fillna(mode_gender_bias, inplace=True)

    if not invalid_values.empty:
        df[target_column].replace({val: mode_gender_bias for val in invalid_values}, inplace=True)


    # Center the subheading
    st.markdown(
        f"<div style='text-align: center;'>Values in '{target_column}' column have been replaced with the mode ({mode_gender_bias}).</div>",
        unsafe_allow_html=True
    )
    return df


def tfidf_vectorization(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def logistic_regression(X_train_tfidf, y_train, best_C=None):
    if best_C is None:
        # Hyperparameter tuning for logistic regression
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_tfidf, y_train)

        # Best hyperparameter
        best_C = grid_search.best_params_['C']

    # Train a logistic regression model with the best hyperparameter
    model = LogisticRegression(C=best_C)
    model.fit(X_train_tfidf, y_train)
    return model, best_C

def svm(X_train_tfidf, y_train, best_C=None):
    if best_C is None:
        # Hyperparameter tuning for SVM
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_tfidf, y_train)

        # Best hyperparameter
        best_C = grid_search.best_params_['C']

    # Train an SVM model with the best hyperparameter
    model = SVC(C=best_C)
    model.fit(X_train_tfidf, y_train)
    return model, best_C

def random_forest(X_train_tfidf, y_train, best_n_estimators=None):
    if best_n_estimators is None:
        # Hyperparameter tuning for Random Forest
        param_grid = {'n_estimators': [50, 100, 200, 300]}
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_tfidf, y_train)

        # Best hyperparameter
        best_n_estimators = grid_search.best_params_['n_estimators']

    # Train a Random Forest model with the best hyperparameter
    model = RandomForestClassifier(n_estimators=best_n_estimators)
    model.fit(X_train_tfidf, y_train)
    return model, best_n_estimators

def detect_bias(dataset, model_name, use_tfidf=True, use_gridsearch=True):
    # Extract the 'Sentence' column for bias detection
    X = dataset['Sentence']
    y = dataset['Gender_Bias']

    # Replace NaN values in 'Sentence' column with a dummy string
    X = X.fillna("NaN")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # TF-IDF Vectorization
    global tfidf_vectorizer  # Declare tfidf_vectorizer as a global variable
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_vectorization(X_train, X_test)

    # Choose the model based on the user's radio button selection
    if model_name == 'Logistic Regression':
        model, best_C = logistic_regression(X_train_tfidf, y_train)
    elif model_name == 'SVM':
        model, best_C = svm(X_train_tfidf, y_train)
    elif model_name == 'Random Forest':
        model, best_n_estimators = random_forest(X_train_tfidf, y_train)
    else:
        return None, None, None, None, None  # Return None for all metrics if model_name is invalid

    # Make predictions
    predictions = model.predict(X_test_tfidf)

    # Convert 'Gender_Bias' to integers
    y_test = y_test.astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Provide an if-else clause to interpret accuracy
    if accuracy >= 0.8:
        interpretation = "High accuracy, model is performing well."
    elif 0.6 <= accuracy < 0.8:
        interpretation = "Moderate accuracy, some improvements may be needed."
    else:
        interpretation = "Low accuracy, model may need significant improvements."

    return accuracy, precision, recall, f1, interpretation, model_name


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

# Define the model variable outside the main function
model = None


# Define the model variable outside the main function
model = None
tfidf_vectorizer = None

def check_gender_bias(sentence, threshold=0.5):
    global model, tfidf_vectorizer  # Declare model and tfidf_vectorizer as global variables
    if model is None or tfidf_vectorizer is None:
        st.warning("Please select a model before checking gender bias.")
        return None
    
    # Tokenize the sentence and vectorize it using the same TF-IDF vectorizer
    sentence_tfidf = tfidf_vectorizer.transform([sentence])

    # Get the probability of the positive class (gender bias)
    probability = model.predict_proba(sentence_tfidf)[:, 1]

    # Make a prediction based on the adjusted threshold
    prediction = int(probability > threshold)

    return prediction



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
    st.subheader("Original Dataset")

    dataset_path = 'gender-bias_1.csv'
    dataset = load_dataset(dataset_path)

    # Display the dataset
    st.dataframe(dataset)

    # Data Preprocessing
    st.subheader("Data Preprocessing")
    processed_dataset = preprocess_data(dataset.copy())
    st.dataframe(processed_dataset)
    
    # Create a columns layout for side-by-side charts
    col1, col2 = st.columns(2)

    # Define custom colors
    custom_colors = ['#3498db', '#e67e22', '#8e44ad', '#f0dab5']
    
    # Pie chart showing the distribution of Gender_Bias values before preprocessing
    with col1:
        st.subheader("Gender Bias Distribution Before Preprocessing")
        gender_bias_distribution_before = dataset['Gender_Bias'].value_counts()
        fig_before = px.pie(
            names=gender_bias_distribution_before.index,
            values=gender_bias_distribution_before.values,
            title='Gender Bias Distribution (Before Preprocessing)',
            color_discrete_sequence=custom_colors
        )
        st.plotly_chart(fig_before, use_container_width=True)

    # Pie chart showing the distribution of Gender_Bias values after preprocessing
    with col2:
        st.subheader("Gender Bias Distribution After Preprocessing")
        gender_bias_distribution_after = processed_dataset['Gender_Bias'].value_counts()
        fig_after = px.pie(
            names=gender_bias_distribution_after.index,
            values=gender_bias_distribution_after.values,
            title='Gender Bias Distribution (After Preprocessing)',
            color_discrete_sequence=custom_colors
        )
        st.plotly_chart(fig_after, use_container_width=True)



    # Create another columns layout for side-by-side bar charts
    col3, col4 = st.columns(2)

    # Bar chart showing the counts of Gender_Bias values before preprocessing
    with col3:
        st.subheader("Gender Bias Counts Before Preprocessing")
        fig_bar_before = go.Figure()
        fig_bar_before.add_trace(go.Bar(
            x=gender_bias_distribution_before.index,
            y=gender_bias_distribution_before.values,
            marker_color=['peachpuff', 'mediumpurple', 'palegoldenrod', 'lightsteelblue']
        ))
        fig_bar_before.update_layout(title='Gender Bias Counts (Before Preprocessing)',
                                    xaxis_title='Gender Bias',
                                    yaxis_title='Count',
                                    showlegend=False)
        st.plotly_chart(fig_bar_before, use_container_width=True)

    # Bar chart showing the counts of Gender_Bias values after preprocessing
    with col4:
        st.subheader("Gender Bias Counts After Preprocessing")
        fig_bar_after = go.Figure()
        fig_bar_after.add_trace(go.Bar(
            x=gender_bias_distribution_after.index,
            y=gender_bias_distribution_after.values,
            marker_color=['peachpuff', 'mediumpurple', 'palegoldenrod', 'lightsteelblue']
        ))
        fig_bar_after.update_layout(title='Gender Bias Counts (After Preprocessing)',
                                xaxis_title='Gender Bias',
                                yaxis_title='Count',
                                showlegend=False)
        st.plotly_chart(fig_bar_after, use_container_width=True)




    # Select model using radio buttons
    model_name = st.radio("Select Model", ["Logistic Regression", "SVM", "Random Forest"])

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
        accuracy, precision, recall, f1, interpretation, selected_model = detect_bias(processed_dataset, model_name, tfidf_vectorizer)
        st.write(f"{selected_model} Metrics:")
        st.write(f"Accuracy: {accuracy:.2%}")
        st.write(f"Precision: {precision:.2%}")
        st.write(f"Recall: {recall:.2%}")
        st.write(f"F1 Score: {f1:.2%}")
        st.write(f"Interpretation: {interpretation}")

    # Text input for user input sentence
    user_sentence = st.text_input("Enter a sentence:")
    threshold = 0.5

    
    # Check button to detect gender bias in the user input sentence
    if st.button("Check Gender Bias"):
        # Load the selected model
        if model_name == 'Logistic Regression':
            model, _ = logistic_regression(X_train_tfidf, y_train)
        elif model_name == 'SVM':
            model, _ = svm(X_train_tfidf, y_train)
        elif model_name == 'Random Forest':
            model, best_n_estimators = random_forest(X_train_tfidf, y_train)
            tfidf_vectorizer = tfidf_vectorization(X_train, X_test)[2]
        else:
            st.warning("Please select a model before checking gender bias.")
            return

        # Check if the input sentence exhibits gender bias
        prediction = check_gender_bias(user_sentence)
        if prediction is not None:
            st.write(f"Gender Bias Prediction: {prediction}")

    # Automatically remove bias when selecting a model
    if st.button("Remove Bias"):
        processed_dataset = remove_bias(processed_dataset)
        # Display the processed dataset after bias removal
        st.subheader("Processed Dataset After Bias Removal")
        st.dataframe(processed_dataset)

        # Retrain the model after removing bias
        X_train, X_test, y_train, y_test = train_test_split(
            processed_dataset['Sentence'],
            processed_dataset['Gender_Bias'],
            test_size=0.2,
            random_state=42
        )

        # TF-IDF Vectorization on the updated dataset
        X_train_tfidf, X_test_tfidf, _ = tfidf_vectorization(X_train, X_test)

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

