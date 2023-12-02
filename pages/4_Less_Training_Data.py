# gender-bias_1.csv 
# gender_bias.csv
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# Load data
gender_bias_data = pd.read_csv("/Users/pranjalyadav/Desktop/gender_bias.csv")
actual_data = pd.read_csv("/Users/pranjalyadav/Desktop/gender-bias_1.csv")

st.title("Less Training Data")
st.subheader("Impact of Insufficient Data")



st.subheader("Dataset: gender_bias.csv")
st.write(gender_bias_data)
st.write(f"Number of rows: {len(gender_bias_data)}")

st.subheader("Dataset: gender_bias_1.csv")
st.write(actual_data)
st.write(f"Number of rows: {len(actual_data)}")

# Plot pie chart for the number of rows
fig, ax = plt.subplots()
sizes = [len(gender_bias_data), len(actual_data)]
labels = ['gender_bias.csv', 'gender_bias_1.csv']
colors = ['#FFD700', '#DDA0DD']  # Gold Yellow and Light Purple
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig)

# Split data into features and labels
X_gender_bias = gender_bias_data['Sentence']
y_gender_bias = gender_bias_data['Gender_Bias']

X_actual_data = actual_data['Sentence']
y_actual_data = actual_data['Gender_Bias']

# Split data into training and testing sets
X_train_gender_bias, X_test_gender_bias, y_train_gender_bias, y_test_gender_bias = train_test_split(
    X_gender_bias, y_gender_bias, test_size=0.2, random_state=42
)

X_train_actual_data, X_test_actual_data, y_train_actual_data, y_test_actual_data = train_test_split(
    X_actual_data, y_actual_data, test_size=0.2, random_state=42
)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_gender_bias_tfidf = vectorizer.fit_transform(X_train_gender_bias)
X_test_gender_bias_tfidf = vectorizer.transform(X_test_gender_bias)

X_train_actual_data_tfidf = vectorizer.fit_transform(X_train_actual_data)
X_test_actual_data_tfidf = vectorizer.transform(X_test_actual_data)

# Train Random Forest models
# Train Logistic Regression models
lr_gender_bias = LogisticRegression(random_state=42)
lr_gender_bias.fit(X_train_gender_bias_tfidf, y_train_gender_bias)

lr_actual_data = LogisticRegression(random_state=42)
lr_actual_data.fit(X_train_actual_data_tfidf, y_train_actual_data)

# Evaluate models
y_pred_gender_bias_lr = lr_gender_bias.predict(X_test_gender_bias_tfidf)
y_pred_actual_data_lr = lr_actual_data.predict(X_test_actual_data_tfidf)

# Calculate performance metrics
accuracy_gender_bias = accuracy_score(y_test_gender_bias, y_pred_gender_bias_lr)
precision_gender_bias = precision_score(y_test_gender_bias, y_pred_gender_bias_lr, average='binary', pos_label=1)
recall_gender_bias = recall_score(y_test_gender_bias, y_pred_gender_bias_lr, average='binary')
f1_gender_bias = f1_score(y_test_gender_bias, y_pred_gender_bias_lr, average='binary')

accuracy_actual_data = accuracy_score(y_test_actual_data, y_pred_actual_data_lr)
precision_actual_data = precision_score(y_test_actual_data, y_pred_actual_data_lr, average='micro')  
recall_actual_data = recall_score(y_test_actual_data, y_pred_actual_data_lr, average='micro')  
f1_actual_data = f1_score(y_test_actual_data, y_pred_actual_data_lr, average='micro')
# Display performance metrics and model output
st.subheader("Performance Metrics")
col1, col2 = st.columns(2)

# Metrics for gender_bias.csv
# Metrics for gender_bias.csv
col1.subheader("Metrics for gender_bias.csv")
col1.write(f"Accuracy: {accuracy_gender_bias:.2f}")
col1.write(f"Precision: {precision_gender_bias:.2f}")
col1.write(f"Recall: {recall_gender_bias:.2f}")
col1.write(f"F1 Score: {f1_gender_bias:.2f}")

# Metrics for gender_bias_1.csv
col2.subheader("Metrics for gender_bias_1.csv")
col2.write(f"Accuracy: {accuracy_actual_data:.2f}")
col2.write(f"Precision: {precision_actual_data:.2f}")
col2.write(f"Recall: {recall_actual_data:.2f}")
col2.write(f"F1 Score: {f1_actual_data:.2f}")

# Check Model Output for gender_bias.csv
st.subheader("Check Model Output for gender_bias.csv")
user_input_gender_bias = st.text_input("Enter a sentence:")

# Check if the user has entered a sentence
if user_input_gender_bias:
    # Tokenize and vectorize the input sentence
    user_input_gender_bias_tfidf = vectorizer.transform([user_input_gender_bias])

    # Ensure the number of features in user input does not exceed the number of features used during training
    if user_input_gender_bias_tfidf.shape[1] > X_train_gender_bias_tfidf.shape[1]:
        # Trim the number of features in the user input to match the training data
        user_input_gender_bias_tfidf = user_input_gender_bias_tfidf[:, :X_train_gender_bias_tfidf.shape[1]]

    # Ensure the number of features in user input matches the number of features used during training
    if user_input_gender_bias_tfidf.shape[1] == X_train_gender_bias_tfidf.shape[1]:
        # Make a prediction
        prediction_gender_bias = lr_gender_bias.predict(user_input_gender_bias_tfidf)

        # Display the prediction
        st.subheader("Model Output for gender_bias.csv")
        col1, col2 = st.columns(2)
        col1.write(f"Input Sentence: {user_input_gender_bias}")
        col1.write(f"Prediction: {prediction_gender_bias[0]}")
    else:
        st.write("Error: Number of features in the input sentence does not match the training data.")

# Check Model Output for gender_bias_1.csv
st.subheader("Check Model Output for gender_bias_1.csv")
user_input_actual_data = st.text_input("Enter a sentence:", key="user_input_actual_data_key")

# Check if the user has entered a sentence
if user_input_actual_data:
    # Tokenize and vectorize the input sentence
    input_vector = vectorizer.transform([user_input_actual_data])

    # Ensure the number of features in user input does not exceed the number of features used during training
    if input_vector.shape[1] > X_train_actual_data_tfidf.shape[1]:
        # Trim the number of features in the user input to match the training data
        input_vector = input_vector[:, :X_train_actual_data_tfidf.shape[1]]

    # Ensure the number of features in user input matches the number of features used during training
    if input_vector.shape[1] == X_train_actual_data_tfidf.shape[1]:
        # Make a prediction
        prediction_actual_data = lr_actual_data.predict(input_vector)

        # Display the prediction
        st.subheader("Model Output for gender_bias_1.csv")
        col1, col2 = st.columns(2)
        col2.write(f"Input Sentence: {user_input_actual_data}")
        col2.write(f"Prediction: {prediction_actual_data[0]}")
    else:
        st.write("Error: Number of features in the input sentence does not match the training data.")
