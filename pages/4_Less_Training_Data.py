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
import seaborn as sns
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report





# Load data
gender_bias_data = pd.read_csv("gender_bias.csv")
actual_data = pd.read_csv("gender-bias_1.csv")

st.title("Less Training Data")
st.subheader("Impact of Insufficient Data")
st.markdown(
    f"<div style='text-align: left;'>"
    f"<br><br><b>Aim : Primarily in this section, we will explore how to handle scenarios where we have less training data.This can happen in real-world situations where collecting ample labeled data is challenging. We'll look at techniques such as transfer learning, data augmentation, and fine-tuning pre-trained models to achieve better performance even with limited training data.</b></div>",
    unsafe_allow_html=True
)




st.subheader("Dataset 1 (Less Data) ")
st.write(f"Number of rows: {len(gender_bias_data)}")
st.write(gender_bias_data)


st.subheader("Dataset 2 ")
st.write(f"Number of rows: {len(actual_data)}")

st.write(actual_data)

st.subheader("Data Exploration")
st.markdown(
        f"<div style='text-align: left;'>Firstly, let's compare both datasets and determine the amount of data we will be dealing with. Here, it's evident that Dataset 2 comprises approximately 21% of the total combined datasets.</div><br>",
        unsafe_allow_html=True
    )


# Plot pie chart for the number of rows
fig, ax = plt.subplots()
sizes = [len(gender_bias_data), len(actual_data)]
labels = ['Dataset 1', 'Dataset 2']
colors = ['#FFD700', '#DDA0DD']  # Gold Yellow and Light Purple
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig)

st.subheader("Comparative Analysis")
st.markdown(
        f"<div style='text-align: left;'>Now, let's comprehend the volume of data across different categories in a comparative manner using a double bar chart.</div><br>",
        unsafe_allow_html=True
    )

# Combine the two datasets
gender_bias_data['Dataset'] = 'Dataset 1'
actual_data['Dataset'] = 'Dataset 2'
combined_data = pd.concat([gender_bias_data, actual_data])

# Reset the index to avoid duplicate labels
combined_data = combined_data.reset_index(drop=True)

# Draw a comparative bar chart for the 'Type' column with rotated x-axis labels
plt.figure(figsize=(12, 6))
ax = sns.countplot(x='Type', hue='Dataset', data=combined_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate x-axis labels
plt.title('Comparative Analysis of Types in Datasets')
plt.xlabel('Type')
plt.ylabel('Count')
plt.legend(title='Dataset')

# Display the chart in Streamlit
st.pyplot(plt)

# Data preprocessing for gender_bias.csv
# Dropping Rows with Invalid Values
gender_bias_data = gender_bias_data.dropna(subset=['Gender_Bias', 'Type'])

# Replace NaN values in 'Type' column with "Not Applicable"
gender_bias_data['Type'] = gender_bias_data['Type'].fillna("Not Applicable")

# Shuffling and Processing Data
gender_bias_data = gender_bias_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Data preprocessing for gender_bias_1.csv
# Dropping Rows with Invalid Values
actual_data = actual_data.dropna(subset=['Gender_Bias', 'Type'])

# Replace NaN values in 'Type' column with "Not Applicable"
actual_data['Type'] = actual_data['Type'].fillna("Not Applicable")

# Shuffling and Processing Data
actual_data = actual_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine the datasets for label encoding
combined_data = pd.concat([gender_bias_data, actual_data])

# Label Encoding for Type column
label_encoder = LabelEncoder()
combined_data['Type'] = label_encoder.fit_transform(combined_data['Type'])

# Split the combined data back into individual datasets
gender_bias_data['Type'] = combined_data['Type'][:len(gender_bias_data)]
actual_data['Type'] = combined_data['Type'][-len(actual_data):]

# Split data into features and labels for 'gender_bias'
X_gender_bias = gender_bias_data['Sentence']
y_gender_bias_type = gender_bias_data['Type']  # Select the 'Type' column as the target variable

# Split data into training and testing sets for 'gender_bias'
X_train_gender_bias, X_test_gender_bias, y_train_gender_bias, y_test_gender_bias = train_test_split(
    X_gender_bias, y_gender_bias_type, test_size=0.2, random_state=42
)

# Vectorize the text data using TF-IDF for 'gender_bias'
vectorizer_gender_bias = TfidfVectorizer()
X_train_gender_bias_tfidf = vectorizer_gender_bias.fit_transform(X_train_gender_bias)
X_test_gender_bias_tfidf = vectorizer_gender_bias.transform(X_test_gender_bias)

# Train SVM model for 'gender_bias'
svm_gender_bias = SVC(random_state=42)
svm_gender_bias.fit(X_train_gender_bias_tfidf, y_train_gender_bias)  # Use SVM instead of Logistic Regression

# Evaluate the model for 'gender_bias'
y_pred_gender_bias_svm = svm_gender_bias.predict(X_test_gender_bias_tfidf)
# Add performance metrics for 'gender_bias' if needed


# Similar changes for 'actual_data'
# Split data into features and labels for 'actual_data'
X_actual_data = actual_data['Sentence']
y_actual_data_type = actual_data['Type']  # Select the 'Type' column as the target variable

# Split data into training and testing sets for 'actual_data'
X_train_actual_data, X_test_actual_data, y_train_actual_data, y_test_actual_data = train_test_split(
    X_actual_data, y_actual_data_type, test_size=0.2, random_state=43
)

# Vectorize the text data using TF-IDF for 'actual_data'
vectorizer_actual_data = TfidfVectorizer()
X_train_actual_data_tfidf = vectorizer_actual_data.fit_transform(X_train_actual_data)
X_test_actual_data_tfidf = vectorizer_actual_data.transform(X_test_actual_data)

# Train SVM model for 'actual_data'
svm_actual_data = SVC(random_state=43)
svm_actual_data.fit(X_train_actual_data_tfidf, y_train_actual_data)  # Use SVM instead of Logistic Regression

# Evaluate the model for 'actual_data'
y_pred_actual_data_svm = svm_actual_data.predict(X_test_actual_data_tfidf)
# Add performance metrics for 'actual_data' if needed

# Additional changes for multi-class classification
accuracy_gender_bias_svm = accuracy_score(y_test_gender_bias, y_pred_gender_bias_svm)
precision_gender_bias_svm = precision_score(y_test_gender_bias, y_pred_gender_bias_svm, average='macro')
recall_gender_bias_svm = recall_score(y_test_gender_bias, y_pred_gender_bias_svm, average='macro')
f1_gender_bias_svm = f1_score(y_test_gender_bias, y_pred_gender_bias_svm, average='macro')

accuracy_actual_data_svm = accuracy_score(y_test_actual_data, y_pred_actual_data_svm)
precision_actual_data_svm = precision_score(y_test_actual_data, y_pred_actual_data_svm, average='macro')
recall_actual_data_svm = recall_score(y_test_actual_data, y_pred_actual_data_svm, average='macro')
f1_actual_data_svm = f1_score(y_test_actual_data, y_pred_actual_data_svm, average='macro')

# Display performance metrics and model output
st.subheader("Performance Metrics")
col1, col2 = st.columns(2)



# Metrics for gender_bias.csv
col1.subheader("Metrics for Dataset 1")
col1.write(f"Accuracy: {accuracy_gender_bias_svm:.2f}")
col1.write(f"Precision: {precision_gender_bias_svm:.2f}")
col1.write(f"Recall: {recall_gender_bias_svm:.2f}")
col1.write(f"F1 Score: {f1_gender_bias_svm:.2f}")

# Metrics for gender_bias_1.csv
col2.subheader("Metrics for Dataset")
col2.write(f"Accuracy: {accuracy_actual_data_svm:.2f}")
col2.write(f"Precision: {precision_actual_data_svm:.2f}")
col2.write(f"Recall: {recall_actual_data_svm:.2f}")
col2.write(f"F1 Score: {f1_actual_data_svm:.2f}")


# Check Model Output for gender_bias.csv
st.subheader("Check Model Output for Dataset 1")
user_input_gender_bias = st.text_input("Enter a sentence:")

# Check if the user has entered a sentence
if user_input_gender_bias:
    # Tokenize and vectorize the input sentence using the correct vectorizer
    input_vector_gender_bias = vectorizer_gender_bias.transform([user_input_gender_bias])

    # Ensure the number of features in user input does not exceed the number of features used during training
    if input_vector_gender_bias.shape[1] > X_train_gender_bias_tfidf.shape[1]:
        # Trim the number of features in the user input to match the training data
        input_vector_gender_bias = input_vector_gender_bias[:, :X_train_gender_bias_tfidf.shape[1]]

    # Ensure the number of features in user input matches the number of features used during training
    if input_vector_gender_bias.shape[1] == X_train_gender_bias_tfidf.shape[1]:
        # Make a prediction
        prediction_gender_bias_svm = svm_gender_bias.predict(input_vector_gender_bias)
        flag1= prediction_gender_bias_svm[0]
        if flag1==8:
            gender_bias="Not Detected"
        else:
            gender_bias="Detected"

        # Display the prediction
        st.subheader("Model Output for Dataset 1")
        st.write(f"Input Sentence: {user_input_gender_bias}")
        st.write(f"Prediction - Gender Bias {gender_bias}")
        st.write(f"Type: {label_encoder.inverse_transform([prediction_gender_bias_svm[0]])[0]}")


    else:
        st.write("Error: Number of features in the input sentence does not match the training data.")


# Check Model Output for gender_bias_1.csv
st.subheader("Check Model Output for Dataset 2")
user_input_actual_data = st.text_input("Enter a sentence:", key="user_input_actual_data_key")

# Check if the user has entered a sentence
if user_input_actual_data:
    # Tokenize and vectorize the input sentence using the correct vectorizer for 'gender_bias_1.csv'
    input_vector_actual_data = vectorizer_actual_data.transform([user_input_actual_data])

    # Ensure the number of features in user input does not exceed the number of features used during training
    if input_vector_actual_data.shape[1] > X_train_actual_data_tfidf.shape[1]:
        # Trim the number of features in the user input to match the training data
        input_vector_actual_data = input_vector_actual_data[:, :X_train_actual_data_tfidf.shape[1]]

    # Ensure the number of features in user input matches the number of features used during training
    if input_vector_actual_data.shape[1] == X_train_actual_data_tfidf.shape[1]:
        # Make a prediction
        prediction_actual_data_svm = svm_actual_data.predict(input_vector_actual_data)
        flag1= prediction_gender_bias_svm[0]
        if flag1==8:
            gender_bias="Not Detected"
        else:
            gender_bias="Detected"

        # Display the prediction
        st.subheader("Model Output for Dataset 2")
        st.write(f"Input Sentence: {user_input_actual_data}")
        st.write(f"Prediction - Gender Bias {gender_bias}")
        st.write(f"Type: {label_encoder.inverse_transform([prediction_actual_data_svm[0]])[0]}")


    else:
        st.write("Error: Number of features in the input sentence does not match the training data.")
