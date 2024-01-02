import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Function for data preprocessing
def preprocess_data(df):
    # Display subheading for Data Preprocessing
    st.subheader("Data Preprocessing")

    # Check for missing values
    missing_values = df.isnull().sum()

    # Display missing values
    st.write("Missing Values:")
    st.write(missing_values)

    # Replace missing values in "Type of Bias" with the word "neutral"
    df['Type of Bias'].fillna('neutral', inplace=True)

    # Replace missing values in other columns with the mode
    df = df.apply(lambda x: x.fillna(x.mode().iloc[0]))

    # Display processed dataset
    st.subheader("Processed Data")
    st.dataframe(df)

    return df

# Function to show pie chart
def show_pie_chart(df, title, col):
    bias_counts = df[col].value_counts()
    fig = px.pie(
        names=bias_counts.index,
        values=bias_counts.values,
        title=title
    )
    return fig

# Function to show bar chart
def show_bar_chart(df, title, col_x, col_y):
    fig = px.bar(
        df,
        x=col_x,
        y=col_y,
        title=title,
        labels={col_x: 'Geographical Info', col_y: 'Count'},
        color=col_y
    )
    return fig

# Function to generate frequency distribution and plot bar chart
def generate_frequency_distribution_chart(text, title):
    tokens = word_tokenize(text)
    freq_dist = FreqDist(tokens)

    plt.figure(figsize=(10, 5))
    freq_dist.plot(20, cumulative=False)
    plt.title(f"Top 20 Words for {title}")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    st.pyplot(plt)

# Function for data preprocessing
def preprocess_data(df):
    # Replace missing values in "Type of Bias" with the word "neutral"
    df['Type of Bias'].fillna('neutral', inplace=True)

    # Replace missing values in other columns with the mode
    df = df.apply(lambda x: x.fillna(x.mode().iloc[0]))

    return df

# Function to train the classification model
def train_model(df):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['Text'],
        df['Type of Bias'],
        test_size=0.2,
        random_state=42
    )

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train a Random Forest classifier
    model = RandomForestClassifier()
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    predictions = model.predict(X_test_tfidf)

    # Display model performance
    accuracy = accuracy_score(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)

    st.subheader("Model Performance:")
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write("Classification Report:")
    st.code(classification_rep)

    return model, tfidf_vectorizer

# Function to classify user input
def classify_user_input(model, tfidf_vectorizer, user_input):
    # Tokenize the user input and vectorize it using the TF-IDF vectorizer
    user_input_tfidf = tfidf_vectorizer.transform([user_input])

    # Make a prediction
    prediction = model.predict(user_input_tfidf)

    return prediction[0]

def main():
    # Set the title and subheading
    st.title("Stereotyping and Discrimination Bias")
    # st.subheader("Our Dataset: stereotyping_dis.csv")

    # Load the dataset
    dataset_path = 'pages/stereotyping_dis.csv'
    dataset = load_dataset(dataset_path)

            # Load the dataset
    st.subheader("Introduction")
    # Center the subheading
    st.markdown(
        f"<div style='text-align: left;'>Stereotyping bias occurs when people make assumptions or form opinions about a group of individuals based on certain characteristics, often oversimplifying or generalizing. Discrimination bias involves treating individuals unfairly or unjustly due to factors such as race, gender, or other personal characteristics.</div>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"<div style='text-align: left;'><br><b> Aim :In this section, our goal is to train a model to accurately identify bias in sentences and determine its type based on the provided dataset. It's important to note that the dataset used for training is created by the authors of this project.<br> </div>",
        unsafe_allow_html=True
    )
    # Display the original dataset
    st.subheader("Original Data")
    st.dataframe(dataset)

    # Data preprocessing
    processed_dataset = preprocess_data(dataset)

    # Show pie charts for different types of biases before and after processing
    st.subheader("Biases Distribution Before and After Processing")
    st.markdown(
        f"<div style='text-align: left;'> 1. We have effectively replaced 'none' values in the 'Type of Bias' column with 'neutral', ensuring a proper foundation for further analysis.<br>2. Since this dataset did not necessitate any additional modifications, it remains unchanged for analysis. </div>",
        unsafe_allow_html=True
    )
    

    # Create a columns layout for side-by-side charts
    col1, col2 = st.columns(2)

    # Pie chart for biases distribution before processing
    with col1:
        st.plotly_chart(show_pie_chart(dataset, "Biases Distribution Before Processing", "Type of Bias"), use_container_width=True)

    # Pie chart for biases distribution after processing
    with col2:
        st.plotly_chart(show_pie_chart(processed_dataset, "Biases Distribution After Processing", "Type of Bias"), use_container_width=True)

    
    # Train the classification model
    st.subheader("Training the Classification Model")
    st.markdown(
        f"<div style='text-align: left;'>The Random Forest algorithm was employed to train this dataset, aiming to achieve optimal results.</div>",
        unsafe_allow_html=True
    )
    model, tfidf_vectorizer = train_model(processed_dataset)
    
    

    # User input for classification
    user_input = st.text_input("Enter a sentence for bias classification:")

    if st.button("Classify"):
        # Classify the user input
        prediction = classify_user_input(model, tfidf_vectorizer, user_input)
        st.subheader("Classification Result:")
        st.write(f"The sentence is classified as: {prediction}")

    

if __name__ == "__main__":
    main()






