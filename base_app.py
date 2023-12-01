# sentiment_app.py

import streamlit as st
import pandas as pd
from textblob import TextBlob  # Assuming you'll use TextBlob for sentiment analysis

# Function to analyze sentiments using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Main function to load data and create the app
def main():
    st.title("Sentiment Analyser 🔥")

    # Add a sidebar with options
    st.sidebar.title("Options")
    selected_option = st.sidebar.selectbox("Select Option", ["Sentiment Predictions", "Information", "FAQs", "App Reviews"])

    if selected_option == "Sentiment Predictions":
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            # Read data from CSV
            df = pd.read_csv(uploaded_file)

            # Show the uploaded data
            st.subheader("Uploaded Data")
            st.dataframe(df)

            # Analyze sentiments and add a new column to the DataFrame
            df['Sentiment'] = df['Text'].apply(analyze_sentiment)

            # Show the data with sentiment predictions
            st.subheader("Data with Sentiment Predictions")
            st.dataframe(df)

    elif selected_option == "Information":
        # Provide information about the app
        st.subheader("Information")
        st.write("This tool fetches the tweets from the Twitter site & performs the following tasks:")
        st.write("1. Converts it into a DataFrame")
        st.write("2. Cleans the text")
        st.write("3. Analyzes Subjectivity of tweets and adds an additional column for it")
        st.write("4. Analyzes Polarity of tweets and adds an additional column for it")
        st.write("5. Analyzes Sentiments of tweets and adds an additional column for it")

    elif selected_option == "FAQs":
        # Display frequently asked questions
        st.subheader("Frequently Asked Questions")
        st.write("1. **How to upload a CSV file?**")
        st.write("   - Use the 'Upload a CSV file' option in the 'Sentiment Predictions' section.")
        st.write("2. **What does the app do?**")
        st.write("   - The app analyzes sentiments from the text data in the uploaded CSV file.")

    elif selected_option == "App Reviews":
        # Provide functionality for analyzing app reviews
        st.subheader("App Reviews")
        st.write("Add your functionality for analyzing app reviews here.")

if __name__ == "__main__":
    main()
