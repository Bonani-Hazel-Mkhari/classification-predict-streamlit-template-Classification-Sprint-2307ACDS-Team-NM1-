# sentiment_app.py
import streamlit as st
import pandas as pd
from textblob import TextBlob
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import joblib,os
import joblib
import re 

#modelling imports:
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

#global parameters
additional  = ['retweet', 've', 'RT']
all_stop = (stopwords.words('english'), additional)

#import and read csv's into Jupyter:
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test_with_no_labels.csv")

# Function to clean and preprocess tweets
all_stop = set(stopwords.words('english'))
def tweet_processing(tweet):
    def clean_tweet(tweet):
        tweet_blob = TextBlob(tweet)
        words = tweet_blob.words if isinstance(tweet_blob.words, list) else [tweet_blob.words]
        return ' '.join(tweet_blob.words)

    new_tweet = clean_tweet(tweet)

    def clean_stopwords(message):
        tweet_tokens = word_tokenize(message)
        clean_tokens = [t for t in tweet_tokens if not re.match(r'^RT$|https?://[^\s]+', t)]
        clean_tokens = [word.lower() for word in clean_tokens if word.isalpha() and word.lower() not in all_stop]
        return clean_tokens

    no_punc_tweet = clean_stopwords(new_tweet)

    def stem_words(sentence_arrays):
        stemmer = SnowballStemmer('english')
        stemmed_sentences = []

        for sentence_array in sentence_arrays:
            stemmed_array = []

            for word in sentence_array:
                stemmed_array.append(stemmer.stem(word))

            delimiter = ' '
            sentence = delimiter.join(stemmed_array)
            stemmed_sentences.append(sentence)

        return stemmed_sentences

    stemmed_words = stem_words(no_punc_tweet)

    def lemma(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = [lem.lemmatize(word, 'v') for word in tweet_list]
        return normalized_tweet

    return lemma(stemmed_words)

#split X and y data for modelling
X = df_train['message']
y = df_train['sentiment']
testX = df_test['message']

#train_test_split on data for modelling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

#create the final Pipeline with pre-processing, weighing and modelling combined into a few lines of code.
best_pipe = Pipeline([
    ('vect',CountVectorizer(analyzer=tweet_processing)),  #tokenize the tweets
    ('tfidf', TfidfTransformer()), #weight the classes
    ('classifier', LinearSVC()),
])
best_pipe.fit(X_train, y_train)

#make predictions from from fitted model
y_pred = best_pipe.predict(testX)

#create test sentiment column from predictions
df_test['sentiment'] = y_pred.tolist()

#subset columns for output format
df_final_sub = df_test[['tweetid', 'sentiment']]
#Export prediction data to .csv for Kaggle submission
df_final_sub.to_csv('final_prediction.csv', index=False)

def analyze_sentiment(cleaned_text):
    if isinstance(cleaned_text,list):
        cleaned_text = ' '.join(cleaned_text)
    prediction = best_pipe.predict([cleaned_text])[0]
    return prediction

news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

count_vectorizer = joblib.load("resources/count_vectorizer_model.pkl", "rb")
tweet_processor = joblib.load(count_vectorizer)
# Load your raw data
raw = pd.read_csv("resources/train.csv")
# Main function to load data and create the app
def main():
    st.title("Sentiment Analyser 🔥")

    # Add a sidebar with options
    st.sidebar.title("Options")
    selected_option = st.sidebar.selectbox("Select Option", ["About", "Sentiment Predictions", "FAQs", "App Reviews"])

    if selected_option == "About":
        # Provide information about the app
        st.subheader("About")
        st.write("This tool fetches the tweets from your file & prepocess your dataset and Analyzes Subjectivity and Polarity of tweets and predicts sentiments of tweets ")

    elif selected_option == "Sentiment Predictions":
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            # Read data from CSV
            df = pd.read_csv(uploaded_file)

            # Clean and preprocess tweets
            df['Cleaned_Text'] = df['message'].apply(tweet_processing)

            # Use the model for sentiment predictions
            df['Predicted_Sentiment'] = df['Cleaned_Text'].apply(analyze_sentiment)

            # Show the data with sentiment predictions
            st.subheader("Data with Sentiment Predictions")
            st.dataframe(df)

            if st.checkbox("Plot Most Prevalent Sentiment"):
                st.set_option('deprecation.showPyplotGlobalUse', False)

                def plot_sentiment_distribution(sentiment_column):
                    fig, ax = plt.subplots(figsize=(5, 5))
                    plt.hist(sentiment_column, bins=[-1, 0, 1, 2, 3], align='left', rwidth=0.8, color='skyblue', edgecolor='black')
                    plt.title('Class Distribution', fontweight="bold")
                    plt.xlabel('Sentiment')
                    plt.ylabel('Count')
                    plt.xticks(ticks=[-1, 0, 1, 2], labels=['Anti(-1)', 'Neutral(0)', 'Pro(1)', 'News(2)'])
                    st.pyplot(fig)

                # Function to plot sentiment distribution
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plot_sentiment_distribution(df['Predicted_Sentiment'])

        # Option to generate word cloud for different sentiments
        if st.checkbox("Generate Word Clouds"):
            # Function to generate word cloud
            def plot_word_cloud(text, title):
                wordcloud = WordCloud(background_color='white', max_words=200, max_font_size=100, random_state=42, width=800, height=400) 
                wordcloud.generate(str(text))
                plt.figure(figsize=(24, 16))
                plt.imshow(wordcloud)
                plt.title(title, fontdict={'size': 40, 'verticalalignment': 'bottom'})
                plt.axis('off')
                plt.tight_layout()

            # Combine all cleaned text
            all_cleaned_text = ' '.join(df['Cleaned_Text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)))

            # Plot general word cloud
            plot_word_cloud(all_cleaned_text, title="General Word Cloud")

            plot_word_cloud(df.loc[df['Predicted_Sentiment'] == -1]['Cleaned_Text'], title="Don't believe in climate change")
            plot_word_cloud(df.loc[df['Predicted_Sentiment'] == 0]['Cleaned_Text'], title="Neutral")
            plot_word_cloud(' '.join(df.loc[df['Predicted_Sentiment'] == 1]['Cleaned_Text']), title="Climate change believers")
            st.pyplot(plt.gcf())
            print(len(' '.join(df.loc[df['Predicted_Sentiment'] == -1]['Cleaned_Text'])))

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

        # Add a text box for user input
        user_review = st.text_area("Enter your review here:", "")
    
    st.sidebar.title("Models")
    selected_model = st.sidebar.selectbox("Select Models", ["Information", "Logistic Regression", "Linear SVC", "Nearest Neighbour"])

    # Building out the "Information" page
    if selected_model == "Information":
        st.subheader("🚀 Welcome to the Sentiment Prediction Galaxy! 🌟")
        # You can read a markdown file from supporting resources folder
        st.markdown("Explore the vast universe of machine learning models designed to decode sentiments with precision. From starry-eyed classifiers to cosmic regressors, embark on a journey to unveil the secrets of sentiment analysis! 🌌✨")

        #st.subheader("Raw Twitter data and label")
        #if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            #st.write(raw[['sentiment', 'message']]) # will write the df to the page

    # Building out the prediction page
    elif selected_model == "Logistic Regression":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
            prediction = predictor.predict(vect_text)

            # When the model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

    # Building out the prediction page
    elif selected_model == "Linear SVC":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_processor.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/linear_svc_model.pkl"), "rb"))
            prediction = predictor.predict(vect_text)

            st.success("Text Categorized as: {}".format(prediction))

    elif selected_model == "Nearest Neighbour":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/best_model.pkl"), "rb"))

if __name__ == "__main__":
    main()
