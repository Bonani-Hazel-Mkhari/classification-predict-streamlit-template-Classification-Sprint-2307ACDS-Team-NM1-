"""
Simple Streamlit webserver application for serving developed classification models.

Author: Explore Data Science Academy.

Note:
---------------------------------------------------------------------
Please follow the instructions provided within the README.md file
located within this directory for guidance on how to use this script
correctly.
---------------------------------------------------------------------

Description: This file is used to launch a minimal streamlit web
application. You are expected to extend the functionality of this script
as part of your predict project.

For further help with the Streamlit framework, see:

https://docs.streamlit.io/en/latest/
"""
# Streamlit dependencies
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
import base64
import json
import requests  
from streamlit_lottie import st_lottie

#modelling imports:
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Function to convert image file to base64 encoding
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Get base64 encoding of an example image
img = get_img_as_base64("image8.jpg")

# Define background images and styles using HTML/CSS
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.pexels.com/photos/7130534/pexels-photo-7130534.jpeg");
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image8/png;base64,{img}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

# Apply the background styles using Markdown in Streamlit
st.markdown(page_bg_img, unsafe_allow_html=True)

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

#pipeline_path = os.path.join("resources", "Linearsvc_pipeline.pkl")
#pipeline = joblib.load(open(pipeline_path, "rb"))

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Main function to load data and create the app
# App layout with tabs as buttons
custom_css = """
    <style>
        .sidebar .css-1l0km6i {
            list-style-type: none !important;
            padding: 0 !important;
        }
        .sidebar .css-1l0km6i .stRadio > div {
            display: flex;
            align-items: center;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# App layout with tabs as buttons
tabs = ["Home 🏠", "Predictions 🚀", "Reviews ✍️", "FAQs ❓"]
selected_tab = st.sidebar.radio(" ", tabs)

# Display content based on the selected tab
if selected_tab == "Home 🏠":
    st.title("DisCava 🔥")
    st.subheader("🚀 Welcome to the Sentiment Prediction Galaxy! 🌟")
    st.markdown("Explore the vast universe of machine learning models designed to decode sentiments with precision. From starry-eyed classifiers to cosmic regressors, embark on a journey to unveil the secrets of sentiment analysis! 🌌✨")
    st.subheader("About Us: DS Matrix")
    st.markdown("DS Matrix Analytics is a top-tier data mining and solutions provider, empowering businesses in the digital era. As a trusted marketing partner, we deliver exceptional insights through our app and consulting services to optimize product marketing operations, gain competitive advantages, and drive growth. With deep expertise in Python, classification, data mining, and cleaning, we tailor solutions to diverse business needs. From streamlining processes to enhancing customer engagement, our comprehensive approach ensures tangible results. With a client-centric focus and unwavering commitment to quality, we build long-term partnerships based on trust and exceptional service.")
    st.subheader("Meet The Team")
    # Create five columns
    col1, col2,col3,col4,col5 = st.columns(5)

    # Column 1
    with col1:
        st.write("Bonani")
        # Absolute path to the image file on your local machine
        absolute_path = "https://cdn.discordapp.com/attachments/1175002737562365952/1183940325442588773/Picture1.jpg?ex=658a295a&is=6577b45a&hm=53f7e17ee1a0777130ef3c8d00cc57d60eae4475a26cebdbda64400f51fe3a6f&"
        # Display the image using st.image
        st.image(absolute_path, width=100)

    # Column 2
    with col2:
        st.write("Ngokoana")
        # Absolute path to the image file on your local machine
        absolute_path = "https://cdn.discordapp.com/attachments/1175002737562365952/1183940325950103594/Picture2.jpg?ex=658a295a&is=6577b45a&hm=d46cc42c0c7530c7d6d6907b1b53dd820e7629c87336a307b23cca2845f4a068&"
        # Display the image using st.image
        st.image(absolute_path, width=100)

    # Column 3
    with col3:
        st.write("Maria")
        # Absolute path to the image file on your local machine
        absolute_path = "https://cdn.discordapp.com/attachments/1175002737562365952/1183940326277255279/Picture3.jpg?ex=658a295a&is=6577b45a&hm=93ae7ed170196c2f30b5e925dcf7eed3b1cc811827e0a4afaced3922402d47b9&"
        # Display the image using st.image
        st.image(absolute_path, width=100)

     # Column 4
    with col4:
        st.write("Winnie")
        # Absolute path to the image file on your local machine
        absolute_path = "https://cdn.discordapp.com/attachments/1175002737562365952/1183940326566666261/Picture4.jpg?ex=658a295a&is=6577b45a&hm=eef52580ca8ec84566929b71e0206f31f4bb8f0f362d27df7f791cdfcbd1cfb8&"
        # Display the image using st.image
        st.image(absolute_path, width=100)  

    # Column 5
    with col5:
        st.write("Tshiamo")
        # Absolute path to the image file on your local machine
        absolute_path = "https://cdn.discordapp.com/attachments/1175002737562365952/1183940326860263444/Picture5.jpg?ex=658a295a&is=6577b45a&hm=0d0dc3f2def72bd281c69e1fab8468378899194571e9042190a541af209bb6f0&"
        # Display the image using st.image
        st.image(absolute_path, width=100)
    
    st.markdown("- Partner Associations\n- Data Mining & Application\n- Business Marketing Work")
    st.subheader("Our Clients")
    st.markdown(
    """
    <div style="display: flex;">
        <ul style="flex: 1; padding: 0; list-style-type: none;">
        <li>Ryobi</li>
        <li>Spar</li>
        <li>Bidvest</li>
        </ul>
        <ul style="flex: 1; padding: 0; list-style-type: none;">
        <li>Nissan</li>
        <li>Toyota</li>
        <li>Anglo American</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
    )
elif selected_tab == "Predictions 🚀":
    st.title("Sentiment Analyser")
    st.subheader("🚀 Explore Various Predictive Models 🌟")
    st.write("This is where you can make predictions and view prediction results.")
    st.write("You can choose to upload an entire file or enter a single message and predict the sentiment")
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader(" ", type=["csv"])
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

   
    # Single Sentiment
    st.subheader("Single Sentiment")
    st.markdown(" Enter you text >> choose model >> Classify!!")
    #st.sidebar.title("Models")
    #selected_model = st.sidebar.selectbox("Select Models", ["Linear SVC", "Logistic Regression","Nearest Neighbors"])

    # Creating a text box for user input
    tweet_text = st.text_area("Enter Text", "Type Here")
    option = st.selectbox(
    'Select a model to use',
    ['Linear SVC', 'Logistic Regression', 'Nearest Neighbors']
    )
    if option == "Linear SVC":
        if st.button("Classify"):
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/Linearsvc_pipeline.pkl"), "rb"))
            prediction = predictor.predict(vect_text)
            output_text = {
                '0': 'Neutral',
                '-1': 'Anti climate change',
                '1': 'Pro Climate change',
                '2': 'News'
            }
            # more human interpretable.
            st.success("Tweet categorized by {} model as: {}".format(option, output_text[str(prediction[0])]))
   
    
elif selected_tab == "Reviews ✍️":
    col1, col2, col3, col4= st.columns(4)

    # Column 1
    with col1:
        st.title("✍️Reviews")
    # Column 2
    with col4:
        absolute_path = "https://cdn.discordapp.com/attachments/1175002737562365952/1184071502597988422/SL-022123-56020-02.jpg?ex=658aa385&is=65782e85&hm=c5e6638c4e0e5b2f6a193f99b7c2701569860e0afc3f085f76c0973e5ac0556a&"
        # Display the image using st.image
        st.image(absolute_path,width=150)
    
    st.subheader("We would love to hear from you.")
    st.write("Share your thoughts about our app ✍️.")
    # Add a text box for user input
    user_review = st.text_area("Enter your review here:", "")

elif selected_tab == "FAQs ❓":
    st.title("FAQ❓")
    st.subheader("Frequently Asked Questions")
    st.write("1. **How to upload a CSV file?**")
    st.write("   - Use the 'Upload a CSV file' option in the 'Predictions' section.")
    st.write("2. **What does the app do?**")
    st.write("   - The app analyzes sentiments from the text data in the uploaded CSV file or typed sentiment.")
    st.write("3. **What is climate change?**")
    path = "https://www.youtube.com/watch?v=Sv7OHfpIRfU"
    st.video(path)