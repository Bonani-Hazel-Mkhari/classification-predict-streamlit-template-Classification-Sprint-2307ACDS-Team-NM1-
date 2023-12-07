"""

    Simple Streamlit webserver application for serving developed classification
	models.

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
import joblib,os
<<<<<<< Updated upstream
=======
import joblib
import re 
import base64
>>>>>>> Stashed changes

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

<<<<<<< Updated upstream
# Required to let Streamlit instantiate our web app.  
=======
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
    selected_model = st.sidebar.selectbox("Select Models", ["Information", "Prediction"])

    # Building out the predication page
    if selected_model == "Prediction":
        st.subheader("🚀 Explore Various predictive model 🌟")
        st.markdown(" Enter you text>> choose model>> Classify!!")
    

        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")
        option = st.selectbox(
            'Select a model to use',
            ['Linear SVC', 'Logistic Regression', 'Nearest Neighbors']
        )

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
# Required to let Streamlit instantiate our web app.
>>>>>>> Stashed changes
if __name__ == '__main__':
	main()
