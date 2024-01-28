import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import re
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 

df = pd.read_csv("airline_reviews_cleaned.csv")
st.set_page_config(layout="wide", page_title="AeroAdvisor", page_icon="✈️")

css = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://cdn.discordapp.com/attachments/1075699203046641687/1165351110828101673/PhotoReal_From_a_distance_the_plane_appears_as_a_tiny_speck_ag_1.jpg?ex=654688cb&is=653413cb&hm=096dcc994304b93afd210555607d563f725d74c1dddfd392176f11e15076bcfa&");
background-size: 120%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stExpander"] {{
background: rgba(0,0,0,0.5);
border: 2px solid #000071;
border-radius: 10px;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

#-------------------------------- Function to clean reviews -------------------------------------#

# Check if wordnet is installed
try:                                                                         
    nltk.find("corpora/popular.zip")          
except LookupError:
    nltk.download('popular')


# Defining acronyms
acronyms_dict = pd.read_json('acronym.json', typ = 'series')

# Defining contractions
contractions_dict = pd.read_json('contractions.json', typ = 'series')

# Defining stopwords

alphabets = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
others = ["ã", "å", "ì", "û", "ûªm", "ûó", "ûò", "ìñ", "ûªre", "ûªve", "ûª", "ûªs", "ûówe", "ï", "ûï"]
stops = alphabets + others
stops = list(set(stops))

# Defining tokenizer
regexp = RegexpTokenizer("[\w']+")

# Preprocessing
def preprocess(text):
    
    text = text.lower()                                                                                        # lowercase
    text = text.strip()                                                                                        # whitespaces
    
    # Removing html tags
    html = re.compile(r'<.*?>')
    text = html.sub(r'', text)                                                                                 # html tags
    
    # Removing emoji patterns
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags = re.UNICODE)
    text = emoji_pattern.sub(r'', text)                                                                         # unicode char
    
    # Removing urls
    http = "https?://\S+|www\.\S+" # matching strings beginning with http (but not just "http")
    pattern = r"({})".format(http) # creating pattern
    text = re.sub(pattern, "", text)                                                                            # remove urls
    
    # Removing twitter usernames
    pattern = r'@[\w_]+'
    text = re.sub(pattern, "", text)                                                                            # remove @twitter usernames
    
    # Removing punctuations and numbers
    punct_str = string.punctuation + string.digits
    punct_str = punct_str.replace("'", "")
    punct_str = punct_str.replace("-", "")
    text = text.translate(str.maketrans('', '', punct_str))                                                     # punctuation and numbers
    
    # Replacing "-" in text with empty space
    text = text.replace("-", " ")                                                                               # "-"
    
    # Substituting acronyms
    words = []
    for word in regexp.tokenize(text):
        if word in acronyms_dict.index:
            words = words + acronyms_dict[word].split()
        else:
            words = words + word.split()
    text = ' '.join(words)                                                                                       # acronyms
    
    # Substituting Contractions
    words = []
    for word in regexp.tokenize(text):
        if word in contractions_dict.index:
            words = words + contractions_dict[word].split()
        else:
            words = words + word.split()
    text = " ".join(words)                                                                                       # contractions
    
    punct_str = string.punctuation
    text = text.translate(str.maketrans('', '', punct_str))                                                     # punctuation again to remove "'"
                                                                       
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in regexp.tokenize(text)])                             # lemmatize
    
    # Stopwords Removal
    text = ' '.join([word for word in regexp.tokenize(text) if word not in stops])                              # stopwords
    
    # Removing all characters except alphabets and " " (space)
    filter = string.ascii_letters + " "
    text = "".join([chr for chr in text if chr in filter])                                                      # remove all characters except alphabets and " " (space)
    
    # Removing words with one alphabet occuring more than 3 times continuously
    pattern = r'\b\w*?(.)\1{2,}\w*\b'
    text = re.sub(pattern, "", text).strip()                                                                    # remove words with one alphabet occuring more than 3 times continuously
    
    # return final output
    return text

#-------------------------------- Container 1 for Heading -------------------------------------#
container_1 = st.container()
with container_1:
    empty1, head2, empty3 = st.columns(spec = [2,3,2], gap = 'medium')
    with empty1:
        st.empty()
    with head2:
        st.title("Welcome Aboard")
        st.write("## Tell us about your Experience :airplane:")
    with empty3:
        st.empty()
        
#-------------------------------- Container 2 for main_content --------------------------------# 
container_2 = st.container()
with container_2:
    col1, col2, col3, col4 = st.columns(spec = [1,3,3,1], gap = 'medium')
    with col1:
        st.empty()

    with col2:
        expander_1 = st.expander(label = "Your Trip Info", expanded = True)
        with expander_1:
            airline = st.selectbox(
                label = "Enter your Airline",
                options = tuple(sorted(df['airline'].unique())),
                index = None,
                placeholder = "Choose an option..."
            )

            traveller_type = st.selectbox(
                label = "Enter your trip type",
                options = ("Business", "Solo Leisure", "Couple Leisure", "Family Leisure"),
                index = None,
                placeholder = "Choose an option..."
            )

            cabin = st.selectbox(
                label = "Enter your seat class",
                options = ("Economy Class", "Premium Economy", "Business Class", "First Class"),
                index = None,
                placeholder = "Choose an option..."
            )

            type_of_flight = st.radio(
                label = "Enter your flight type",
                options = ("Direct", 'Indirect'),
                index = 0,
            )

            frequency = st.radio(
                label = "How often do you fly?",
                options = ('Often', 'Occasionally', 'Rarely'),
                index = 1,
            )

    with col3:
        expander_2 = st.expander(label = "Your Ratings", expanded = True)
        with expander_2:
            seat_comfort = st.slider(
                label = "How comfortable are you with your seat?",
                min_value = 1,
                max_value = 5,
                value = 3
            )

            cabin_service = st.slider(
                label = "Please Rate your Cabin Service",
                min_value = 1,
                max_value = 5,
                value = 3
            )

            food_bev = st.slider(
                label = "Please rate the quality of food/beverages",
                min_value = 1,
                max_value = 5,
                value = 3
            )

            entertainment = st.slider(
                label = "Please rate the Entertainment Service",
                min_value = 1,
                max_value = 5,
                value = 3
            )

            ground_service = st.slider(
                label = "Please rate the Ground Service",
                min_value = 1,
                max_value = 5,
                value = 3
            )

            value_for_money = st.slider(
                label = "Value for Money Rating",
                min_value = 1,
                max_value = 5,
                value = 3
            )
        
    with col4:
        st.empty()

#-------------------------------- Container 3 for Final Rating Slider --------------------------------#
container_3 = st.container()
with container_3:
    empty1, head2, empty3 = st.columns(spec = [1,3,1], gap = 'medium')
    with empty1:
        st.empty()
    with head2:
        overall = st.slider(
            label = "How was your overall experience with the Airline?",
            min_value = 1,
            max_value = 10,
            value = 7
        )

        review = st.text_area("Enter your review")
    with empty3:
        st.empty()


#-----------------------------------------------------------------------------------------------------#
# Creating DataFrame using values input by user
temp_df = pd.DataFrame(
                data = [[airline, traveller_type, cabin, type_of_flight, frequency,
                            seat_comfort, cabin_service, food_bev, entertainment,
                            ground_service, value_for_money, overall]], 
                columns = ['airline', 'traveller_type', 'cabin', 'type_of_flight', 'frequency',
                            'seat_comfort', 'cabin_service', 'food_bev', 'entertainment',
                            'ground_service', 'value_for_money', 'overall']
)

# Loading Model using joblib file
model = joblib.load('ml_model.joblib')

# Defining a function to store the nlp_model in streamlit cache memory
@st.cache_resource
def cache_model(model_name):
    model = tf.keras.models.load_model(model_name)
    return model

# Loading the nlp_model
nlp_model = cache_model("nlp_model")

#-------------------------------- Container 4 for Final Predictions --------------------------------#

container_4 = st.container()
with container_4:
    empty1, head2, empty3 = st.columns(spec = [1.2,2,1.5], gap = 'medium')
    with empty1:
        st.empty()
        
    with head2:
        # Creating a button to get prediction
        if st.button('Get Prediction'):
            y_pred = model.predict(temp_df)
            y_pred_prob = model.predict_proba(temp_df)

            if review=="":
                st.warning("Please enter your review")
                st.stop()

            clean_review = preprocess(review)
            review_pred_proba = nlp_model.predict([clean_review])
            review_pred = np.where(review_pred_proba > 0.5, 1, 0)[0][0]

            if (y_pred[0] == 'yes') & (review_pred == 1):
                st.success("Thank you for your positive feedback! \nWe're delighted to hear that you had a great experience with our service.")
                st.balloons()
            elif (y_pred[0] == 'yes') & (review_pred == 0):
                st.warning("We appreciate your positive rating, but we're sorry to hear about your concerns in the review. \nPlease share more details so we can address them and enhance your experience.")
            elif (y_pred[0] == 'no') & (review_pred == 0):
                st.error("We apologize for falling short of your expectations. \nYour feedback is valuable, and we're committed to improving. \nPlease provide specific details about your experience for us to better understand and address the issues.")
            elif (y_pred[0] == 'no') & (review_pred == 1):
                st.error("We're sorry to hear about your negative rating, but we're glad to see your positive comments in the review. \nWe'd like to learn more about your concerns to ensure we address any issues and enhance your satisfaction.")
    
    with empty3:
        st.empty()