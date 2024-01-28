# Airline Recommendation System

## Overview

Welcome to the **Airline Recommendation System** project. The system combines machine learning and deep learning models to predict user experience for airline services based on customer ratings and sentiment analysis of customer reviews.

## Introduction

The goal of this project is to enhance the user experience in choosing airlines by leveraging customer ratings and sentiment analysis of reviews. Two datasets were utilized to achieve this:

- **airline_reviews_cleaned:** Contains ratings provided by customers based on various flight parameters. A machine learning model was trained to predict customer satisfaction.

- **airline_sentiment:** Consists of textual reviews posted by customers. A deep learning model was trained for sentiment analysis to determine if the sentiment is positive or negative.

Finally, a Streamlit app was developed to integrate both models and predict the overall user experience with airline services.


## Installation

To set up and run the project locally, follow these steps:

1. Clone the repository:

    ```
    git clone https://github.com/prithush92/Airline-Recommendation-System.git
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```
3. Download the pretrained nlp model from the link and extract in your current working directory: <a href="https://drive.google.com/file/d/1MprHXoAlVVEL5CJ8G_McujoXdT5iUDJv/view?usp=sharing">Link</a>
## Usage

Run the Streamlit app locally by executing the following command:

```
streamlit run app/app.py
```

Visit the provided localhost link to interact with the Airline Recommendation System.

## Streamlit App

The Streamlit app combines the outputs of both models to provide a comprehensive recommendation for airline services. A screenshot of the app is shared below.

![airline app screenshot](https://github.com/prithush92/Airline-Recommendation-System/assets/126896351/2cb54f3a-6750-4074-a56c-a3342d294f3f)


## Contributing

Please feel free to provide any feedback !! Both positive and negative feedbacks are highly appreciated.

<hr>

*Thank you for visiting this repository*
