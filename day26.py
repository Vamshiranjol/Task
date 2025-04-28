# import warnings
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# warnings.filterwarnings('ignore')
# data=pd.read_csv("hospital.csv")
# # print(data)
# print(data.columns)
# df=pd.DataFrame(data)
# # x=df.drop(columns=["Sentiment Label","None"])
# def xyz(Ratings):
#     if Ratings >= 4:
#         return 'Positive'
#     else:
#         return 'Negative'

# # Assuming the rating column is called 'rating'
# df['sentiment'] = df['Ratings'].apply(xyz)

# print(df[['Ratings', 'sentiment']].head())
# print(xyz)

# print(data.columns)
# x=df.drop(columns=['sentiment'])
# y=['sentiment']
# #print(x)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


data=pd.read_csv('hospital_feedback.csv')
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text
    return ""
data['cleaned_text']=data['Feedback'].apply(clean_text)


data.set_index('Feedback', inplace=True)
print(data)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(data['cleaned_text'])
y=data['Rating']

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

model=MultinomialNB()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
print((classification_report(y_test, y_pred)))


import numpy as np
import pandas as pd
import re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize the Flask app
app = Flask(__name__)

# Load the CSV and prepare the data
data = pd.read_csv('hospital_feedback.csv')

# Clean the text (preprocessing function)
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text
    return ""

# Apply cleaning function to the 'Text' column
data['cleaned_text'] = data['Feedback'].apply(clean_text)

# Vectorize the cleaned text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['Rating']

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Define a function for cleaning new feedback and predicting sentiment
def predict_sentiment(feedback):
    cleaned_feedback = clean_text(feedback)
    X_new = vectorizer.transform([cleaned_feedback])
    sentiment = model.predict(X_new)[0]
    return sentiment

# Define routes
@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        # Get feedback from the form
        feedback = request.form['feedback']
        sentiment = predict_sentiment(feedback)
    return render_template('index26.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
    
    