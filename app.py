import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


# Create Flask app
app = Flask(__name__)

# Load the pickle model
model = joblib.load(open('Sentiment Analysis World cup 2022-tweets.joblib', 'rb'))


@app.route("/")
def Home():
    return render_template("index1.html")

# Define a route for the prediction endpoint
@app.route('/predict', methods=["POST"])

def predict():
    tweet = request.form['Review']  # Get the tweet from the form

    #Preprocess the review
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    all_stopwords.remove('no')
    all_stopwords.remove('but')
    all_stopwords.remove("won't")
    all_stopwords.remove("isn't")
    all_stopwords.remove("very")
    tweet = [ps.stem(word) for word in tweet if not word in set(all_stopwords)]
    tweet = ' '.join(tweet)
    print(tweet)

    # Make predictions using the loaded model
    prediction = model.predict([tweet])

    # Convert the prediction to sentiment labels
    sentiment = "positive" if prediction[0] == 1 else "negative"

    return render_template("index1.html", prediction_text="The sentiment is {}".format(sentiment))

# Run the app if executed directly
if __name__ == "__main__":
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)
    app.run(debug=True)
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
