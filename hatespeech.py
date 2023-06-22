from flask import Flask, request
import tensorflow as tf
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.utils import pad_sequences
import re
import emoji
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.add("s")
stop_words.add("th")
stop_words.add("nd")
stop_words.add("rd")

app = Flask(__name__)

# Load the tokenizer from the file
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model('fin.h5')

def pre(text):
    text = re.sub("RT", "", text)  # remove RT
    text = re.sub("#([a-zA-Z0-9_]+)", ' ', text)  # remove hash tag
    text = re.sub("@[A-Za-z0-9_]+", " ", text)  # remove mention
    text = emoji.demojize(text, delimiters=(" ", " "))  # replace emoji with text
    text = re.sub("_", ' ', text)
    text = text.lower()  # lowercase
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)  # remove website
    text = re.sub(":", " ", text)
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '', text)  # remove email
    text = re.sub(r'[!"\$%&\'()+,\-.\/:;=#@?\[\\\]^_`{|}~]', '', text)  # remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # keep only text
    text = text.split()  # tokenizing text
    text = str([word for word in text if not word in stop_words])  # remove stopwords
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])  # lemmatize

    return text


@app.route('/classify', methods=['POST'])
def classify_text():
    # Get the input text from the request
    text = request.form['text']

    # Preprocess the text
    cleaned_text = pre(text)

    # Tokenize the preprocessed text
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])

    # Pad the tokenized text to a fixed length
    padded_sequence = pad_sequences(tokenized_text, maxlen=500, padding='post', truncating='post')

    # Make predictions using the model
    predictions = model.predict(padded_sequence)
    predictions[0,2] += 0.1
    predicted_label = np.argmax(predictions, axis=1)[0]

    # Determine the result based on the prediction
    labels = ["Hate", "Safe", "Terrorism"]
    result = labels[predicted_label]

    # Return the result as the API response
    return f"Text is: {result}"


@app.route('/')
def check():
    return 'API is working...'


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
