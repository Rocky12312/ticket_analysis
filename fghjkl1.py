import io
import os
import csv
import rake
import pickle
import tensorflow
import numpy as np
import pandas as pd
from io import TextIOWrapper
from keras.layers import Embedding
from healthcheck import HealthCheck
from tensorflow.keras.models import load_model
from flask import Flask, make_response, request, jsonify, json, render_template
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
import logging

app = Flask(__name__)


with open('tokenizer.picklex', 'rb') as handle:
    vec = pickle.load(handle)


pl = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("feature.pkl", "rb")))


word2idx = vec.word_index
vocab_size = len(vec.word_index) + 1

word2vec = {}
with open(os.path.join("glove.6B.100d.txt")) as f:
    for line in f:
        values = line.split()
        word = values[0]
        arr = np.asarray(values[1:], dtype="float32")
        word2vec[word] = arr


num_words = min(vocab_size, len(word2idx)+1)
Embedding_dim = 100
embedding_matrix = np.zeros((num_words, Embedding_dim))
for word, i in word2idx.items():
    if i < num_words:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(num_words, Embedding_dim, weights=[embedding_matrix], input_length=22, trainable=False)


model = load_model("without_oov.h5")

logging.basicConfig(filename="flask.log", level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(threadName)s:%(message)s")
logging.info("All useful files loaded")

health = HealthCheck(app, "/hcheck")


def app_ab():
    return True, "i am good"


health.add_check(app_ab)


def transform(text):
    text_m = rake.rake(text)
    tok = []
    for k, j in text_m.items():
        tok.append(k)
    text_m1 = " ".join(tok)
    p_f1 = vec.texts_to_sequences([text_m1])
    pp_f1 = pad_sequences(p_f1, maxlen=22, padding='post')
    predictions = model.predict(pp_f1)
    out_class = predictions.argmax(axis=-1)
    key = [key for (key, value) in pl.vocabulary.items() if value == out_class[0]]
    return key[0]


@app.route('/')
def form():
    return """
    <html>
        <body>
            <h1>Tagging the tickets..</h1>
            </br>
            </br>
            <p> Insert the data CSV file and then download the Result
            <form action="/transform" method="post" enctype="multipart/form-data">
                <input type="file" name="data_file" class="btn btn-block"/>
                </br>
                </br>
                <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
            </form>
        </body>
    </html>
"""


@app.route('/transform', methods=["POST"])
def transform_view():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('data_file'))
        df = df.dropna()
        pic = []
        for rec in range(len(df["Description"])):
            pic.append(transform(df["Description"][rec]))
        df["predicted_category"] = pic
        response = make_response(df.to_csv())
        response.headers["Content-Disposition"] = "attachment; filename=result.csv"
        response.headers["Content-Type"] = "text/csv"
        app.logger.info("Logged in")
        return response


if __name__ == "__main__":
    app.run(debug=True)
