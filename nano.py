import os
import rake
import pickle
import numpy as np
import tensorflow
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import load_model
from flask import Flask, jsonify, make_response, request, render_template, url_for

app = Flask(__name__)
padding_size = 22

with open('tokenizer.pickle', 'rb') as handle:
    vec = pickle.load(handle)

pl = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("feature.pkl", "rb")))

word2idx = vec.word_index

word2vec = {}
with open(os.path.join("glove.6B.100d.txt")) as f:
    for line in f:
        values = line.split()
        word = values[0]
        arr = np.asarray(values[1:], dtype="float32")
        word2vec[word] = arr

num_words = min(7300, len(word2idx) + 1)
Embedding_dim = 100
embedding_matrix = np.zeros((num_words, Embedding_dim))
for word, i in word2idx.items():
    if i < 7300:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

num_words = 7300
embedding_layer = Embedding(num_words, Embedding_dim, weights=[embedding_matrix], input_length=22, trainable=False)

model = load_model("imp1ls_space.h5")


def out_cat(text):
    text_m = rake.rake(text)
    tok = []
    for k, j in text_m.items():
        tok.append(k)
    text_m1 = " ".join(tok)
    p_f1 = vec.texts_to_sequences([text_m1])
    pp_f1 = pad_sequences(p_f1, maxlen=22, padding='post')
    predictions = model.predict(pp_f1)
    out_class = predictions.argmax(axis=-1)
    #key = [key for (key, value) in pl.vocabulary.items() if value == out_class[0]]
    return out_class[0]


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/SenCla', methods=['POST'])
def predict():
    if request.method == "POST":
        text = request.form["message"]
        print(text)
        res_wod = out_cat(text)
    return render_template("result.html", prediction=res_wod)


if __name__ == "__main__":
    app.run(debug=True)
