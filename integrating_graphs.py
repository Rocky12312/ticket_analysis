import io
import os
import dash
import base64
import pickle
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objs as go
from bert import BertModelLayer
from healthcheck import HealthCheck
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from flask import Flask, make_response, request, render_template
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer


server = Flask(__name__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=external_stylesheets,
    routes_pathname_prefix='/dash/'
)


bert_model_name = "uncased_L-12_H-768_A-12"
bert_ckpt_dir = os.path.join("model/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))


with open('classes.pkl', 'rb') as handle:
    classes1 = pickle.load(handle)


def create_model(max_seq_len, bert_ckpt_file):
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)

    print("bert shape", bert_output.shape)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=len(classes1), activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    load_stock_weights(bert, bert_ckpt_file)

    return model


def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return df


model_x = create_model(44, bert_ckpt_file)
model_x.load_weights("bert_ext.h5")


logging.basicConfig(filename="flask.log", level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(threadName)s:%(message)s")
logging.info("All useful files loaded")


health = HealthCheck(server, "/hcheck")


def app_ab():
    return True, "i am good"


health.add_check(app_ab)

print(44)


def transform(tex_arr):
    tokens = tokenizer.tokenize(tex_arr)
    tokens = ["[CLS]"] + tokens
    if len(tokens) >= 44:
        tokens = tokens[0:44 - 2]
    tokens = tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = token_ids + [0] * (44 - len(token_ids))
    return token_ids


color = {"background": "#111111", "text": "#7FDBFF"}

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
} 

app.layout = html.Div([
    dcc.Upload(
        id="upload-data",
        children=html.Div([
            "Drag and Drop or ",
            html.A("Select Files")
        ]),
        style={
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px"
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    dcc.Graph(id="Mygraph"),
    html.Div(id="output-data-upload")
])


@app.callback(Output("Mygraph", "figure"), [
    Input("upload-data", "contents"),
    Input("upload-data", "filename")
])
def update_graph(contents, filename):
    x = []
    y = []
    if contents:
        contents = contents[0]
        filename = filename[0]
        df_n = parse_data(contents, filename)
        df_n = df_n.set_index(df_n.columns[0])
        x = df_n["predicted_category"].unique()
        y = df_n["predicted_category"].value_counts().to_numpy()
    fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=y
            )
            ],
        layout=go.Layout(
            plot_bgcolor=color["background"],
            paper_bgcolor=color["background"],
            font={"color": color["text"]},
        ))
    return fig


@server.route('/')
def form():
    return render_template("home.html")


@server.route('/transform', methods=["POST"])
def transform_view():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('data_file'))
        df = df.dropna(subset=["Description"], axis=0)
        df.reset_index(inplace=True, drop=True)
        list_n = []
        for rec in range(len(df["Description"])):
            a = transform(df["Description"][rec])
            list_n.append(a)
        token_input = np.array(list_n)
        predictions = model_x.predict(token_input).argmax(axis=-1)
        pic = []
        for i in range(len(predictions)):
            pic.append(classes1[predictions[i]])
        df["predicted_category"] = pic
        #df.to_csv("trans_file.csv")
        response = make_response(df.to_csv())
        response.headers["Content-Disposition"] = "attachment; filename=result.csv"
        response.headers["Content-Type"] = "text/csv"
        server.logger.info("Logged in")
        return response


if __name__ == "__main__":
    server.run(debug=True)
