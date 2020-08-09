import io
import os
import time
import dash
import nltk
import base64
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objs as go
from nltk.corpus import stopwords
import dash_core_components as dcc
import dash_html_components as html
from healthcheck import HealthCheck
from nltk.tokenize import word_tokenize
from dash.dependencies import Input, Output, State
from flask import Flask, make_response, request, render_template
from transformers import DistilBertTokenizer, DistilBertConfig, TFDistilBertModel


server = Flask(__name__)


app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/dash/'
)

nltk.download('stopwords')


bert_model_name = "uncased"
bert_ckpt_dir = os.path.join("MODEL/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "distilbert-base-uncased-tf_model.h5")
bert_config_file = os.path.join(bert_ckpt_dir, "distilbert-base-uncased-config.json")
tokenizer = DistilBertTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "distilbert-base-uncased-vocab.txt"))


with open('lxkeywords.pkl', 'rb') as handle:
    words = pickle.load(handle)

with open('classes/move_mainc1122.pkl', 'rb') as handle:
    classes_main = pickle.load(handle)

with open('classes/move1.pkl', 'rb') as handle1:
    subclasses1 = pickle.load(handle1)

with open('classes/move2.pkl', 'rb') as handle2:
    subclasses2 = pickle.load(handle2)

with open('classes/move3.pkl', 'rb') as handle3:
    subclasses3 = pickle.load(handle3)


Access_Issue = ["Application Access"]
Email = ["Lotus Notes Issue", "Shared Mailbox", "Outlook"]
VDI = ["Citrix", "VmWare"]
Account_Management = ["Password Reset", "Account Unlock"]
Hardware = ["Workstation", "Printer"]
O365 = ["Skype", "Lync", "One Drive", "Sharepoint"]
ApplicationSoftware = ["Application Installation", "Software Installation", "Application", "ebiz Issue"]
Browser = ["Internet Explorer", "Chrome", "Others Browser"]
Network = ["Wi-Fi", "VPN", "RSA"]
Mobility = ["MDM"]
Office = ["MS Office"]


def create_model_main(max_seq_len, classes):
    config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    tfm = TFDistilBertModel.from_pretrained('./MODEL/uncased/', config=config)
    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    bert_output = tfm(input_ids)[0]

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=512, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=256, activation="tanh")(logits)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=128, activation="tanh")(logits)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    return model


def create_model(max_seq_len, classes):
    config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    tfm = TFDistilBertModel.from_pretrained('./MODEL/uncased/', config=config)
    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    bert_output = tfm(input_ids)[0]

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=512, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=256, activation="tanh")(logits)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    return model


model_main = create_model_main(60, classes_main)
model_main.load_weights("weights/move_weights1122.h5")

models1 = create_model(40, subclasses1)
models1.load_weights("weights/move1.h5")

models2 = create_model(37, subclasses2)
models2.load_weights("weights/move2.h5")

models3 = create_model(42, subclasses3)
models3.load_weights("weights/move3.h5")


logging.basicConfig(filename="flask.log", level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(threadName)s:%(message)s")
logging.info("All useful files loaded")


health = HealthCheck(server, "/hcheck")


def app_ab():
    return True, "i am good"


health.add_check(app_ab)

print(44)


def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return df


def removing_extrawords_stext(text):
    filtered_tokens = word_tokenize(text.lower())
    filtered_tokens_rew = [word for word in filtered_tokens if word in words]
    return " ".join(word for word in filtered_tokens_rew)


def removing_extrawords(df):
    df["des"] = df["Description"]
    for i in range(len(df)):
        filtered_tokens = word_tokenize(df["Description"][i].lower())
        filtered_tokens_rew = [word for word in filtered_tokens if word in words]
        df["des"][i] = " ".join(word for word in filtered_tokens_rew)
        if df["des"][i] == " ":
            df["des"][i] = "others"
    return df


def transform(inp_text, max_len):
    text = inp_text
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens
    if len(tokens) >= max_len:
        tokens = tokens[0:max_len - 2]
    tokens = tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [token_ids + [0] * (max_len - len(token_ids))]
    return np.array(token_ids)


color = {"background": "#111111", "text": "#7FDBFF"}

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

app.layout = html.Div(children=[
    html.H1("Tickets cats visualization", style={"textAlign": "centre", "color": color["text"]}),
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
], style={"backgroundColor": color["background"]})


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
    return render_template("home1.html")


@server.route('/transform', methods=["POST"])
def transform_view():
    if request.method == 'POST':
        start_time = time.time()
        df = pd.read_csv(request.files.get('data_file'))
        #df = df.dropna(subset=["Description"], axis=0)
        df["predicted_category"] = "0"
        df["predicted_subcategory"] = "0"
        df.reset_index(inplace=True, drop=True)
        df = df.dropna()
        df.reset_index(inplace=True, drop=True)
        df_filtered = removing_extrawords(df)
        #df_filtered = df
        print(len(df_filtered))
        df_filtered = df_filtered.dropna()
        print(len(df_filtered))
        df_filtered.reset_index(inplace=True, drop=True)
        for rec in range(len(df_filtered)):
            a = transform(df_filtered["des"][rec], 42)
            cat = model_main.predict(a).argmax(axis=-1)
            #list_cat.append(classes_main[cat[0]])
            df_filtered["predicted_category"][rec] = classes_main[cat[0]]
            if classes_main[cat[0]] in ["Access Issue", "Email", "VDI", "Browser"]:
                a1 = transform(df_filtered["des"][rec], 40)
                sub_cats_all = models1.predict(a1)#.argmax(axis=-1)
                sub_cat = sub_cats_all.argmax(axis=-1)
                sub_category = subclasses1[sub_cat[0]]
                ll = []
                for i in sub_cats_all:
                    ll = i
                if classes_main[cat[0]] == "Access Issue":
                    if sub_category in Access_Issue:
                        #list_subcat.append(sub_category)
                        df_filtered["predicted_subcategory"][rec] = sub_category
                    else:
                        ll[sub_cat[0]] = 0
                        for j in range(7):
                            al = ll.argmax(axis=-1)
                            if subclasses1[al] in Access_Issue:
                                #list_subcat.append(subclasses1[al])
                                df_filtered["predicted_subcategory"][rec] = subclasses1[al]
                                break
                            else:
                                ll[al] = 0
                elif classes_main[cat[0]] == "Browser":
                    if sub_category in Access_Issue:
                        #list_subcat.append(sub_category)
                        df_filtered["predicted_subcategory"][rec] = sub_category
                    else:
                        ll[sub_cat[0]] = 0
                        for j in range(7):
                            al = ll.argmax(axis=-1)
                            if subclasses1[al] in Browser:
                                #list_subcat.append(subclasses1[al])
                                df_filtered["predicted_subcategory"][rec] = subclasses1[al]
                                break
                            else:
                                ll[al] = 0
                        #ll.clear()
                elif classes_main[cat[0]] == "Email":
                    if sub_category in Email:
                        #list_subcat.append(sub_category)
                        df_filtered["predicted_subcategory"][rec] = sub_category
                    else:
                        ll[sub_cat[0]] = 0
                        for j in range(7):
                            al = ll.argmax(axis=-1)
                            if subclasses1[al] in Email:
                                #list_subcat.append(subclasses1[al])
                                df_filtered["predicted_subcategory"][rec] = subclasses1[al]
                                break
                            else:
                                ll[al] = 0
                        #ll.clear()
                else:
                    if sub_category in VDI:
                        #list_subcat.append(sub_category)
                        df_filtered["predicted_subcategory"][rec] = sub_category
                    else:
                        ll[sub_cat[0]] = 0
                        for j in range(7):
                            al = ll.argmax(axis=-1)
                            if subclasses1[al] in VDI:
                                #list_subcat.append(subclasses1[al])
                                df_filtered["predicted_subcategory"][rec] = subclasses1[al]
                                break
                            else:
                                ll[al] = 0
                        #ll.clear()
            elif classes_main[cat[0]] in ["Account Management", "O365", "Hardware"]:
                a2 = transform(df_filtered["des"][rec], 37)
                sub_cats_all = models2.predict(a2)#.argmax(axis=-1)
                sub_cat = sub_cats_all.argmax(axis=-1)
                sub_category = subclasses2[sub_cat[0]]
                mm = []
                for i in sub_cats_all:
                    mm = i
                if classes_main[cat[0]] == "Account Management":
                    if sub_category in Account_Management:
                        #list_subcat.append(sub_category)
                        df_filtered["predicted_subcategory"][rec] = sub_category
                    else:
                        mm[sub_cat[0]] = 0
                        for j in range(12):
                            bl = mm.argmax(axis=-1)
                            if subclasses2[bl] in Account_Management:
                                #list_subcat.append(subclasses2[bl])
                                df_filtered["predicted_subcategory"][rec] = subclasses2[bl]
                                break
                            else:
                                mm[bl] = 0
                        #mm.clear()
                elif classes_main[cat[0]] == "O365":
                    if sub_category in O365:
                        #list_subcat.append(sub_category)
                        df_filtered["predicted_subcategory"][rec] = sub_category
                    else:
                        mm[sub_cat[0]] = 0
                        for j in range(12):
                            bl = mm.argmax(axis=-1)
                            if subclasses2[bl] in O365:
                                #list_subcat.append(subclasses2[bl])
                                df_filtered["predicted_subcategory"][rec] = subclasses2[bl]
                                break
                            else:
                                mm[bl] = 0
                        #mm.clear()
                else:
                    if sub_category in Hardware:
                        #list_subcat.append(sub_category)
                        df_filtered["predicted_subcategory"][rec] = sub_category
                    else:
                        mm[sub_cat[0]] = 0
                        for j in range(12):
                            bl = mm.argmax(axis=-1)
                            if subclasses2[bl] in Hardware:
                                #list_subcat.append(subclasses2[bl])
                                df_filtered["predicted_subcategory"][rec] = subclasses2[bl]
                                break
                            else:
                                mm[bl] = 0
                        #mm.clear()
            elif classes_main[cat[0]] in ["ApplicationSoftware", "Network", "Mobility", "Office"]:
                a3 = transform(df_filtered["des"][rec], 42)
                sub_cats_all = models3.predict(a3)#.argmax(axis=-1)
                sub_cat = sub_cats_all.argmax(axis=-1)
                sub_category = subclasses3[sub_cat[0]]
                nn = []
                for i in sub_cats_all:
                    nn = i
                if classes_main[cat[0]] == "ApplicationSoftware":
                    if sub_category in ApplicationSoftware:
                        #list_subcat.append(sub_category)
                        df_filtered["predicted_subcategory"][rec] = sub_category
                    else:
                        nn[sub_cat[0]] = 0
                        for j in range(11):
                            cl = nn.argmax(axis=-1)
                            if subclasses3[cl] in ApplicationSoftware:
                                #list_subcat.append(subclasses3[cl])
                                df_filtered["predicted_subcategory"][rec] = subclasses3[cl]
                                break
                            else:
                                nn[cl] = 0
                        #nn.clear()
                elif classes_main[cat[0]] == "Network":
                    if sub_category in Network:
                        #list_subcat.append(sub_category)
                        df_filtered["predicted_subcategory"][rec] = sub_category
                    else:
                        nn[sub_cat[0]] = 0
                        for j in range(11):
                            cl = nn.argmax(axis=-1)
                            if subclasses3[cl] in Network:
                                #list_subcat.append(subclasses3[cl])
                                df_filtered["predicted_subcategory"][rec] = subclasses3[cl]
                                break
                            else:
                                nn[cl] = 0
                        #nn.clear()
                elif classes_main[cat[0]] == "Mobility":
                    if sub_category in Mobility:
                        #list_subcat.append(sub_category)
                        df_filtered["predicted_subcategory"][rec] = sub_category
                    else:
                        nn[sub_cat[0]] = 0
                        for j in range(11):
                            cl = nn.argmax(axis=-1)
                            if subclasses3[cl] in Mobility:
                                #list_subcat.append(subclasses3[cl])
                                df_filtered["predicted_subcategory"][rec] = subclasses3[cl]
                                break
                            else:
                                nn[cl] = 0
                        #nn.clear()
                else:
                    if sub_category in Office:
                        #list_subcat.append(sub_category)
                        df_filtered["predicted_subcategory"][rec] = sub_category
                    else:
                        nn[sub_cat[0]] = 0
                        for j in range(11):
                            cl = nn.argmax(axis=-1)
                            if subclasses3[cl] in Office:
                                #list_subcat.append(subclasses3[cl])
                                df_filtered["predicted_subcategory"][rec] = subclasses3[cl]
                                break
                            else:
                                nn[cl] = 0
                        #nn.clear()
            else:
                #list_subcat.append("Left")
                df_filtered["predicted_subcategory"][rec] = "left"
        print("--- %s seconds ---" % (time.time() - start_time))
        #df_filtered.drop("des", axis=1, inplace=True)
        response = make_response(df_filtered.to_csv())
        response.headers["Content-Disposition"] = "attachment; filename=result.csv"
        response.headers["Content-Type"] = "text/csv"
        server.logger.info("Logged in")
        return response


if __name__ == "__main__":
    server.run(debug=True, host="0.0.0.0", port=4500)
