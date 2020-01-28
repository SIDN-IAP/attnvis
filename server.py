import argparse
import json
import os

from flask import Flask as Flask, send_from_directory, request, Response, \
    redirect, url_for
#### only needed for cross-origin requests:
# from flask_cors import CORS
from transformers import pipeline

__author__ = 'Hendrik Strobelt'

from api import AttentionGetter

app = Flask(__name__)
#### only needed for cross-origin requests:
# CORS(app)

# load huggingface model
loaded_models = {}


# redirect requests from root to index.html
@app.route('/')
def hello_world():
    return redirect('client/index.html')


# functional backend taking sentences as request and returning
# sentiment direction and score as JSON result
@app.route('/api/attn', methods=['POST'])
def attn():
    sentence = request.json['sentence']
    model_name = request.json.get('model_name', 'gpt2')  # type:str

    # lazy loading
    if model_name not in loaded_models:
        loaded_models[model_name] = AttentionGetter(model_name)

    model = loaded_models[model_name]

    if model_name.startswith('gpt'):
        results = model.gpt_analyze_text(sentence)
    else:
        results = model.bert_analyze_text(sentence)

    # return object with request (sentences) and result (sentiments)
    return json.dumps({
        "request": {"sentence": sentence, 'model_name': model_name},
        "results": results
    })


# just a simple example for GET request
@app.route('/api/data/')
def get_data():
    options = request.args
    name = str(options.get("name", ''))
    y = int(options.get("y", 0))

    res = {
        'name': name,
        'y': [10 * y]
    }
    json_res = json.dumps(res)
    return Response(json_res, mimetype='application/json')


# send everything from client as static content
@app.route('/client/<path:path>')
def send_static(path):
    """ serves all files from ./client/ to ``/client/<path:path>``
    :param path: path from api call
    """
    return send_from_directory('client/', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodebug", default=False)
    parser.add_argument("--port", default="8888")
    args = parser.parse_args()

    print(args)

    app.run(port=int(args.port), debug=not args.nodebug)
