from sys import platform as _platform
import json
from flask import Flask, jsonify, render_template, request

from src.modHero import ModHero

# modelS path
MODELS_PATH = './src/models/'

# input preprocessing
MAX_FEAT = 5000

LABELS = ["toxic", "severe_toxic", "obscene",
          "threat", "insult", "identity_hate"]

# load modHero class (model + self contained helper functions)
modHero = ModHero(MODELS_PATH, MAX_FEAT, LABELS)

# HTTP API
app = Flask(__name__)


def run_inference_on_text(text):
    results = modHero.classify(text)

    return results


@app.route('/v1/api/classify', methods=['POST'])
def classifyText():
    results = run_inference_on_text(request.form['text'])

    return json.dumps({"results": results})


if __name__ == '__main__':
    app.run()
