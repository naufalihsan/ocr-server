from flask import Flask, request, jsonify, Response
from core.ocr import read_card
import numpy as np


app = Flask(__name__)

app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]


@app.route("/", methods=["POST"])
def main():
    img_file = request.files['file'].read()
    encoded = np.fromstring(img_file, np.uint8)
    resp = Response(response=read_card(encoded),
                    status=200,
                    mimetype="application/json")
    return resp
