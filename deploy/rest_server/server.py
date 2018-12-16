import json
import os
import sys
from flask import Flask, request
from flask import jsonify

import numpy as np
import cv2

from tqdm import tqdm

import server_utils
from werkzeug.serving import WSGIRequestHandler

from deploy.inference_model import InferenceModel

app = Flask(__name__)
app.config.from_object(__name__)

model = InferenceModel("../models/best_2.h5")


@app.route('/glass_no_glass/', methods=['GET', 'POST'])
def glass_no_glass():
    image = server_utils.get_image_from_request(request)
    model.run(image)
    
    return jsonify(resp)

@app.route('/ping/', methods=['GET'])
def ping():
    return jsonify({"success": "True"})

if __name__ == '__main__':
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host="0.0.0.0", port=8888, debug=False)