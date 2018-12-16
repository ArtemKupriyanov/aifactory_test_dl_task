import json
import os
import sys
from flask import Flask, request
from flask import jsonify

import dlib
import numpy as np
import cv2
import numpy as np
import cv2

from tqdm import tqdm

from werkzeug.serving import WSGIRequestHandler

from inference_model import InferenceModel

app = Flask(__name__)
app.config.from_object(__name__)

model = InferenceModel("./models/best_2.h5")

def get_image_from_request(request):
    filestr = request.files['image'].read()
    nparr = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/glass_no_glass/', methods=['GET', 'POST'])
def glass_no_glass():
    image = get_image_from_request(request)
    preds = model.run(image)
    preds_str = "glasses" if preds else "no_glasses"
    return jsonify({"answer": preds_str})

@app.route('/ping/', methods=['GET'])
def ping():
    return jsonify({"success": "True"})

if __name__ == '__main__':
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host="0.0.0.0", port=8888, debug=False)