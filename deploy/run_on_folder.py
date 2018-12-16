#!/usr/bin/python3
import sys
import cv2
from glob import glob
import os
from inference_model import InferenceModel
from tqdm import tqdm


if len(sys.argv) < 2 or "deploy" not in sys.argv[0]:
    raise AttributeError("please, run: python3 deploy/run_on_folder.py path/to/you/folder/")

model = InferenceModel("./models/best_2.h5")

for image_path in tqdm(glob(sys.argv[1] + "/*")):
    image = cv2.imread(image_path)
    pred = model.run(image)
    if pred == 1:
        print(image_path)
    