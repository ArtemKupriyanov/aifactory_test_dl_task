from glob import glob
from skimage import io
import cv2
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm_notebook
import dlib

import matplotlib.pyplot as plt

class InferenceModel:
    def __init__(self, model_path="./models/best_2.h5"):
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.dnn_face_detector = dlib.cnn_face_detection_model_v1("./models/mmod_human_face_detector.dat")

        self.model = load_model(model_path)
        self.model._make_predict_function()
    
    def run(self, image):
        to_nn_image = self.preprocessing(image)
        predictions = np.argmax(self.model.predict(to_nn_image)[0])
        return predictions
    
    def preprocessing(self, image):
        eye_zone_image = self.get_eyes_zone(image)
        to_nn_image = np.array([self.make_square(eye_zone_image)])
        return to_nn_image
        
    def get_rect(self, image):
        """
        image -- bgr image
        returns: face rect
        """
        rects = self.dlib_detector(image, 0)
        if len(rects) == 1:
            return rects[0]
        else:
            rects = self.dnn_face_detector(image, 0)
            if len(rects) == 1:
                return rects[0].rect
            return []
        
    def get_eyes_zone(self, image):
        rect = self.get_rect(image)
        if rect == []:
            return []
        h, w = image.shape[:2]
        top = np.max([0, rect.top()])
        bottom = np.min([h, rect.bottom() - rect.height() // 2])
        left = np.max([0, rect.left()])
        right = np.min([w, rect.right()])
        return image[top:bottom, left:right]
    
    def make_square(self, image, size=(82, 82)):
        w, h = image.shape[:2]
        if w > h:
            top, bottom = 0, 0
            left = (w - h) // 2
            right = (w - h) - left
        else:
            left, right = 0, 0
            top = (h - w) // 2
            bottom = (h - w) - top
        pad_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return cv2.resize(pad_image, size)