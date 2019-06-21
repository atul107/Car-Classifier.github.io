# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:12:25 2019

@author: atk
"""

from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from utils import load_model
# Define a flask app
app = Flask(__name__)

import cv2 as cv
import keras.backend as K
import scipy.io

# Load your trained model
model = load_model()
model.load_weights('models/model.96-0.89.hdf5')

print('Model loaded. Start serving...')

cars_meta = scipy.io.loadmat('devkit/cars_meta')
class_names = cars_meta['class_names']  # shape=(1, 196)
class_names = np.transpose(class_names)

print('Model loaded. Check http://127.0.0.1:5000/')

#==============================================================================

def model_predict(img_path, model):
    img_width, img_height = 224, 224
    
    bgr_img =  cv.imread(img_path)
    bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    rgb_img = np.expand_dims(rgb_img, 0)
    preds = model.predict(rgb_img)
    
    K.clear_session()

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        prob = np.max(preds)
        class_id = np.argmax(preds)
        print(class_id)
        label =  str(class_names[class_id][0][0])
        prob = str(prob)
        result = 'Model : ' + label + ' (' + 'prob: ' + prob + ')'
        
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
