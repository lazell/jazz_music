

from flask import Flask, render_template, requests
import numpy as np
import keras.models
import sys
import os


'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Web-App for Fetches Convolutional Neural Net model (for Lindy, Balboa, Shag, Charleston)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
from CNN_Models import *


app = Flask(__name__)

global model, graph
model, graph = init()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    youtubelink = requests.get_data()
    x = to_array(youtubelink)
    # look-up download you tube link

    with graph.as_default():
        out = model.predict(x)
        response = np.array_str(np.argmax(out))
        return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
