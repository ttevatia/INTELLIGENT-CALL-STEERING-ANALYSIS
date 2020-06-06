from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow import keras
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras as ks
from keras import backend
from keras import backend as K
import os
import importlib
importlib.reload(K)

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend("theano")

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'models/mymodel.h5'

# Load your trained model
#model = load_model(MODEL_PATH)
model = tf.keras.models.load_model('models/mymodel.h5')
graph = tf.get_default_graph()
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
model.save('models/mymodel.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='caffe')

#     preds = model.predict(x)
#     return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    from tensorflow import keras


    dataset = keras.datasets.imdb
    dataset.load_data(num_words=10000)
    dicti = dataset.get_word_index()
    #print(dicti)

    dicti = {k:(v+3) for (k,v) in dicti.items()}
    dicti["<PAD>"]=0
    dicti["<START>"]=1
    dicti["<UNK>"]=2
    dicti["<UNUSED"]=3
    t = request.form['text']
    s=str(t)
    text=s.split()
    #predictio=1
    #decoded_text= [dicti.get(word) for word in text]
    decoded_text="positive"
    decoded_text1="negative"
    model._make_predict_function()  
    global graph
    #with graph.as_default():
    #prediction1 = model.predict([[[4,6,9]]])
  
    # if prediction[[0]]>0.5:
    #     output="poasitive"
    # else:
    #     output="negative,mind your language"
    return render_template('base.html', prediction_text='The input text/speech is {}'.format(decoded_text))


if __name__ == '__main__':
    app.run(debug=True)

