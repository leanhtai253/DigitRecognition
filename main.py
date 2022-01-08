from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import base64
import re 
from io import BytesIO
import os

app = Flask(__name__)

model = tf.keras.models.load_model('HandwrittenDigitRecognizer.h5')

#GET request
@app.route("/", methods=['GET'])
def drawing():
    return render_template('drawing.html')
# POST request
@app.route('/', methods=['POST'])
def canvas():
    #Receive base64 data from the user form
    canvasdata = request.form['canvasimg']
    # encoded_data = request.form['canvasimg'].split(',')[1]
    # #Decode base64 image to python array
    # nparray = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    image_data = re.sub('^data:image/.+;base64,', '', request.form['canvasimg'])
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    # img = Image.open(nparray)
    img = img.convert('L')
    img = img.resize((28,28))
    img = np.array(img)
    img = img.reshape((1,28,28,1))
    # img /= 255.0
    prediction = model.predict([img])[0]
    res = np.argmax(prediction)
    return render_template('drawing.html', response=str(res), canvasdata=canvasdata, success=True)

port = int(os.environ.get('PORT', 5000))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)