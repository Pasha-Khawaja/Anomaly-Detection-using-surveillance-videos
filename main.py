# import required modules
from flask import Flask, render_template, request
import cv2
import os
import numpy as np
import tensorflow as tf
import time


# create flask app
app = Flask(__name__)

def processing(filepath):
    my_predictions_list = [0]*32
    counter = 0
    my_model = tf.keras.models.load_model('m1-1.h5')
    print('model loaded')


    cap = cv2.VideoCapture(filepath)

    while(cap.isOpened()):
        ret, frame = cap.read()

        resized_image=cv2.resize(frame/255.0, (224, 224))

        
        my_predictions = my_model.predict(np.expand_dims(resized_image, axis=0))

        my_predictions = np.argmax(my_predictions, axis=1) #0/1 0 = normal, 1=abnormal

        
        my_predictions_list[counter%32] = my_predictions[0]
        #if my_predictions[0] 
        
        print('Frame prediction is: ', my_predictions[0])
        print('Counter is: ', counter)

        counter += 1

        if counter == 32:
            counter = 0

        final_prediction = np.mean(my_predictions_list)
        print(final_prediction)
        if final_prediction > 0.75:
            print('Anomaly Occuring')

    
# main route (show html page)
@app.route('/')
def index():
    return render_template('index.html')

# api endpoint for image upload
@app.route('/api/upload', methods=['POST'])
def upload():
    # receive the file from the client
    file = request.files['file']
    filepath = f'static/temp/{file.filename}'
    file.save(filepath) # save to directory

    processing(filepath)

    # return server url to client
    return f"{request.url_root}{filepath}"


# Run flask server
if __name__ == '__main__':
    app.run(debug=True) #