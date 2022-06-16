from flask import Flask, render_template, request
from distutils.log import debug
from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import cv2
import os
import numpy as np
import tensorflow as tf
import time
import glob

app = Flask(__name__)
socketio = SocketIO(app)

# on connection
@socketio.on('connect')
def connect():
    print('connected')


@socketio.on("my event")
def test_message(pred):
    emit("my response", {"data": pred['data']})

# main route (show html page)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    
    files = glob.glob('/home/pasha/Desktop/pasha/Anomaly_Frames/*')
    for f in files:
        os.remove(f)
    # receive the file from the client
    file = request.files['file']
    filepath = f'static/{file.filename}'
    file.save(filepath) # save to directory
    
    my_model = tf.keras.models.load_model('m1-1.h5')
    
    my_pred_list = []
    cap = cv2.VideoCapture(file.filename)
    totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x = [i for i in range (1, totalframecount) if divmod(i, int(30))[1]==0]
    Frame_count = 0
    
    for myFrameNumber in x:
        cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)
        ret, frame = cap.read()
        resized_image=cv2.resize(frame/255.0, (224, 224))
        my_predictions = my_model.predict(np.expand_dims(resized_image, axis=0))
        my_predictions = np.argmax(my_predictions, axis=1) #0/1 0 = normal, 1=abnormal
        #print('Prediction for frame: ', myFrameNumber, ' is ', my_predictions)
        if my_predictions == 1:
            anom_result = 'Alert!!!'
            socketio.emit("my response", {"data": str(anom_result)})
            #socketio.emit("my color", {"data": "red"})
            cv2.imwrite(r'/home/pasha/Desktop/pasha/Anomaly_Frames/Anomaly_Frame_'+str(myFrameNumber)+'.jpg', frame)

        else:
            anom_result = ''
            socketio.emit("my response", {"data": str(anom_result)})
            #socketio.emit("my color", {"data": "green"})
        #test = 'Anomaly'
        #socketio.emit("my response", {"data": str(test)})
        #socketio.emit("my response", {"data": str(anom_result)})
        socketio.emit("frame count", {"data": 'Total frames extracted: ' + str(len(x))})
        socketio.emit("Anomaly Frame Location", {"data": 'Anomaly Frames Are Stored in "/home/pasha/Desktop/pasha/Anomaly_Frames"'})
        #time.sleep(1)
    # return server url to client
    return f"Processing Complete "


if __name__ == '__main__':
   socketio.run(app, debug = True)

    