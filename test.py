import cv2
import os
import numpy as np
import tensorflow as tf
import time

cap = cv2.VideoCapture('Abuse023_x264.mp4')
#framespersecond = int(cap.get(cv2.CAP_PROP_FPS))
#print('frames are: ', framespersecond)
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
x = [i for i in range (1, totalFrames) if divmod(i, int(30))[1]==0]
my_model = tf.keras.models.load_model('m1-1.h5')
prediction = 0

pred_list = []
# print(myFrameNumber)
for myFrameNumber in x:
            #set which frame to read
            cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)
            # read frame
            ret, frame = cap.read()
            
            
            resized_image=cv2.resize(frame/255.0, (224, 224))

            
            prediction = my_model.predict(np.expand_dims(resized_image, axis=0))

            prediction = np.argmax(prediction, axis=1)
            # display frame
            
            print('Prediction for frame: ', myFrameNumber, ' is ', prediction)
            
            cv2.imwrite(r'/home/pasha/Desktop/pasha/Anomaly_Frames/Anomaly_Frame_'+str(myFrameNumber)+'.jpg', frame)

            pred_list.append(prediction)
            
for x in range(len(pred_list)):
    print (pred_list[x])


#print(len(actual_list)