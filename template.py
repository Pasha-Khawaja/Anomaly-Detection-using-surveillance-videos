import cv2
import os
import numpy as np
import tensorflow as tf
import time

my_predictions_list = [0]*32
count = 0
my_model = tf.keras.models.load_model('m1-1.h5')
print('model loaded')
my_pred_list = []

cap = cv2.VideoCapture('Explosion001_x264.mp4')

totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Total frames in the video are: ', totalframecount)

my_predictions = 0
Frame_count = 0
#while(cap.isOpened()):
while(Frame_count < totalframecount - 1):
    
    ret, frame = cap.read()

    resized_image=cv2.resize(frame/255.0, (224, 224))

    
    my_predictions = my_model.predict(np.expand_dims(resized_image, axis=0))

    my_predictions = np.argmax(my_predictions, axis=1) #0/1 0 = normal, 1=abnormal

    
    #my_predictions_list[counter%32] = my_predictions[0]
    #if my_predictions[0] 
    
    #print('Frame prediction is: ', my_predictions[0])
    #print('Counter is: ', counter)

    #counter += 1

    #if counter == 32:
        #counter = 0

    #final_prediction = np.mean(my_predictions_list)
    #print(final_prediction)
    #if final_prediction > 0.75:
        #print('Anomaly Occuring')
       # print('Anomaly at frame: ', Frame_count)
    
    Frame_count = Frame_count + 1
    #time.sleep(3)
    
    if my_predictions > 0.75:
        my_pred_list.append(my_predictions)
        count += 1
        cv2.imwrite(r'/home/pasha/Desktop/pasha/Anomaly_Frames/Anomaly_Frame_'+str(Frame_count)+'.jpg', frame)
    

print('Video Processed')
