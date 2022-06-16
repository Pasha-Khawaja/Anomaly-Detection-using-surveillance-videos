import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


test_images = np.load('/home/pasha/Desktop/pasha/total_testing_images.npy', mmap_mode = 'r')
test_labels = np.load('/home/pasha/Desktop/pasha/total_testing_labels.npy')

print('files read')

my_model = tf.keras.models.load_model('m1-1.h5')
print('model loaded')

my_predictions = my_model.predict(test_images[0:100,...])

my_predictions = np.argmax(my_predictions, axis=1)

print(confusion_matrix(np.argmax(test_labels[0:100,...],axis=1), my_predictions))