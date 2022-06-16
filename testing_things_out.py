import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


my_model = tf.keras.models.load_model('m1-1.h5')
tf.saved_model.save(my_model,'model')
print('Done')