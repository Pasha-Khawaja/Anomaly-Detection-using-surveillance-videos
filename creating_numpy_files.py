import numpy as np
import os
import random
from tqdm import tqdm
import glob
from sklearn.utils import shuffle
import cv2
from tensorflow.keras.utils import to_categorical

dir_1_path = 'C:/Users/Asad/Desktop/Images/Normal/*.png'
dir_2_path = 'C:/Users/Asad/Desktop/Images/Abnormal/*.png'

images_dir_1 = glob.glob(dir_1_path)
images_dir_2 = glob.glob(dir_2_path)

random.shuffle(images_dir_1)
random.shuffle(images_dir_2)

testing_data_images_dir_1 = images_dir_1[0:2000]
testing_data_images_dir_2 = images_dir_2[0:2000]

training_data_images_dir_1 = images_dir_1[2000:7000]
training_data_images_dir_2 = images_dir_2[2000:7000]

total_training_images = []
total_training_labels = []

total_training_images.extend(training_data_images_dir_1)
total_training_images.extend(training_data_images_dir_2)

total_training_labels.extend([0] * len(training_data_images_dir_1))
total_training_labels.extend([1] * len(training_data_images_dir_2))

total_testing_images = []
total_testing_labels = []

total_testing_images.extend(testing_data_images_dir_1)
total_testing_images.extend(testing_data_images_dir_2)

total_testing_labels.extend([0] * len(testing_data_images_dir_1))
total_testing_labels.extend([1] * len(testing_data_images_dir_2))

total_training_images, total_training_labels = shuffle(total_training_images, total_training_labels)
total_testing_images, total_testing_labels = shuffle(total_testing_images, total_testing_labels)

total_training_images_bucket = np.zeros((len(total_training_images), 224, 224, 3), dtype=np.float32)
total_testing_images_bucket = np.zeros((len(total_testing_images), 224, 224, 3), dtype=np.float32)


for counter, image in tqdm(enumerate(total_training_images)):
    total_training_images_bucket[counter, ...] = cv2.resize(cv2.imread(image, -1)/255.0, (224, 224))

for counter, image in tqdm(enumerate(total_testing_images)):
    total_testing_images_bucket[counter, ...] = cv2.resize(cv2.imread(image, -1)/255.0, (224, 224))

np.save('C:/Users/Asad/Desktop/Images/total_training_images.npy', total_training_images_bucket)
np.save('C:/Users/Asad/Desktop/Images/total_testing_images.npy', total_testing_images_bucket)

total_training_labels = to_categorical(total_training_labels, num_classes=2)
total_testing_labels = to_categorical(total_testing_labels, num_classes=2)

np.save('C:/Users/Asad/Desktop/Images/total_training_labels.npy', total_training_labels)
np.save('C:/Users/Asad/Desktop/Images/total_testing_labels.npy', total_testing_labels)
