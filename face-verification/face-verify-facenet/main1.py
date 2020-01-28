# Face verification using the facenet model. 
import numpy as np

from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt
from keras.preprocessing import image


from inception_resnet_v1 import *
model = InceptionResNetV1()   # this is the model


model.load_weights('facenet_weights.h5')   #loading the pretrained model



def preprocess_image(image_path):
    img = load_img(image_path, target_size=(160, 160))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def l2_normalize(x):
 return x / np.sqrt(np.sum(np.multiply(x, x)))


def findEuclideanDistance(source_representation, test_representation):
 euclidean_distance = source_representation - test_representation
 euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
 euclidean_distance = np.sqrt(euclidean_distance)
 return euclidean_distance


img1_representation = l2_normalize(model.predict(preprocess_image('1.png'))[0,:])
img2_representation = l2_normalize(model.predict(preprocess_image('2.png'))[0,:])
 
euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
print(euclidean_distance)


threshold = 0.35 #euclidean distance is compared with the threshold, if it is less than the threshold then the images are same else diferent
if euclidean_distance < threshold:
 print("verified... they are same person")
else:
 print("unverified! they are not same person!")


#below is the code for the cosine distance function, we can use either the euclidean distance or the cosine distance. just the threshold will be different
'''


def findCosineSimilarity(source_representation, test_representation):
 a = np.matmul(np.transpose(source_representation), test_representation)
 b = np.sum(np.multiply(source_representation, source_representation))
 c = np.sum(np.multiply(test_representation, test_representation))
 return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
img1_representation = model.predict(preprocess_image('img1.jpg'))[0,:]
img2_representation = model.predict(preprocess_image('img2.jpg'))[0,:]
 
cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
print("cosine similarity: ",cosine_similarity)
threshold = 0.07
if cosine_similarity < threshold:
 print("verified... they are same person")
else:
 print("unverified! they are not same person!")




'''
 
