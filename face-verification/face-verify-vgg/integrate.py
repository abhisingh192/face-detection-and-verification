# Code for face verification using the VGG architecture.
# Provide two images to the verify face function and it will tell whether the two images are same or not


import cv2
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN


from keras.models import Sequential,model_from_json
from keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D,Dropout,Flatten,Activation
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np


import sys
import os


from hand import upload

args = sys.argv[1:]


#############################################################################

# draw an image with detected objects
def draw_image_with_boxes0(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		
		cropped_img = data[y:y+height, x:x+height]
		gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
		cv2.imwrite('cropped0.jpg',gray)
		
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		break
	# show the plot
	pyplot.show()


def draw_image_with_boxes1(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		
		cropped_img = data[y:y+height, x:x+height]
		gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
		cv2.imwrite('cropped1.jpg',gray)
		
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		break
	# show the plot
	pyplot.show()
	

filename0 = args[0]
filename1 = args[1]

# load image from file
pixels0 = pyplot.imread(filename0)
# create the detector, using default weights
detector0 = MTCNN()
# detect faces in the image
faces0 = detector0.detect_faces(pixels0)
# display faces on the original image
draw_image_with_boxes0(filename0, faces0)
#cv2.imwrite('bounded.jpg',)


# load image from file
pixels1 = pyplot.imread(filename1)
# create the detector, using default weights
detector1 = MTCNN()
# detect faces in the image
faces1 = detector1.detect_faces(pixels1)
# display faces on the original image
draw_image_with_boxes1(filename1, faces1)
#cv2.imwrite('bounded.jpg',)


#############################################################################


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Finding the cosine distance to compare it to the threshold
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# Eucildean distance can also be used instead of cosine distance, threshold would change accordingly 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def fun(img1, img2):
    img1_representation = model.predict(preprocess_image(img1))[0,:]
    img2_representation = model.predict(preprocess_image(img2))[0,:]
    error = 0
    for i in range(img1_representation.shape[0]):
        error += (img1_representation[i] - img2_representation[i])**2
    print(error)
    return error


# defining the model
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.load_weights('vgg_face_weights.h5')

model.pop()


epsilon = 0.45 #threshold, if less than threshold images are the same

#epsilon = 120 #euclidean distance
 
def verifyFace(img1, img2):
    print("veifying the images", img1, img2)
    img1_representation = model.predict(preprocess_image(img1))[0,:]
    img2_representation = model.predict(preprocess_image(img2))[0,:]
 
    cosine_similarity = findCosineDistance(img1_representation, img2_representation)
    print("the cosine distance between the two images is", cosine_similarity)
    # euclidean_distance = findEuclideanDistance(img1_representation, img2_representation) 
    if(cosine_similarity < epsilon):
        print("Verified! Same person")
    else:
        print("Sorry, not the same person!")

verifyFace('cropped0.jpg','cropped1.jpg')

