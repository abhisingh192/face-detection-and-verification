#Face detection using the MTCNN
#Give it an image containing a face, it will detect the face from the image and save the face part of the image in a new file

import cv2
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN

# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
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
		cv2.imwrite('bounded2.jpg',gray)
		
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		break
	# show the plot
	pyplot.show()
	

filename = r'C:\Users\ABHINEET SINGH\Desktop\face-detect-and-verify\2. face detection\face-detection-mtcnn//Mad.jpg'
print(filename)
# load image from file
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_image_with_boxes(filename, faces)
#cv2.imwrite('bounded.jpg',)

