# ECE 5470 - Digital image Processing
# Homework #4: Automatic Hole Filling
# April 24, 2019

# 1. Form a marker image F
# 		F(x, y) = 1 - I(x, y) if (x, y) is on the border of I, 0 otherwise
# 2. Form a mask as G = I_compliment(x, y)
# 3. Perform the morphological reconstruction - Geodesic Dilation with mask I_compliment on marker F
# 4. Obtain the complement H = [Step 3]_complement

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def mkMarker(img, h, w):
	'''
	Form a marker image.
	F(x, y) = 1 - I(x, y)	if (x, y) is on the border of I
			= 0			 	otherwise

	Arguments:
	img		-- Input image
	h		-- Height of the image
	w		-- Width of the image

	Returns:
	marker	-- Marker image
	'''
	# Create a numpy array that is size (h - 2, w - 2)
	mk = np.zeros((h - 2, w - 2), dtype = int)

	# Pad the border of the marker with 1's
	marker = np.pad(mk, 1, 'constant', constant_values = 1)

	# Form the marker image
	marker = marker * img

	return marker.astype("uint8")
	

def mkMask(img):
	'''
	Create a mask using compliment of image

	Arguments:
	img		-- Compliment of the input image

	Returns:
	mask	-- Mask image
	'''

	# Complement of the image
	mask = abs(img.astype(int) - 255)

	return mask.astype("uint8")

def gDilation(size, marker, mask):
	'''
	Geodesic Dilation

	Arguments:
	size	-- Structuring ELement size
	marker	-- Marker image
	mask	-- Mask image
	
	Returns:
	rec		-- Morphological reconstructed image
	'''

	# Generate N x N structuring element
	se = cv.getStructuringElement(cv.MORPH_RECT, (size, size))

	# prev = np.zeros((h, w), dtype = int)
	prev = marker

	while True:
		# Dilate marker with structuring element
		rec = cv.dilate(prev, se, iterations = 1)

		# Intersect dilated image with the mask
		rec = np.bitwise_and(rec, mask)

		# Continue dilation until it can no longer be done
		if (rec == prev).all():
			break
		else:
			prev = rec

	return rec.astype("uint8")

def ahFill(img, size):
	'''
	Automatic Hole Filling

	Arguments:
	img		-- Input image
	size	-- Structuring Element size

	Returns:

	'''

	# Threshold the image to ensure the image is binary
	_, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

	h, w = thresh.shape[:2]

	# Form mask
	mask = mkMask(thresh)

	# Form marker
	marker = mkMarker(mask, h, w)

	# Perform morphological reconstruction
	rImg = gDilation(size, marker, mask)

	# Compliment of the morphological reconstruction to obtained the final result
	result = abs(rImg.astype(int) - 255).astype("uint8")

	return result

def getInputs():
	'''
	Gathers the necessary parameters

	Returns:
	imgNum		-- Image number (1 or 2)
	size		-- Structuring Element size
	'''

	# Selects input image
	imgNum = int(input("Select an image (1 or 2): "))
	while imgNum < 1 or imgNum > 2:
		imgNum = int(input("Invalid input. Please select again (1 or 2): "))

	# Size of the structuring element
	size = int(input("Select the size of the N x N Structuring Element (N > 0): "))
	while size < 1:
		size = int(input("Invalid input. Please select an N > 0: "))

	return imgNum, size

if __name__ == "__main__":

	# Obtain the input image and structuring element size
	imgNum, size = getInputs()

	# Load the selected image
	path = "images/img{}.jpg".format(imgNum)
	img = cv.imread(path, 0)

	# Perform automatic hole filling
	fill = ahFill(img, size)

	# Display results
	plt.subplot(121), plt.imshow(img, cmap = 'gray')
	plt.title("Original"), plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(fill, cmap = 'gray')
	plt.title("Result"), plt.xticks([]), plt.yticks([]), plt.show()

	cv.waitKey(0)
	cv.destroyAllWindows()