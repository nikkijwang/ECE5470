# ECE 5470 - Digital Image Processing
# Homework 1: Histogram Equalization

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path = "images/gray1.jpg"
img = cv.imread(path, 0)
# width - img.shape[1]
# height - img.shape[0]
height, width = img.shape[:2]

# Resize image if too big
if height > 800 or width > 800:
        res = cv.resize(img, (int(width/4), int(height/4)), interpolation = cv.INTER_CUBIC)
else:
        res = img

def adjust_gamma(image, gamma = 1.0):
	invGamma = 1.0 / gamma

	# Build lookup table mapping pixel values to adjusted gamma values
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# Apply gamma correction using the lookup table
	return cv.LUT(image, table)

# Perform gamma transformation on given image
def apply_diff_gammas(img):
	images = []
	images.append(img)

	gamma = 0.5
	for i in range(2):
		images.append(adjust_gamma(img, gamma))
		gamma += 1

	# Displays original, g = 0.5, g = 1.5
	#row = np.hstack((images[0], images[1], images[2]))
	#cv.imshow("Gamma Transformations - g = {1, 0.5, 1.5}", row)

	return images

# Histogram Equalization for grayscale image(s)
def histEqual(img, len = 1):
	hist = cv.calcHist([img], [0], None, [256], [0, 256])

	# Find the CDF
	cdf = hist.cumsum()
	cdf_norm = cdf * hist.max() / cdf.max()

	cdf_m = np.ma.masked_equal(cdf, 0)
	cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
	cdf = np.ma.filled(cdf_m, 0).astype('uint8')

	# Apply CDF to image
	equ = cdf[img]

	if len is 1:
		# Plot original and equalized image
		comp = np.hstack((img, equ))
		cv.imshow("Comparison", comp)

		# Plot histograms of original and equalized
		plt.hist(img.flatten(), 256, [0, 256], color = 'r')
		plt.hist(equ.flatten(), 256, [0, 256], color = 'b')
		plt.legend(("Original", "Equalized"), loc = 'upper left')
		plt.show()
	else:
		return equ

# Histogram Equalization of a single grayscale image
histEqual(res)

# Histogram Equalization of a single grayscale image with multiple gammas
# -------------------- Start Here --------------------
# images = apply_diff_gammas(res)
# equ = []

# for i in range(len(images)):
# 	equ.append(histEqual(images[i], len(images)))

# # Original and equalized images
# for i in range(len(images)):
# 	r = np.hstack((images[i], equ[i]))
# 	if i == 0:
# 		cv.imshow("g = 1", r)
# 	elif i == 1:
# 		cv.imshow("g = 0.5", r)
# 	else:
# 		cv.imshow("g = 1.5", r)

# # Histogram Plots
# plt.figure(1),
# plt.hist(images[0].flatten(), 256, [0, 256], color = 'r')
# plt.hist(equ[0].flatten(), 256, [0, 256], color = 'b')
# plt.legend(("Original", "Equalized"), loc = 'best')
# plt.title("g = 1")

# plt.figure(2),
# plt.hist(images[1].flatten(), 256, [0, 256], color = 'r')
# plt.hist(equ[1].flatten(), 256, [0, 256], color = 'b')
# plt.legend(("Original", "Equalized"), loc = 'best')
# plt.title("g = 0.5")

# plt.figure(3)
# plt.hist(images[2].flatten(), 256, [0, 256], color = 'r')
# plt.hist(equ[2].flatten(), 256, [0, 256], color = 'b')
# plt.legend(("Original", "Equalized"), loc = 'best')
# plt.title("g = 1.5")

# plt.show()
# -------------------- End Here --------------------

cv.waitKey(0)
cv.destroyAllWindows()