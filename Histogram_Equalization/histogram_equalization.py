# ECE 5470 - Digital Image Processing
# Homework 1: Histogram Equalization

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path = "gray1.jpg"
img = cv.imread(path)
# width - img.shape[1]
# height - img.shape[0]
height, width = img.shape[:2]

# Resize image if too big
if height > 800 or width > 800:
        res = cv.resize(img, (int(width/3), int(height/3)), interpolation = cv.INTER_CUBIC)
else:
        res = img

def adjust_gamma(image, gamma = 1.0):
	invGamma = 1.0 / gamma

	# Build lookup table mapping pixel values to adjusted gamma values
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# Apply gamma correction using the lookup table
	return cv.LUT(image, table)

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

	# Plot original and equalized image
	comp = np.hstack((img, equ))
	cv.imshow("Comparison", comp)

	if len is 1:
		# Plot histograms of original and equalized
		plt.hist(img.flatten(), 256, [0, 256], color = 'r')
		plt.hist(equ.flatten(), 256, [0, 256], color = 'b')
		plt.legend(("Original", "Equalized"), loc = 'upper left')
		plt.show()
	else:
		return equ

def apply_diff_gammas(img):
	images = []
	images.append(img)

	gamma = 0.5
	for i in range(3):
		images.append(adjust_gamma(img, gamma))
		gamma += 0.5

	r1 = np.hstack((images[0], images[1]))
	r2 = np.hstack((images[2], images[3]))
	v = np.vstack((r1, r2))
	cv.imshow("Gamma Transformations", v)

	return images

# cv.imshow("Original Image", res)
# adj1 = adjust_gamma(res, gamma = 0.5)
# adj2 = adjust_gamma(res, gamma = 0.1)
# adj3 = adjust_gamma(res, gamma = 1.5)
# cv.imshow("g = 0.5", adj1)
# cv.imshow("g = 0.1", adj2)
# cv.imshow("g = 1.5", adj3)

# r1 = np.hstack((res, adj1))
# r2 = np.hstack((adj2, adj3))
# allPics = np.vstack((r1, r2))
# cv.imshow("All Pics", allPics)

# histEqual(res)

cv.waitKey(0)
cv.destroyAllWindows()
