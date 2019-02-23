# ECE 5470 - Digital Image Processing
# Spatial Filtering

import numpy as np
import cv2 as cv

def flipKernel(kernel):
	'''
	Flips the 2D kernel

	Arguments:
	kernel -- kernel to be flipped

	Returns:
	res -- flipped kernel
	'''

	res = np.flip(kernel, 1)	# Flip across vertical axis
	res = np.flip(res, 0)		# Flip across horizontal axis

	return res

def scaleValues(img):
	'''
	Scales the values of the filtered image

	Arguments:
	img -- image to be scaled

	Returns:
	scaled -- scaled image
	'''

	# Find the minimum and maximum values of the image
	imgMax = img.max()
	imgMin = img.min()

	# Calculate the slope
	slope = 255 / (imgMax - imgMin)

	scaled = np.zeros((img.shape[0], img.shape[1]), dtype = 'int')

	# Scale the image
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			scaled[i][j] = ((slope*img[i][j]) - (slope*imgMin)).astype('int')

	return scaled

def mnfilter(image, kernel, scale = 1):
	'''
	Filters the input image with the input kernel

	Arguments:
	image -- input image
	kernel -- m x n matrix (preferably m = n = odd number)
	scale -- set to 0 if scaling is not desired, 1 (default) if scaling is desired

	Returns:
	res -- filtered image
	'''

	# Dimensions of the kernel
	m = kernel.shape[0]		# rows
	n = kernel.shape[1]		# cols

	# Dimensions of the image
	x = image.shape[0]
	y = image.shape[1]

	# Calculate a and b
	a = int((m - 1) / 2)
	b = int((n - 1) / 2)

	# Flip kernel
	w = flipKernel(kernel)
	# print(w)

	# Pad image using replicate padding
	padImg = np.pad(image, (a, b), 'edge')

	# Resulting filtered image of the same size as the input
	res = np.zeros((x, y))

	# Filter the image
	for i in range(1, x):
		for j in range(1, y):
			res[i][j] = sum((w[a - s][b - t] * padImg[i - s + a][j - t + b]) for s in range(a, -a - 1, -1) for t in range(b, -b - 1, -1))

	# Scale if desired and needed
	if scale:
		if res.max() > 255 or res.min() < 0:
			res = scaleValues(res)

	return res.astype('uint8')

def genGaussianKernel(m, sigma = 1, k = 1):
	'''
	Generates an m x m Gaussian kernel with the chosen sigma

	Arguments:
	m -- size of the kernel, where m is an odd number
	sigma -- sigma value; default sigma = 1
	k -- constant; default k = 1

	Returns:
	kernel -- m x m Gaussian kernel
	'''

	# Calculate a and b (they're the same in this case)
	ab = int((m - 1) / 2)

	kernel = np.zeros((m, m))

	# Calculate kernel values
	for s in range(-ab, ab + 1):
		for t in range(-ab, ab + 1):
			e = -(s**2 + t**2) / (2 * (sigma ** 2))
			kernel[s + ab][t + ab] = k * np.exp(e)

	kernel = kernel / np.sum(kernel)

	return kernel

def LoG():
	# a) Gaussian of the image, parameters - [sigma, w]
	# b) Laplacian of Gaussian
	None

def sharpen():
	None

# For testing purposes
def genImpulseMatrix(m):
	mat = np.zeros((m, m), dtype = 'int')
	mat[int(m/2)][int(m/2)] = 1

	return mat

def genKernel(m):
	mat = np.full((m, m), 1, dtype = 'int')
	x = 1
	for i in range(m):
		for j in range(m):
			mat[i][j] *= x
			x += 1

	return mat

# Input image
path = "images/img2.jpg"
img = cv.imread(path, 0)

# # Kernel used for mnfilter (assumes an m x m matrix where m is odd)
# kernel = np.array([[1, 2, 3],
# 				[4, 5, 6],
# 				[7, 8, 9]], dtype = 'int')

#print(img)
#print(kernel)
# res = mnfilter(img, kernel)

# print("img"); print(img)
# print("res"); print(res)

# # Show images
# cv.imshow('Original', img)
# cv.imshow('Filtered', res)

fil = genGaussianKernel(3)
print(fil)	
print(np.sum(fil))


cv.waitKey(0)
cv.destroyAllWindows()
