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

def mnfilter(image, kernel):
	'''
	Filters the input image with the input kernel

	Arguments:
	image -- input image
	kernel -- m x n matrix (preferably m = n = odd number)

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
	res = np.zeros((x, y), dtype = 'int')

	# Filter the image
	for i in range(1, x):
		for j in range(1, y):
			res[i][j] = sum((w[a - s][b - t] * padImg[i - s + a][j - t + b]) 
				for s in range(a, -a - 1, -1) for t in range(b, -b - 1, -1))

	return res

def genGaussianKernel():
	None

def LoG():
	# a) Gaussian of the image, parameters - [sigma, w]
	# b) Laplacian of Gaussian
	None

def sharpen():
	None

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
path = "images/img1.jpg"
img = cv.imread(path)

# Kernel used for mnfilter (assumes an m x m matrix where m is odd)
kernel = np.array([[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9]], dtype = 'int')

#print(img)
#print(kernel)
res = mnfilter(img, kernel)
print(res)
