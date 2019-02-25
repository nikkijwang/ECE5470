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

	scaled = np.zeros((img.shape[0], img.shape[1]))

	# Scale the image
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			scaled[i][j] = ((slope*img[i][j]) - (slope*imgMin))

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
	x = image.shape[1]
	y = image.shape[0]

	# Calculate a and b
	a = int((m - 1) / 2)
	b = int((n - 1) / 2)

	# Flip kernel
	w = flipKernel(kernel)

	# Pad image using replicate padding
	padImg = np.pad(image, (a, b), 'edge')

	# Resulting filtered image of the same size as the input
	res = np.zeros((y, x))

	# Filter the image
	for i in range(y):
		for j in range(x):
			res[i][j] = sum((w[a - s][b - t] * padImg[i - s + a][j - t + b]) \
				for s in range(a, -a - 1, -1) for t in range(b, -b - 1, -1))

	# Scale if desired and needed
	if scale == 1:
		if res.max() > 255 or res.min() < 0:
			res = scaleValues(res)

	res = res.astype('uint8')

	return res

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

def gaussian(img, m, sigma = 1):
	'''
	Applies a gussian filter to the input image

	Arguments:
	img -- input image
	m -- kernel size
	sigma -- sigma value (default = 1)

	Returns:
	g -- Gaussian filtered image
	'''

	gk = genGaussianKernel(m, sigma)	# Generate the Gaussian kernel
	g = mnfilter(img, gk)

	return g

def laplacian(img, k = 0):
	'''
	Applies a laplacin filter to the input image

	Arguments:
	img -- input image
	k -- Laplacian filter choice

	Returns:
	l -- Laplacian filtered image
	'''

	if k == 0:
		lk = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])		# isotropic 45
	else:
		lk = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])	# isotropic 90

	l = mnfilter(img, lk, 0)

	return l

def LoG(img, m, sigma = 1, k = 0):
	'''
	Takes the Laplacian of the Gaussian of an image.

	Arguments:
	img -- input image
	m -- size of Gaussian kernel (odd number)
	sigma -- sigma value (default = 1)
	k -- 0: isotropic 45 (default), 1: isotropic 90

	Returns:
	log -- Laplacian of Gaussian
	'''

	# Gaussian of the image
	g = gaussian(img, m, sigma)
	cv.imshow("Gaussian", g)

	# Laplacia of Gaussian
	log = laplacian(g)
	cv.imshow("Laplacian of Gaussian", log)

	return log

def sharpen(img, m, sigma = 1, c = 1, k = 0):
	'''
	Sharpens a given image.

	Arguments:
	img -- input image
	m -- size of Gaussian kernel (odd number)
	sigma -- sigma value (default = 1)
	c -- constant value (default = 1)
	k -- Laplacian kernel (refer to laplacian function)

	Returns:
	sImg -- sharpened image
	'''
	
	log = LoG(img, m, sigma, k)

	sImg = img + (c * log)

	if sImg.max() > 255 or sImg.min() < 0:
		sImg = scaleValues(sImg)

	return sImg.astype("uint8")

def getInput():
	'''
	Obtains required parameters from user and outputs the original, Gaussin filtered, LoG filtered, and sharpened image.
	
	Arguments:
	None

	Returns:
	None
	'''

	# Selects 1 of 5 images
	imgNum = int(input("Pick a number from 1 to 5 to choose an image: "))
	while imgNum < 1 or imgNum > 5:
		imgNum = int(input("Invalid entry. Pick a number from 1 to 5 to choose an image: "))

	# Gaussian kernel size
	ksize = int(input("Pick a odd kernel size (m x m) for the Gaussian filter; m: "))
	while (ksize % 2 - 1) != 0 and ksize > 0:
		ksize = int(input("Invalid entry. Please pick an odd number for the kernel size: "))

	# Sigma value used in generating the Gaussian kernel
	sigma = float(input("Pick a sigma value: "))

	# Use of an isotropic 45 or isotropic 90 3x3 Laplacian kernel
	lk = int(input("Select a Laplacian kernel (0: isotropic 45, 1: isotropic 90): "))
	while lk < 0 or lk > 1:
		lk = int(input("Invalid entry. Select 0 for isotropic 45 or 1 for isotopic 90: "))

	# Contstant value used in sharpening
	c = float(input("Pick a positive value for c [g = f + c(LoG)]: "))
	while c <= 0:
		c = float(input("Invalid entry. Pick a positive c value: "))

	# Load image
	path = "images/img{}.jpg".format(imgNum)
	img = cv.imread(path)
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	# Original image
	cv.imshow("Original", img)

	# Sharpened image
	sharpened = sharpen(img, ksize, sigma, c, lk)
	cv.imshow("Sharpened", sharpened)


# ----------------- For testing purposes --------------------
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
# -----------------------------------------------------------

# Input image
# path = "images/img2.jpg"
# img = cv.imread(path)
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img = genImpulseMatrix(5)
# # Kernel used for mnfilter (assumes an m x m matrix where m is odd)
# kernel = genKernel(3)

# #print(img)
# #print(kernel)
# res = mnfilter(img, kernel)

# print("img"); print(img)
# print("res"); print(res)

# res = sharpen(img, 31, 7, 2)

# # Show images
# cv.imshow('Original', img)
# cv.imshow('Sharpened', res)

getInput()

cv.waitKey(0)
cv.destroyAllWindows()
