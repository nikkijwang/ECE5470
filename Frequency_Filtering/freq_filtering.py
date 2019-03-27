# ECE 5470 - Digital Image Processing
# Filtering in the Frequency Domain
# March 27, 2019

# Goal: LoG in the Frequency Domain
# 1. Pad Image
# 2. Center Fourier (-1) ^ x + y
# 3. Fourier of centered + padded image
# 4. Generte Gaussian
# 5. Generate Laplacian
# 6. F(u, v) x N(u, v) = L(u, v)
# 7. Inverse Fourier
# 8. Uncenter by (-1) ^ x + y

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def padImage(img):
	'''
	Pad the given image to be twice its size

	Arguments:
	img		-- image to be padded

	Returns:
	padImg	-- padded image
	'''

	# Resize the image to be 255 x 255
	# h, w = 255, 255
	# res = cv.resize(img, (h, w), interpolation = cv.INTER_CUBIC)

	# padImg = np.pad(res, ((0, res.shape[0]), (0, res.shape[1])), 'constant')
	padImg = np.pad(img, ((0, img.shape[0]), (0, img.shape[1])), 'constant')

	return padImg

def centerFourier(img):
	'''
	Center or uncenter the image's Fourier transform

	Arguments:
	img		-- input image

	Returns:
	ctr		-- image with its Fourier transform centered
	'''
	x, y = img.shape[:2]
	ctr = np.zeros((x, y), dtype = "uint8")
	for i in range(x):
		for j in range(y):
			ctr[i, j] = (-1)**(i + j)

	cImg = img * ctr

	return cImg

def findDist(x, y, u, v):
	d = np.sqrt((x - u)**2 + (y - v)**2)
	return d

def genDist(img):
	h, w = img.shape[:2]		# Kernel height and width
	ch, cw = h // 2, w // 2		# Center of kernel
	d = np.zeros((h, w))

	for rows in range(h):
		for cols in range(w):
			d[rows][cols] = findDist(ch, cw, rows, cols)

	return d


def glp(img, sigma = 1):
	'''
	Gaussian Low Pass Filter

	Arguments:
	img		-- image to be filtered
	sigma	-- sigma value (Default: 1)

	Returns:
	h		-- Gaussin filter kernel
	'''

	height, width = img.shape[:2]
	# Center of the kernel
	ch, cw = height // 2, width // 2

	# Generate distance matrix
	d = genDist(img)

	h = np.zeros((height, width))

	# Calculate the Gaussian kernel
	for i in range(height):
		for j in range(width):
			h[i][j] = np.exp(-(d[i][j]**2)/(2 * (sigma**2)))

	return h

def lap(img):
	''' 
	Laplacian Filter

	Arguments:
	img		-- image to be filtered

	Returns:
	l		-- Laplacian filter kernel
	'''
	height, width = img.shape[:2]

	# Generate distance matrix
	d = genDist(img)

	l = np.zeros((height, width))

	for i in range(height):
		for j in range(width):
			l[i][j] = -4 * (np.pi ** 2) * (d[i][j] ** 2)

	return l

def LoG(img, sigma = 1):
	'''
	Laplacian of Gaussian

	Arguments:
	img		-- image to be filtered
	sigma	-- sigma value (Default: 1)

	Returns:
	lg		-- Laplacian of Gaussian filtered image
	'''

	# g = 1 - glp(img, sigma)		# Generate Gaussian kernel
	g = glp(img, sigma)
	l = lap(img)				# Generate Laplacian kernel

	lg = img * g * l			# Filter the image

	return lg

def sharpen(img, c = 1, sigma = 1):
	'''
	Sharpens the given input image by filtering through a Laplacian of Gaussian filter.

	Arguments:
	img		-- input image
	c		-- constant (Default: 1)
	sigma	-- sigma value (Default: 1)

	Returns:
	s		-- sharpened image
	'''
	
	# plt.figure(1)
	# plt.imshow(img, cmap = 'gray')
	# plt.title("Original"), plt.xticks([]), plt.yticks([])

	# 1. Pad the image
	pad = padImage(img)
	# plt.figure(2)
	plt.subplot(231), plt.imshow(pad, cmap = 'gray')
	plt.title("Pad"), plt.xticks([]), plt.yticks([])

	# 2. Center the image's Fourier
	ctred = centerFourier(pad)
	plt.subplot(232), plt.imshow(ctred, cmap = 'gray')
	plt.title("Centered"), plt.xticks([]), plt.yticks([])

	# Normalize the image to be [0, 1]
	ctred = ctred // np.amax(ctred)
	# print("ctred.dtype = {}".format(ctred.dtype))

	# Fourier of the padded and centered image
	dft = np.fft.fft2(ctred)

	# Laplacian of Gaussian filtered image
	log = LoG(dft, sigma)

	# 3. Inverse Fourier
	idft = np.abs(np.fft.ifft2(log))
	plt.subplot(233), plt.imshow(idft, cmap = 'gray')
	plt.title("Inverse"), plt.xticks([]), plt.yticks([])

	# Normalize the image to be [-1, 1]
	nidft = idft // np.amax(idft)

	# 4. Uncenter
	unc = centerFourier(nidft)
	# unc = centerFourier(idft)
	plt.subplot(234), plt.imshow(unc, cmap = 'gray')
	plt.title("Uncentered"), plt.xticks([]), plt.yticks([])

	# 5. Unpad
	unp = unc[:unc.shape[0]//2, :unc.shape[1]//2]
	plt.subplot(235), plt.imshow(unp, cmap = 'gray')
	plt.title("Unpad"), plt.xticks([]), plt.yticks([])

	# 6. Sharpen
	s = img + c*unp
	# cv.imshow("Sharpened", s)
	plt.subplot(236), plt.imshow(s, cmap = 'gray')
	plt.title("Sharpened"), plt.xticks([]), plt.yticks([]), plt.show()

	return s

def selInput():
	'''
	Selectes and image to be filtered, and the sigma value for the Gaussian kernel

	Arguments:
	None

	Returns:
	imgNum	-- selected image
	c		-- constant value
	sigma	-- sigma value (Default: 0.5)
	'''

	# Selects an image
	imgNum = int(input("Pick a number from 1 to 3 to select an image: "))
	while imgNum < 1 or imgNum > 3:
		imgNum = int(input("Invalid entry. Pick a number from 1 to 3 to select an image: "))

	# Selects a constant value, c
	c = float(input("Pick a value for c (Returns: -c): "))
	c = -c

	# Select sigma value for Gaussian kernel
	sigma = 1.0
	print("\nNote: If a choice other than (y/Y) is chosen, program will default to 'no'.")
	choice = input("Would you like to pick a sigma value (default: 1)? (y/n) ")
	if choice.lower() == 'y':
		sigma = float(input("Pick a positive sigma value: "))
		while sigma <= 0:
			sigma = float(input("Invalid entry. Please pick a positive sigma value: "))
	else:
		pass

	return imgNum, c, sigma

if __name__ == '__main__':
	# Select input image
	imgNum, c, sigma = selInput()

	# Load the image
	path = "images/img{}.jpg".format(imgNum)
	img = cv.imread(path, 0)
	
	# Sharpen the image
	res = sharpen(img, c, sigma)

	plt.subplot(121), plt.imshow(img, cmap = 'gray')
	plt.title("Original"), plt.xticks([]), plt.yticks([])

	plt.subplot(122), plt.imshow(res, cmap = 'gray')
	plt.title("Sharpened"), plt.xticks([]), plt.yticks([])
	plt.show()

	cv.waitKey(0)
	cv.destroyAllWindows()