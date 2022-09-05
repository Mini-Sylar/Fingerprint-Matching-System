import os
import cv2
import sys
import math
import warnings
import numpy as np
from zipfile import ZipFile

import pandas as pd

#import gdal
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


from scipy import ndimage as ndi
from skimage import feature

from PIL import Image
from IPython.display import display

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Load helpers from Libs folder
sys.path.append("Algorithms/Minutiae/Libs")

from Libs.enhancing import *
from Libs.basics import *
from Libs.processing import *


# Process images here
img  = cv2.imread("C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching "
                        "Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Real\\1__M_Left_index_finger.BMP",0)

print(img)

print(f"base image shape: {img.shape}")

"""
    the first thing to do here will be to equalize the image
    reason for this is stated in the literature review
    basically contrast adjustment 
"""
equalized_image = histogram_equalisation(img)
# Display equalized image vs original image (SHOW IN WINDOW)
display_image(np.hstack((img,equalized_image)),title="Equalized Image Comparison")

# # Visualize Histogram Equalization Graph Here
# color_range = 255
#
# plt.plot(cdf_normalised(img, color_range), color = 'red')
# plt.hist(img.flatten(), color_range, [0, color_range], color = 'teal')
# plt.xlim([0, color_range])
# plt.legend(('CDF','Hist'))
# plt.title('Unprocessed image')
# plt.show()
#
# plt.plot(cdf_normalised(equalized_image, color_range), color = 'red')
# plt.hist(equalized_image.flatten(), color_range, [0, color_range], color = 'teal')
# plt.xlim([0, color_range])
# plt.legend(('CDF','Hist'))
# plt.title('Equalised image')
# plt.show()

# Visualize Magnitude Spectrum Here
# magnitude_spectrum = 20 * np.log(np.abs(fourier_transform(img)))
# img_hpf = high_pass_filter(img)
#
# display_image(img, 'Input Image')
# display_image(magnitude_spectrum, 'Magnitude Spectrum - DFT')
# display_image(img_hpf, 'Image after HPF')


#####
#  Feature Extraction Done Here
####

# After histogram normalization/equalization, we binarize the image
# one way of binarizing is by using the OTSU's thresholding

binarized_image = binarise(img)
img_blur = cv2.GaussianBlur(img, (5,5), 0)
threshold, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

display_image(np.hstack((img_blur, img_otsu, binarized_image)),title="Binarized Image")
print(f'Threshold OTSU: {threshold}')


"""
 The next thing to do after binarization is to detect the edges of the images
 this is done to isolate the background from the actual image
"""

# Edge detection
img_laplace = cv2.Laplacian(img, cv2.CV_64F)
img_blur = cv2.GaussianBlur(img_laplace, (5, 5), 0)
ret, img_bin = cv2.threshold(img_blur, 0, 255, 0)
img_bin_custom = binarise(img_blur)

# display_image(np.hstack((img_laplace, img_blur)))
# display_image(np.hstack((img_bin, img_bin_custom)))

# show image singled out
_, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bin_custom = binarise(img_laplace)

display_image(img_laplace)
display_image(img_bin_custom)

# From here the image is skeletized to grab the edges
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

ret, img_base = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False

while not done:
    eroded = cv2.erode(img_base, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(img_base, temp)
    skel = cv2.bitwise_or(skel, temp)
    img_base = eroded.copy()

    zeros = size - cv2.countNonZero(img_base)
    if zeros == size:
        done = True

# Smoothing required after binarisation.
display_image(np.hstack((img, img_bin, skel)))

# Then from here use the gabor filter to smoothen the edges
def build_filters():
    filters = []
    ksize = 5
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), theta=theta, sigma=2.0,
                                  lambd=15.0, gamma=0.25, psi=0, ktype=cv2.CV_64F)
        # ksize – Size of the filter returned.
        # sigma – Standard deviation of the gaussian envelope.
        # theta – Orientation of the normal to the parallel stripes of a Gabor function.
        # lambd – Wavelength of the sinusoidal factor.
        # gamma – Spatial aspect ratio.
        # psi – Phase offset.
        # ktype – Type of filter coefficients. It can be CV_32F or CV_64F .
        kern /= 1.1 * kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)

    return accum

img_gb = process(binarise(img), build_filters())
# threshold, img_gb = cv.threshold(img_gb, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

display_image(img_gb)


# Image enhancement SIDE