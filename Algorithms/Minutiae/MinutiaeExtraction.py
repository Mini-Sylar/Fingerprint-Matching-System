from PIL import Image, ImageFilter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import convolve
import math
import skimage
import matplotlib.image as mpimg
from skimage.morphology import square
from skimage.morphology import convex_hull_image, erosion
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import imageio
import PIL
import cv2
import pandas as pd


DATA_DIR = "C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Real\\"
list_dirs = list(glob.glob(DATA_DIR+"*.BMP"))
num_images = len(list_dirs)
random.seed(42)

r = random.randint(0, num_images)
display_list = list_dirs[r:r+3]

image1 = imageio.v3.imread(display_list[0])
image2 = imageio.v3.imread(display_list[1])
image3 = imageio.v3.imread(display_list[2])

fig, axes = plt.subplots(1, 3, figsize=(16, 16))
axes[0].imshow(image1)
axes[1].imshow(image2)
axes[2].imshow(image3)


# Image Transforms
#
# 1. Image Smoothening
# 2. Thresholding
# 3. Edge Detection
#
# Image enhancement and preprocessing techniques such as smoothing, thresholding and edge detection are used to make features more prominent in data for extraction to be more accurate.


gauss_blur = cv2.GaussianBlur(image1, (1, 1), 0)
median_blur = cv2.medianBlur(image1, 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 16))
axes[0].set_title("original Image")
axes[0].imshow(image1)
axes[1].set_title("Gaussian Blurred Image")
axes[1].imshow(gauss_blur)
axes[2].set_title("Median Blurred Image")
axes[2].imshow(median_blur)

# %% [markdown]
# #### Histograms

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(image1.ravel(), bins=256, color="r")
axes[1].hist(image2.ravel(), bins=256)
axes[2].hist(image3.ravel(), bins=256, color="g")

# %% [markdown]
# #### Data seems to be almost binary - implementing mean and adaptive thresholding

# %%
# mean thresholding - gives bad results
THRESHOLD1 = image1.mean()
THRESHOLD2 = image2.mean()
THRESHOLD3 = image3.mean()

image1 = np.array(image1 > THRESHOLD1).astype(int) * 255
image2 = np.array(image2 > THRESHOLD2).astype(int) * 254
image3 = np.array(image3 > THRESHOLD3).astype(int) * 254

fig, axes = plt.subplots(1, 3, figsize=(16, 16))
axes[0].imshow(image1)
axes[1].imshow(image2)
axes[2].imshow(image3)

# %%
# Adaptive thresholding from OpenCV library - better than Mean Thresholding

img1 = cv2.imread(display_list[0], 0)
img2 = cv2.imread(display_list[1], 0)
img3 = cv2.imread(display_list[2], 0)

# Otsu's thresholding
ret1, th1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret2, th2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret3, th3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

fig, axes = plt.subplots(1, 3, figsize=(12, 12))
axes[0].set_title("Otsu's thresholding - Image 1")
axes[0].imshow(th2)
axes[1].set_title("Otsu's thresholding - Image 2")
axes[1].imshow(th2)
axes[2].set_title("Otsu's thresholding - Image 3")
axes[2].imshow(th2)

# ### Edge detection:

# convert to grayscale
img_name = display_list[0]
gray_img_array = np.array(Image.open(img_name).convert('P'))

# Robert, Sobel, Prewitt Filters

vertical_robert_filter = np.array([[1, 0], [0, -1]])
horizontal_robert_filter = np.array([[0, 1], [-1, 0]])

vertical_sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
horizontal_sobel_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

vertical_prewitt_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
horizontal_prewitt_filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

print("vertical robert filter\n", vertical_robert_filter)
print("horizontal robert filter\n", horizontal_robert_filter)
print("vertical sobel filter: \n", vertical_sobel_filter)
print("horizontal sobel filter: \n", horizontal_sobel_filter)

print("vertical prewitt filter: \n", vertical_prewitt_filter)
print("horizontal prewitt filter: \n", horizontal_prewitt_filter)

# implementing:
gray_img = Image.fromarray(gray_img_array)

convolved_img1 = convolve(gray_img, vertical_robert_filter)
convolved_img1 = convolve(convolved_img1, horizontal_robert_filter)

convolved_img2 = convolve(gray_img, vertical_sobel_filter)
convolved_img2 = convolve(convolved_img2, horizontal_sobel_filter)

convolved_img3 = convolve(gray_img, vertical_prewitt_filter)
convolved_img3 = convolve(gray_img, horizontal_prewitt_filter)

fig, axes = plt.subplots(1, 3, figsize=(12, 12))
axes[0].set_title("Robert")
axes[0].imshow(convolved_img1)
axes[1].set_title("Sobel")
axes[1].imshow(convolved_img2)
axes[2].set_title("Prewitt")
axes[2].imshow(convolved_img3)

# ### Ridge Detection

src_path = img_name


def detect_ridges(gray, sigma=0.1):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges


def plot_images(*images):
    images = list(images)
    n = len(images)
    fig, ax = plt.subplots(ncols=n, sharey=True, figsize=(12, 12))
    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    plt.show()


img = cv2.imread(src_path, 0)  # 0 imports a grayscale
if img is None:
    raise(ValueError(f"Image didn\'t load. Check that '{src_path}' exists."))

a, b = detect_ridges(img, sigma=0.15)

plot_images(img, a, b)

# ### Termination and Bifurcation detection and Minutiae Extraction
#
# The given code extracts features like Termination, Bifurcation and Minutiae from finger prints, the output is shown below the code:
#

def getTerminationBifurcation(img, mask):
    img = img == 255
    (rows, cols) = img.shape
    minutiaeTerm = np.zeros(img.shape)
    minutiaeBif = np.zeros(img.shape)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if(img[i][j] == 1):
                block = img[i-1:i+2, j-1:j+2]
                block_val = np.sum(block)
                if(block_val == 2):
                    minutiaeTerm[i, j] = 1
                elif(block_val == 4):
                    minutiaeBif[i, j] = 1

    mask = convex_hull_image(mask > 0)
    mask = erosion(mask, square(5))
    minutiaeTerm = np.uint8(mask)*minutiaeTerm
    return(minutiaeTerm, minutiaeBif)


class MinutiaeFeature(object):
    def __init__(self, locX, locY, Orientation, Type):
        self.locX = locX
        self.locY = locY
        self.Orientation = Orientation
        self.Type = Type


def computeAngle(block, minutiaeType):
    angle = 0
    (blkRows, blkCols) = np.shape(block)
    CenterX, CenterY = (blkRows-1)/2, (blkCols-1)/2
    if(minutiaeType.lower() == 'termination'):
        sumVal = 0
        for i in range(blkRows):
            for j in range(blkCols):
                if((i == 0 or i == blkRows-1 or j == 0 or j == blkCols-1) and block[i][j] != 0):
                    angle = -math.degrees(math.atan2(i-CenterY, j-CenterX))
                    sumVal += 1
                    if(sumVal > 1):
                        angle = float('nan')
        return(angle)
    elif(minutiaeType.lower() == 'bifurcation'):
        (blkRows, blkCols) = np.shape(block)
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        angle = []
        sumVal = 0
        for i in range(blkRows):
            for j in range(blkCols):
                if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                    angle.append(-math.degrees(math.atan2(i -
                                 CenterY, j - CenterX)))
                    sumVal += 1
        if(sumVal != 3):
            angle = float('nan')
        return(angle)


def extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif):
    FeaturesTerm = []

    minutiaeTerm = skimage.measure.label(minutiaeTerm, connectivity=2)
    RP = skimage.measure.regionprops(minutiaeTerm)

    WindowSize = 2
    FeaturesTerm = []
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = skel[row-WindowSize:row+WindowSize +
                     1, col-WindowSize:col+WindowSize+1]
        angle = computeAngle(block, 'Termination')
        FeaturesTerm.append(MinutiaeFeature(row, col, angle, 'Termination'))

    FeaturesBif = []
    minutiaeBif = skimage.measure.label(minutiaeBif, connectivity=2)
    RP = skimage.measure.regionprops(minutiaeBif)
    WindowSize = 1
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = skel[row-WindowSize:row+WindowSize +
                     1, col-WindowSize:col+WindowSize+1]
        angle = computeAngle(block, 'Bifurcation')
        FeaturesBif.append(MinutiaeFeature(row, col, angle, 'Bifurcation'))
    return(FeaturesTerm, FeaturesBif)


def ShowResults(skel, TermLabel, BifLabel):
    minutiaeBif = TermLabel * 0
    minutiaeTerm = BifLabel * 0

    (rows, cols) = skel.shape
    DispImg = np.zeros((rows, cols, 3), np.uint8)
    DispImg[:, :, 0] = skel
    DispImg[:, :, 1] = skel
    DispImg[:, :, 2] = skel

    RP = skimage.measure.regionprops(BifLabel)
    for idx, i in enumerate(RP):
        (row, col) = np.int16(np.round(i['Centroid']))
        minutiaeBif[row, col] = 1
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 1)
        skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))

    RP = skimage.measure.regionprops(TermLabel)
    for idx, i in enumerate(RP):
        (row, col) = np.int16(np.round(i['Centroid']))
        minutiaeTerm[row, col] = 1
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 1)
        skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

    plt.figure(figsize=(6, 6))
    plt.title("Minutiae extraction results")
    plt.imshow(DispImg)


img_name = display_list[1]
img = cv2.imread(img_name, 0)
img = np.array(img > THRESHOLD1).astype(int)
skel = skimage.morphology.skeletonize(img)
skel = np.uint8(skel)*255
mask = img*255

(minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask)
FeaturesTerm, FeaturesBif = extractMinutiaeFeatures(
    skel, minutiaeTerm, minutiaeBif)
BifLabel = skimage.measure.label(minutiaeBif, connectivity=1)
TermLabel = skimage.measure.label(minutiaeTerm, connectivity=1)
ShowResults(skel, TermLabel, BifLabel)
