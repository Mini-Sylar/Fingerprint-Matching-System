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

print(f"base image shape: {img.shape}")

"""
    the first thing to do here will be to equalize the image
    reason for this is stated in the literature review
    basically contrast adjustment 
"""
equalized_image = histogram_equalisation(img)
# Display equalized image vs original image (SHOW IN WINDOW)
display_image(np.hstack((img,equalized_image)),title="Equalized Image Comparison")
