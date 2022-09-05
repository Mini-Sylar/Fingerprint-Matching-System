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




path_train = 'C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Real\\1__M_Left_index_finger.BMP'
path_test = 'C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Altered\\Easy\\1__M_Left_index_finger_CR.BMP'

# Image loading
img_train = load_image(path_train, True)
img_test = load_image(path_test, True)

# img = histogram_equalisation(img)

img_enhanced_train = enhance_image(img_train)
img_enhanced_test = enhance_image(img_test)

display_image(img_enhanced_train)