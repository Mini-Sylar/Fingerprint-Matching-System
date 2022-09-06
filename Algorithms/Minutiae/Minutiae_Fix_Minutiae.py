import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Load helpers from Libs folder
sys.path.append("Algorithms/Minutiae/Libs")

from Libs.basics import *
from Libs.minutiae import *
from Libs.edges import *

path_train = 'C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Real\\1__M_Left_index_finger.BMP'
path_test = 'C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Altered\\Easy\\1__M_Left_index_finger_CR.BMP'

# Image loading
img_train = load_image(path_train, True)
img_test = load_image(path_test, True)

# img = histogram_equalisation(img)

img_enhanced_train = enhance_image(img_train)
img_enhanced_test = enhance_image(img_test)

display_image(img_enhanced_train,title="enhanced")


# # Moving on the minutiae extraction part
# minutiae_base = process_minutiae(img_enhanced_train)
# plot_minutiae(img_enhanced_train, minutiae_base, size=8)
#
# minutiae_test = process_minutiae(img_enhanced_test)
# plot_minutiae(img_enhanced_test, minutiae_test, size=8)
#
#
# # Perform Corner Detection Here
# img1 = cv2.imread(path_train, 0)
# img2 = cv2.imread(path_test, 0)
#
# edges1, desc1 = edge_processing(img1, threshold=155)
# edges2, desc2 = edge_processing(img2, threshold=155)
#
# plot_edges(img1, img2, edges1, edges2)
#
# # Plot connective Matches on the corners
# #Get number of bigurcations and terminations here
# matches = match_edge_descriptors(desc1, desc2)
# plot_matches(img1, img2, edges1, edges2, matches)