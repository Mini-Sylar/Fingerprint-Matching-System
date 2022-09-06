import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

from Algorithms.Minutiae.Libs.basics import load_image, display_image
from Algorithms.Minutiae.Libs.enhancing import enhance_image
from Algorithms.Minutiae.Libs.matching import match_tuples, evaluate
from Algorithms.Minutiae.Libs.minutiae import process_minutiae, generate_tuple_profile


class Minutiae:
    def __init__(self):
        self.thinned_image = None
        self.terminations_profile = None
        self.minutiae_profile = None

    def detectAndComputeMinutiae(self, image_path):
        # Load images
        image_raw = load_image(image_path, gray=True)
        self.thinned_image = enhance_image(image_raw, skeletonise=True)
        minutiae = process_minutiae(self.thinned_image)
        # Split Minutiae into terminations and bifurcations
        terminations, bifurcation = minutiae
        # Generate Tuple Profile Here
        # print(terminations)
        # print(bifurcation)
        self.terminations_profile = generate_tuple_profile(terminations)
        self.minutiae_profile = generate_tuple_profile(bifurcation)
        # Draw Graph here
        # self.plot_terminations_bifurcations()
        return terminations, bifurcation

    def plot_terminations_bifurcations(self):
        fig, ax = plt.subplots(1, 1)
        # Show images on Axis
        for y, x in self.terminations_profile.keys():
            termination = plt.Circle((x, y), radius=1, linewidth=2, color='red', fill=False)
            ax.add_artist(termination)
            plt.imshow(self.thinned_image)
        for y, x in self.minutiae_profile.keys():
            bifurcation = plt.Circle((x, y), radius=1, linewidth=2, color='blue', fill=False)

            ax.add_artist(bifurcation)
            plt.imshow(self.thinned_image)
        display_image(self.thinned_image, title="Terminations & Bifurcations")


fm = Minutiae()
fm2 = Minutiae()
img_test = "C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Real\\101_2.tif"
img_test2 = "C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Altered\\Easy\\101_2.tif"

coor_termination1, coor_bifurcation1 = fm.detectAndComputeMinutiae(img_test)
image_1_coor = coor_termination1 + coor_bifurcation1

coor_termination2, coor_bifurcation2 = fm2.detectAndComputeMinutiae(img_test2)
image_2_coor = coor_termination2 + coor_bifurcation2

img_profile1 = generate_tuple_profile(image_1_coor)
img_profile2 = generate_tuple_profile(image_2_coor)

# Common points
common_points_query, common_points_train = match_tuples(img_profile1, img_profile2)
if evaluate(common_points_train, coor_termination1, coor_termination2):
    print("Matched")


a = load_image(img_test)
b = load_image(img_test2)

fig, ax = plt.subplots(1, 2)
for y, x in img_profile1.keys():
    termination = plt.Circle((x, y), radius=1, linewidth=2, color='red', fill=False)
    ax[0].add_artist(termination)
    ax[0].imshow(a)
    print("ImageProfile", y, x)

for y, x in img_profile2.keys():
    termination = plt.Circle((x, y), radius=1, linewidth=2, color='red', fill=False)
    ax[1].add_artist(termination)
    ax[1].imshow(b)

print(common_points_query)

# Drawing Lines
x, y = 75, 36

for x, y in common_points_query:
    xy = (y, x)
    print("in common points", xy)
    con = ConnectionPatch(xyA=xy, xyB=xy, coordsA="data", coordsB="data",
                          axesA=ax[0], axesB=ax[1], color="green")
    ax[1].add_artist(con)

    ax[0].plot(x, y, 'gx', markersize=5)
    ax[1].plot(x, y, 'gx', markersize=5)

# display_image(np.hstack((ax[0].fig,ax[1].fig)),title="Equalized Image Comparison")
plt.show()
