from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

from Algorithms.Minutiae.Libs.basics import load_image, display_image
from Algorithms.Minutiae.Libs.enhancing import enhance_image
from Algorithms.Minutiae.Libs.matching import match_tuples
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
        self.terminations_profile = generate_tuple_profile(terminations)
        self.minutiae_profile = generate_tuple_profile(bifurcation)
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


# # Create 2 instances of minutiae here
# fm = Minutiae()
# fm2 = Minutiae()
# img_test = "C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Real\\101_2.tif"
# img_test2 = "C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Altered\\Easy\\101_2.tif"
#
# # Get Minutiae And Bifurcation From Here
# coor_termination1, coor_bifurcation1 = fm.detectAndComputeMinutiae(img_test)
# coor_termination2, coor_bifurcation2 = fm.detectAndComputeMinutiae(img_test2)
#
# # Image Profiles
# img_profile1_term = generate_tuple_profile(coor_termination1)  # Image 1 Termination
# img_profile1_bif = generate_tuple_profile(coor_bifurcation1)  # Image 1 Bifurcation
#
# # Image 2 Profiles
# img_profile2_term = generate_tuple_profile(coor_termination2)
# img_profile2_bif = generate_tuple_profile(coor_bifurcation2)
#
# # Load Images here (should already be loaded when tranformed into class)
# # Plot Terminations as red and bifurcations as blue
# train_image = load_image(img_test)
# query_image = load_image(img_test2)
# # ----- Maybe include enhanced images here
#
# # Plot Termination and Bifurcation Circles
# fig,ax  = plt.subplots(1, 2)
# # Images Here
# for y, x in img_profile1_term.keys():
#     termination = plt.Circle((x, y), radius=1, linewidth=2, color='red', fill=False)
#     ax[0].add_artist(termination)
#     ax[0].imshow(train_image)
#
# for y, x in img_profile1_bif.keys():
#     bifurcation = plt.Circle((x, y), radius=1, linewidth=2, color='blue', fill=False)
#     ax[0].add_artist(bifurcation)
#     ax[0].imshow(train_image)
#
# # FOr Query Image
# for y, x in img_profile2_term.keys():
#     termination = plt.Circle((x, y), radius=1, linewidth=2, color='red', fill=False)
#     ax[1].add_artist(termination)
#     ax[1].imshow(train_image)
#
# for y, x in img_profile1_bif.keys():
#     bifurcation = plt.Circle((x, y), radius=1, linewidth=2, color='blue', fill=False)
#     ax[1].add_artist(bifurcation)
#     ax[1].imshow(train_image)
#
# # # Common points Termination
# common_points_query_termination, common_points_train_termination = match_tuples(img_profile1_term, img_profile2_term)
# common_points_query_bifurcation, common_points_train_bifurcation = match_tuples(img_profile1_bif, img_profile2_bif)
#
# # Draw Lines to Match points, points with "X" means there was no match on the other image
# for x, y in common_points_query_termination:
#     # Reverse points since ConnectPatch is flipped
#     xy = (y, x)
#     con = ConnectionPatch(xyA=xy, xyB=xy, coordsA="data", coordsB="data",
#                           axesA=ax[0], axesB=ax[1], color="red")
#     ax[1].add_artist(con)
#
#     ax[0].plot(x, y, 'rx', markersize=5)
#     ax[1].plot(x, y, 'rx', markersize=5)
#
# for x, y in common_points_query_bifurcation:
#     # Reverse points since ConnectPatch is flipped
#     xy = (y, x)
#     con = ConnectionPatch(xyA=xy, xyB=xy, coordsA="data", coordsB="data",
#                           axesA=ax[0], axesB=ax[1], color="blue")
#     ax[1].add_artist(con)
#
#     ax[0].plot(x, y, 'bx', markersize=5)
#     ax[1].plot(x, y, 'bx', markersize=5)
#
# plt.show()
