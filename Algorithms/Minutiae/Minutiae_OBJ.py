from matplotlib import pyplot as plt

from Algorithms.Minutiae.Libs.basics import load_image, display_image
from Algorithms.Minutiae.Libs.edges import edge_processing
from Algorithms.Minutiae.Libs.enhancing import enhance_image
from Algorithms.Minutiae.Libs.minutiae import process_minutiae, generate_tuple_profile, plot_minutiae


class Minutiae:
    def detectAndComputeMinutiae(self, image_path):
        #         Load images
        image_raw = load_image(image_path, gray=True)
        enhanced_image = enhance_image(image_raw, skeletonise=True)
        # # display_image(enhanced_image,title="Enhanced Image")
        minutiae = process_minutiae(enhanced_image)
        terminations, bifurcation = minutiae
        # Generate Tuple Profile Here
        terminations_profile = generate_tuple_profile(terminations)
        minutiae_profile = generate_tuple_profile(bifurcation)
        # Generate Axis Here
        fig, ax = plt.subplots(1, 1)
        # Show images on Axis
        for y, x in terminations_profile.keys():
            termination = plt.Circle((x, y), radius=1, linewidth=2, color='red', fill=False)
            ax.add_artist(termination)
            plt.imshow(enhanced_image)
        for y, x in minutiae_profile.keys():
            bifurcation = plt.Circle((x, y), radius=1, linewidth=2, color='green', fill=False)
            ax.add_artist(bifurcation)
            plt.imshow(enhanced_image)
        display_image(enhanced_image,title="Terminations & Bifurcations")


fm = Minutiae()

img_test = "C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Real\\101_1.tif"
fm.detectAndComputeMinutiae(img_test)
