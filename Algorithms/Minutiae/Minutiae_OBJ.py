from matplotlib import pyplot as plt

from Algorithms.Minutiae.Libs.basics import load_image, display_image
from Algorithms.Minutiae.Libs.enhancing import enhance_image
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
        # Draw Graph here
        self.plot_terminations_bifurcations()

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

img_test = "C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Real\\101_1.tif"
fm.detectAndComputeMinutiae(img_test)
