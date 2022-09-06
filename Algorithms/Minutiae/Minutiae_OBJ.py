from matplotlib import pyplot as plt

from Algorithms.Minutiae.Libs.basics import load_image, display_image
from Algorithms.Minutiae.Libs.edges import edge_processing
from Algorithms.Minutiae.Libs.enhancing import enhance_image
from Algorithms.Minutiae.Libs.minutiae import process_minutiae, generate_tuple_profile, plot_minutiae


class Minutiae:
    def detectAndComputeMinutiae(self, image_path):
        #         Load images
        image_raw = load_image(image_path, True)
        enhanced_image = enhance_image(image_raw, skeletonise=True)
        # # display_image(enhanced_image,title="Enhanced Image")
        minutiae = process_minutiae(enhanced_image)
        profile = generate_tuple_profile(minutiae)

        fig, ax = plt.subplots(1, 1)

        for y, x in profile.keys():
            termination = plt.Circle((x, y), radius=2, linewidth=2, color='red', fill=False)
            ax.add_artist(termination)
            plt.imshow(enhanced_image)
        plt.show()



fm = Minutiae()

img_test = "C:\\Users\\Andy\\OneDrive\\Desktop\\FingerPrint Matching Project\\Fingerprint-Matching-System\\Algorithms\\Images\\Real\\1__M_Left_index_finger.BMP"
fm.detectAndComputeMinutiae(img_test)
