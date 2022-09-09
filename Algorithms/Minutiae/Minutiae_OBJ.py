from Algorithms.Minutiae.Libs.basics import load_image, display_image
from Algorithms.Minutiae.Libs.enhancing import enhance_image
from Algorithms.Minutiae.Libs.minutiae import process_minutiae
from Algorithms.Minutiae.Libs.enhance_updated import FingerprintImageEnhancer
from Algorithms.Minutiae.Libs.processing import thin_image, clean_points, binarise, histogram_equalisation

enhancer = FingerprintImageEnhancer()


def detectAndComputeMinutiae(image_path):
    # Load images
    image_raw = load_image(image_path, gray=True)
    enhanced_image = enhance_image(image_raw, skeletonise=True, min_wave_length=3)
    # Use New Enhancer Here
    # enhanced_image = enhancer.enhance(image_raw)
    # image_enhanced = thin_image(enhanced_image)
    # image_enhanced = clean_points(image_enhanced)

    minutiae = process_minutiae(enhanced_image)
    # Split Minutiae into terminations and bifurcations
    terminations, bifurcation,total = minutiae
    return terminations, bifurcation,total


def showEqualizedImage(image):
    display_image(histogram_equalisation(image))
