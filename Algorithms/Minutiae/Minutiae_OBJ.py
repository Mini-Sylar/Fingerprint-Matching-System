from Algorithms.Minutiae.Libs.basics import load_image, display_image
from Algorithms.Minutiae.Libs.enhancing import enhance_image
from Algorithms.Minutiae.Libs.minutiae import process_minutiae
from Algorithms.Minutiae.Libs.enhance_updated import FingerprintImageEnhancer
from Algorithms.Minutiae.Libs.processing import thin_image, clean_points, binarise, histogram_equalisation,thresh

# Fallback to new enhancer if errors persist
enhancer = FingerprintImageEnhancer()


def detectAndComputeMinutiae(image_path):
    # Load images
    image_raw = load_image(image_path, gray=True)
    try:
        enhanced_image = enhance_image(image_raw, skeletonise=True, min_wave_length=3)
        enh_img = enhancer.enhance(image_raw, resize=True)
        thinned = thin_image(enh_img)
        cleaned = clean_points(thinned)
    except:
        print("Using new enhancer")
        enh_img = enhancer.enhance(image_raw,resize=True)
        thinned = thin_image(enh_img)
        cleaned = clean_points(thinned)

    try:
        minutiae = process_minutiae(cleaned)
    except:
        print("Processing new enhancer image")
        minutiae = process_minutiae(cleaned)
    # Split Minutiae into terminations and bifurcations
    terminations, bifurcation,total = minutiae
    return terminations, bifurcation,total


def showEqualizedImage(image):
    display_image(histogram_equalisation(image))
