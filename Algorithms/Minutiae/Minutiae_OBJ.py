import numpy as np

from Algorithms.Minutiae.Libs.minutiae import plot_minutiae, process_minutiae, generate_tuple_profile
from Algorithms.Minutiae.Libs.matching import match_tuples, evaluate
from Algorithms.Minutiae.Libs.edges import match_edge_descriptors
from Algorithms.Minutiae.Libs.basics import load_image
from Algorithms.Minutiae.Libs.enhancing import enhance_image
from Algorithms.Minutiae.Libs.edges import edge_processing, sift_match


class Minutiae:
    def __init__(self, threshold=125):
        self.threshold = threshold

    def detectAndComputeMinutiae(self):
        ...
