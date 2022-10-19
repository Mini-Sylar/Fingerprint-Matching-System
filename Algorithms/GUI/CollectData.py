import itertools
import re
import math
from pathlib import Path
import glob
import os
from datetime import datetime

import cv2
import numpy as np
import xlsxwriter
from PyQt5.QtGui import QImageReader

from Algorithms.Minutiae.Libs.matching import match_tuples
from Algorithms.Minutiae.Libs.minutiae import generate_tuple_profile
# Import Minutiae
from Algorithms.Minutiae.Minutiae_OBJ import *
# Import SIFT
from Algorithms.SIFT.SIFT_OBJ import SIFT

#
# # collect Data Here
workbook = xlsxwriter.Workbook(f"Data_Subject2.xlsx")
worksheet = workbook.add_worksheet()
worksheet.set_column(0, 13, 50)
# Set titles here
sheet_titles = {0: "Fingerprint image",

                1: "Alteration Type",

                2: "Match Score (SIFT)",
                3: "Time (SIFT)",
                4: "Verdict (SIFT)",

                5: "Match Score (Minutiae)",
                6: "Time (Minutiae)",
                7: "Verdict (Minutiae)",
                }
for value, title in enumerate(sheet_titles.values()):
    worksheet.write(0, value, title)

# Glob Here
file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

# Path to Real
path_to_real = "C:\\Users\\Ugo\\Desktop\\Fingerprint-Matching-System\\SOCOFing\\Real\\"
# Get Real Images
real_images = []
for img in sorted(glob.glob(f"{path_to_real}*.BMP"),
                  key=get_order):
    real_images.append(img.strip(path_to_real))

# # Get Altered Images
path_to_altered = "C:\\Users\\Ugo\\Desktop\\Fingerprint-Matching-System\\SOCOFing\\Altered\\Altered-Easy\\"
altered_easy = []
for img in sorted(glob.glob(f"{path_to_altered}*.BMP"), key=get_order):
    # n= cv2.imread(img)
    altered_easy.append(img.strip(path_to_altered))

counter = 0
counter_end = 3
# Initial SIFT
sift_query = SIFT()
sift_train = SIFT()
row= 1
altered_type ="Easy"
verdict = ""
verdict_minutiae = ""
# Loop to pairs
for i in range(5,6): # where to end multiply by 6 control where to start and end,
    for j in range(counter, counter_end):
        MIN_MATCH_COUNT = 18
        query = cv2.imread(path_to_real+real_images[i],0)
        train = cv2.imread(path_to_altered+altered_easy[j],0)
        kp1, des1 = sift_query.computeKeypointsAndDescriptors(query)
        kp2, des2 = sift_train.computeKeypointsAndDescriptors(train)

        start = datetime.now()

        # # Initialize and use FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=37)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        #
        # # Lowe's ratio test
        good = set()
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.add(m)
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        time_taken = datetime.now() - start
    #     Verdict Here
        if len(good) > 35:
            verdict = "Fingerprints/Images Are A Good Match!"
        elif len(good) > 18:
            verdict = "Fingerprints/Images Match With A Low Score!"
        else:
            verdict = "Not enough matches are found"

    #     MINUTIAE
        start = datetime.now()
        coor_termination1, coor_bifurcation1, total_bif_term1 = detectAndComputeMinutiae(path_to_real+real_images[i])
        coor_termination2, coor_bifurcation2, total_bif_term2 = detectAndComputeMinutiae(path_to_altered+altered_easy[j])
        # Image Profiles
        img_profile1_term = generate_tuple_profile(coor_termination1)  # Image 1 Termination
        img_profile1_bif = generate_tuple_profile(coor_bifurcation1)  # Image 1 Bifurcation
        # This was created only for display purposes
        termin_disp = img_profile1_term
        bif_disp = img_profile1_bif
        # Image 2 Profiles
        img_profile2_term = generate_tuple_profile(coor_termination2)
        img_profile2_bif = generate_tuple_profile(coor_bifurcation2)
        # For caluclation process
        calc_bif_term1 = generate_tuple_profile(total_bif_term1)
        calc_bif_term2 = generate_tuple_profile(total_bif_term2)
        # Load Images here (should already be loaded when tranformed into class)
        # # Common points Termination
        common_points_query_termination, common_points_train_termination = match_tuples(img_profile1_term,
                                                                                        img_profile2_term)
        common_points_query_bifurcation, common_points_train_bifurcation = match_tuples(img_profile1_bif,
                                                                                        img_profile2_bif)

        common_points_both_train, common_points_both_query = match_tuples(calc_bif_term1, calc_bif_term2)

        minutiae_value = len(common_points_both_query)
        # Time ends here
        time_taken_minutiae = datetime.now() - start
    #     Minutiae Verdict
        if minutiae_value >= 7:
            verdict_minutiae = "Fingerprints Are A Good Match"
        elif minutiae_value >= 3:
            verdict_minutiae= "Fingerprints Match With A Really Low Score"
        else:
            verdict_minutiae="Fingerprints do not match"
    #     Record DATA
        # Write Query Image Here
        worksheet.write(row, 0, f"{real_images[i]}\n{altered_easy[j]}")
        # Add Alteration Type
        worksheet.write(row, 1, altered_type)
        #### SIFT ####
        worksheet.write(row, 2, str(len(good)))
        # Time
        worksheet.write(row, 3, time_taken)
        # Verdict
        worksheet.write(row, 4, verdict)
        #--------- MINUTIAE ---------------
        worksheet.write(row,5,str(len(common_points_both_query)))
        worksheet.write(row,6,time_taken_minutiae)
        worksheet.write(row,7,verdict_minutiae)
        row = row + 2
    counter += 3
    counter_end += 3

print("Saved Workbook successfully")
workbook.close()