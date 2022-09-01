from datetime import datetime

import cv2
import numpy as np
from matplotlib import pyplot as plt

from SIFT import Performant_SIFT
from SIFT import SIFT_OBJ



def perform_sift_research(query, train):
    # start = datetime.now()
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread(query, 0)  # queryImage
    img2 = cv2.imread(train, 0)  # trainImage

    sift1 = SIFT_OBJ.SIFT()
    sift2 = SIFT_OBJ.SIFT()

    # Compute SIFT keypoints and descriptors
    kp1, des1 = sift1.computeKeypointsAndDescriptors(img1)
    kp2, des2 = sift2.computeKeypointsAndDescriptors(img2)

    sift2.showGaussianBlurImages()
    sift2.showDOGImages()

    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=37)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = set()
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.add(m)
    # print(datetime.now() - start)
    if len(good) > MIN_MATCH_COUNT:
        # ---------------------- Draw Results Old -----------------------
        # Estimate homography between template and scene
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        # Draw detected template in scene image
        h, w = img1.shape
        pts = np.float32([[0, 0],
                          [0, h - 1],
                          [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        h1, w1 = img1.shape
        h2, w2 = img2.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = img1
            newimg[:h2, w1:w1 + w2, i] = img2

        for m in good:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        plt.imshow(newimg)
        plt.get_current_fig_manager().canvas.set_window_title("Match Shown")
        plt.title("Matches Obtained")
        plt.show()

    # Calculate Match Score here

    # ---------------------- Draw Results Old END -----------------------
    #     # DrawResults ----- NEW
    #     result  = cv2.drawMatches(img1,kp1,img2,kp2,good,None)
    #     # result = cv2.resize(result,None,fx=2.5,fy=2.5)
    #     result = cv2.resize(result, (500, 500), interpolation= 2)
    #     cv2.imshow("Result",result)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # #     Draw Results New End
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))


def perform_sift_performant(query, train):
    Performant_SIFT.perform_sift_performant(query, train)


if __name__ == "__main__":
    # path_to_real_image = input("Enter path to original fingerprint: ")
    path_to_real_image = "Images/Real/1__M_Left_index_finger.BMP"
    # path_to_query_image = input("Enter path to image used for comparison: ")
    path_to_query_image = "Images/Altered/Easy/1__M_Left_index_finger_CR.BMP"
    # Perform Research Version
    perform_sift_research(path_to_query_image, path_to_real_image)
