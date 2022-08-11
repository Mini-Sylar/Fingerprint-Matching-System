import sys
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtGui import QImageReader
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from AlgorithmExamination import Ui_MainWindow
from Algorithms.SIFT.SIFT_OBJ import SIFT

text_filter = "Images ({})".format(
    " ".join(["*.{}".format(fo.data().decode()) for fo in QImageReader.supportedImageFormats()]))


class UiCode(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(UiCode, self).__init__()
        self.toolbar = None
        self.canvas = None
        self.figure_match = None
        self.canvas_DOG = None
        self.canvas_Gaussian = None
        self.query_image = None
        self.des2 = None
        self.kp2 = None
        self.train_image = None
        self.kp1 = None
        self.des1 = None
        self.setupUi(self)

        # Set Other Canvases Here
        self.canvases()
        # Create 2 SIFT OBJECTS HERE
        self.sift_query = SIFT()
        self.sift_train = SIFT()
        # Connect functions to UI here
        self.set_parameters_sift()
        self.connect_functions()

    def connect_functions(self):
        self.QueryImage.clicked.connect(self.load_image_query)
        self.TrainImage.clicked.connect(self.load_image_train)
        self.run_sift_research.clicked.connect(self.enableButtons)
        self.generate_DOG_images.clicked.connect(self.show_DOG_SIFT_Research)
        self.generate_gaussian_images.clicked.connect(self.show_Gaussian_SIFT_Research)
        # Run SIFT PERFORMANT VERSION
        self.run_sift_performance.clicked.connect(self.run_sift_performant_version)

    def check_if_path_filled(self):
        default_text_train = "Path to training image will show here"
        default_text_query = "Path to query image will show here"
        if self.Path_To_Train.text() != default_text_train and self.Path_To_Query.text() != default_text_query:
            self.run_sift_research.setEnabled(True)
            self.run_sift_performance.setEnabled(True)
            self.run_minutiae.setEnabled(True)

    def enableButtons(self):
        self.run_sift_research_version()
        self.generate_DOG_images.setEnabled(True)
        self.generate_gaussian_images.setEnabled(True)

    def load_image_query(self, image=None):
        """Load query image here, throws an error if image is invalid"""
        if not image:
            image = QFileDialog.getOpenFileName(self, 'Open Query Image', '', text_filter)
        if image[0]:
            try:
                #   Set label to path
                self.Path_To_Query.setText(image[0])
                #   Assign Image to sift_query
                self.query_image = cv2.imread(image[0], 0)
                #   Add info on status bar
                self.statusbar.showMessage("Successfully loaded query image", msecs=5000)
                self.check_if_path_filled()
            except AttributeError as e:
                print(e)
                self.Path_To_Query.setText("Invalid image file loaded")
                self.statusbar.showMessage("Unable to process image, try again", msecs=5000)

    def load_image_train(self, image=None):
        """Load training image here, throws an error if image is invalid"""
        if not image:
            image = QFileDialog.getOpenFileName(self, 'Open Query Image', '', text_filter)
        if image[0]:
            try:
                #   Set label to path
                self.Path_To_Train.setText(image[0])
                #   Assign Image to sift_query
                self.train_image = cv2.imread(image[0], 0)

                #   Add info on status bar
                self.statusbar.showMessage("Successfully loaded training image", msecs=5000)
                self.check_if_path_filled()
            except AttributeError as e:
                print(e)
                self.Path_To_Train.setText("Invalid image file loaded")
                self.statusbar.showMessage("Unable to process image, try again", msecs=5000)

    def set_parameters_sift(self):
        """Sets the parameters being used in both algorithms, might make them adjustable in future"""
        self.Sigma.setText('1.6')
        self.Min_Match.setText("18")
        self.Assumed_Blur.setText("0.5")

    def run_sift_research_version(self):
        """ Runs the research version of the sift algorithm
            Allows you to generate DOG images and Gaussian images
        """
        self.canvas.figure.clear()
        self.statusbar.showMessage("Running research version of SIFT", msecs=10000)
        MIN_MATCH_COUNT = 18
        kp1, des1 = self.sift_query.computeKeypointsAndDescriptors(self.query_image)
        kp2, des2 = self.sift_train.computeKeypointsAndDescriptors(self.train_image)

        # Initialize and use FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=37)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = set()
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.add(m)
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

            # Draw detected template in scene image
            h, w = self.query_image.shape
            pts = np.float32([[0, 0],
                              [0, h - 1],
                              [w - 1, h - 1],
                              [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(self.train_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            h1, w1 = self.query_image.shape
            h2, w2 = img2.shape
            nWidth = w1 + w2
            nHeight = max(h1, h2)
            hdif = int((h2 - h1) / 2)
            newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

            for i in range(3):
                newimg[hdif:hdif + h1, :w1, i] = self.query_image
                newimg[:h2, w1:w1 + w2, i] = img2

            for m in good:
                pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
                pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
                cv2.line(newimg, pt1, pt2, (255, 0, 0))
            # create an axis and draw the images
            plt.figure(num=1)
            plt.imshow(newimg)
            self.canvas.draw()
            plt.title("Matches Obtained Research Version")
            plt.tight_layout()
            self.statusbar.showMessage("Matches found!", msecs=10000)
            #     Populate some labels
            # Get Match Score Here
            if len(good) > 37:
                self.Match_Score.setStyleSheet("color:green;")
                self.Match_Score.setText("%d" % (len(good)))
                # Set Verdict Here
                self.Verdict.setStyleSheet("color:green;")
                self.Verdict.setText("Fingerprints/Images Are A Good Match!")
            elif len(good) > 18:
                self.Match_Score.setStyleSheet("color:orange;")
                self.Match_Score.setText("%d" % (len(good)))
                # Set Verdict Here
                self.Verdict.setStyleSheet("color:orange;")
                self.Verdict.setText("Fingerprints/Images Match With A Really Low Score!")

        else:
            # TODO: Add clearing of diagram once matches are too low
            self.statusbar.showMessage("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT),
                                       msecs=10000)
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            # Set Match Score
            self.Match_Score.setStyleSheet("color:red;")
            self.Match_Score.setText("%d" % (len(good)))
            # Set Verdict Here
            self.Verdict.setStyleSheet("color:red;")
            self.Verdict.setText("Fingerprints/Images Are Not A Match!")

    def canvases(self):
        # Create Canvas for Matches
        # Set Canvases Here
        self.figure_match = plt.figure(num=1)
        self.canvas = FigureCanvas(self.figure_match)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.Final_Image_Container.addWidget(self.toolbar)
        self.Final_Image_Container.addWidget(self.canvas)
        # Create Canvas For Gaussian Images
        figure_Gaussian = plt.figure(num=2, figsize=(10, 10))
        self.canvas_Gaussian = FigureCanvas(figure_Gaussian)
        toolbar_Gaussian = NavigationToolbar(self.canvas_Gaussian, self)
        self.Show_Gaussian_Images.addWidget(toolbar_Gaussian)
        self.Show_Gaussian_Images.addWidget(self.canvas_Gaussian)
        # Create Canvas And Add
        figure_DOG = plt.figure(num=3, figsize=(10, 10))
        self.canvas_DOG = FigureCanvas(figure_DOG)
        # canvas_DOG.setParent(canvas_DOG)
        toolbar_DOG = NavigationToolbar(self.canvas_DOG, self)
        self.Show_DOG_Image.addWidget(toolbar_DOG)
        self.Show_DOG_Image.addWidget(self.canvas_DOG)

    def show_Gaussian_SIFT_Research(self):
        self.canvas_Gaussian.figure.clear()
        # Render Images
        Gaussian_images = self.sift_train.showGaussianBlurImages()
        # create a 10 by 10 grid here
        plt.figure(num=2, figsize=(10, 10))  # specifying the overall grid size
        for i in range(len(Gaussian_images)):
            plt.subplot(7, 6, i + 1)  # the number of images in the grid is 7*6 (42)
            plt.imshow(Gaussian_images[i], cmap='Greys_r')
            self.canvas_Gaussian.draw()
        plt.tight_layout()

    def show_DOG_SIFT_Research(self):
        self.canvas_DOG.figure.clear()
        # Render Images
        doG_images = self.sift_train.showDOGImages()
        # # create a 10 by 10 grid here
        plt.figure(num=3, figsize=(10, 10))  # specifying the overall grid size
        for i in range(len(doG_images)):
            plt.subplot(7, 5, i + 1)  # the number of images in the grid is 5*5 (25)
            plt.imshow(doG_images[i], cmap='Greys_r')
            self.canvas_DOG.draw()
        plt.tight_layout()

    ###############################
    # SIFT PERFORMANT VERSION #
    ###############################

    def run_sift_performant_version(self):
        self.canvas.figure.clear()
        start = datetime.now()
        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.query_image, None)
        kp2, des2 = sift.detectAndCompute(self.train_image, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for _ in range(len(matches))]
        good = set()
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                good.add(m)
        print(datetime.now() - start)
        draw_params = dict(matchColor=(0, 255, 255),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(self.query_image, kp1, self.train_image, kp2, matches, None, **draw_params)
        # create an axis and draw the images
        plt.figure(num=1)
        plt.imshow(img3)
        self.canvas.draw()
        plt.title("Matches Obtained Performant Version")
        plt.tight_layout()
        # Extra Stuff
        self.statusbar.showMessage("Matches found!", msecs=10000)
        #     Populate some labels
        # Get Match Score Here
        if len(good) > 37:
            self.Match_Score.setStyleSheet("color:green;")
            self.Match_Score.setText("%d" % (len(good)))
            # Set Verdict Here
            self.Verdict.setStyleSheet("color:green;")
            self.Verdict.setText("Fingerprints/Images Are A Good Match!")
        elif len(good) > 15:
            self.Match_Score.setStyleSheet("color:orange;")
            self.Match_Score.setText("%d" % (len(good)))
            # Set Verdict Here
            self.Verdict.setStyleSheet("color:orange;")
            self.Verdict.setText("Fingerprints/Images Match With A Really Low Score!")
        else:
            self.statusbar.showMessage("Not enough matches are found %d/37" % (len(good)),
                                       msecs=10000)
            # Set Match Score
            self.Match_Score.setStyleSheet("color:red;")
            self.Match_Score.setText("%d" % (len(good)))
            # Set Verdict Here
            self.Verdict.setStyleSheet("color:red;")
            self.Verdict.setText("Fingerprints/Images Are Not A Match!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    UI = UiCode()
    UI.showNormal()
    sys.exit(app.exec_())
