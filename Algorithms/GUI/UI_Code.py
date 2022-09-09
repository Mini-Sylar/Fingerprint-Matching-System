import sys
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtGui import QImageReader
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

from AlgorithmExamination import Ui_MainWindow
from Algorithms.Minutiae.Libs.matching import match_tuples
from Algorithms.Minutiae.Libs.minutiae import generate_tuple_profile
# Import Minutiae
from Algorithms.Minutiae.Minutiae_OBJ import *
# Import SIFT
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
        self.train_image = None
        self.setupUi(self)

        # Set Other Canvases Here
        self.canvases()
        # Create 2 SIFT OBJECTS HERE
        self.sift_query = SIFT()
        self.sift_train = SIFT()
        self.sift_performance = cv2.SIFT_create()
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
        # Run minutiae
        self.run_minutiae.clicked.connect(self.run_minutiae_algorithm)

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
            image = QFileDialog.getOpenFileName(self, 'Open Training Image', '', text_filter)
        if image[0]:
            try:
                #   Set label to path
                self.Path_To_Train.setText(image[0])
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
        # initialize training image here instead to since research version distorts image after processing
        self.train_image = cv2.imread(self.Path_To_Train.text(), 0)
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
            if m.distance < 0.6 * n.distance:
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
            plt.title("Matches Obtained Research Version")
            plt.tight_layout()
            # Moved canvas draw to end of plot to make sure title shows everytime
            self.canvas.draw()
            self.statusbar.showMessage("Matches found!", msecs=10000)
            self.Match_Score.setText("%d" % (len(good)))
            #     Populate some labels
            # Get Match Score Here
            if len(good) > 35:
                self.Match_Score.setStyleSheet("color:green;")
                # Set Verdict Here
                self.Verdict.setStyleSheet("color:green;")
                self.Verdict.setText("Fingerprints/Images Are A Good Match!")
            elif len(good) > 18:
                self.Match_Score.setStyleSheet("color:orange;")
                # Set Verdict Here
                self.Verdict.setStyleSheet("color:orange;")
                self.Verdict.setText("Fingerprints/Images Match With A Low Score!")

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
        self.figure_Gaussian = plt.figure(num=2, figsize=(10, 10))
        self.canvas_Gaussian = FigureCanvas(self.figure_Gaussian)
        self.toolbar_Gaussian = NavigationToolbar(self.canvas_Gaussian, self)
        self.Show_Gaussian_Images.addWidget(self.toolbar_Gaussian)
        self.Show_Gaussian_Images.addWidget(self.canvas_Gaussian)
        # Create Canvas And Add
        figure_DOG = plt.figure(num=3, figsize=(10, 10))
        self.canvas_DOG = FigureCanvas(figure_DOG)
        toolbar_DOG = NavigationToolbar(self.canvas_DOG, self)
        self.Show_DOG_Image.addWidget(toolbar_DOG)
        self.Show_DOG_Image.addWidget(self.canvas_DOG)
        #     Creating Canvas For Minutiae
        self.figure_Minutiae_Match = plt.figure(num=4, figsize=(10, 10))
        self.canvas_minutiae_match = FigureCanvas(self.figure_Minutiae_Match)
        toolbar_minutiae_match = NavigationToolbar(self.canvas_minutiae_match, self)
        self.min_matches_layout.addWidget(toolbar_minutiae_match)
        self.min_matches_layout.addWidget(self.canvas_minutiae_match)
        # Creating Canvas for Equalized Image
        self.figure_equalized, self.canvas_equalized, self.toolbar_equalized = self.initialize_canvas(5)
        self.norm_layout.addWidget(self.toolbar_equalized)
        self.norm_layout.addWidget(self.canvas_equalized)
        # Creating Canvas for Binarized Image
        self.figure_binarized, self.canvas_binarized, self.toolbar_binarized = self.initialize_canvas(6)
        self.bin_layout.addWidget(self.toolbar_binarized)
        self.bin_layout.addWidget(self.canvas_binarized)
        # Creating Canvas for Thinned Image
        self.figure_thinned, self.canvas_thinned, self.toolbar_thinned = self.initialize_canvas(7)
        self.thin_layout.addWidget(self.toolbar_thinned)
        self.thin_layout.addWidget(self.canvas_thinned)
        # Creating Canvas for Enhanced Images
        self.figure_enhanced, self.canvas_enhanced, self.toolbar_enhanced = self.initialize_canvas(8)
        self.enhance_layout.addWidget(self.toolbar_enhanced)
        self.enhance_layout.addWidget(self.canvas_enhanced)

    def initialize_canvas(self, figure_num, row: int = 10, column: int = 10):
        figure = plt.figure(num=figure_num, figsize=(row, column))
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)
        return figure, canvas, toolbar

    def show_Gaussian_SIFT_Research(self):
        self.canvas_Gaussian.figure.clear()
        # Render Images
        Gaussian_images = self.sift_train.showGaussianBlurImages()
        # create a 10 by 10 grid here
        plt.figure(num=2, figsize=(10, 10))  # specifying the overall grid size
        # Change font size to make sure everything fits in the canvas
        plt.rcParams.update({'font.size': 8})
        for i in range(len(Gaussian_images)):
            plt.subplot(7, 6, i + 1)  # the number of images in the grid is 7*6 (42)
            plt.imshow(Gaussian_images[i], cmap='Greys_r')
        plt.tight_layout()
        # Set Details in Label Here
        self.G_Scale_Count.setText(str(len(Gaussian_images)))
        self.G_Octaves.setText(str((len(Gaussian_images)) // 6))
        self.canvas_Gaussian.draw()
        self.canvas_Gaussian.updateGeometry()

    def show_DOG_SIFT_Research(self):
        self.canvas_DOG.figure.clear()
        # Render Images
        doG_images = self.sift_train.showDOGImages()
        # # create a 10 by 10 grid here
        plt.figure(num=3, figsize=(10, 10))  # specifying the overall grid size
        plt.rcParams.update({'font.size': 8})
        plt.title("Gaussian Scale Space and Extrema")
        for i in range(len(doG_images)):
            plt.subplot(7, 5, i + 1)  # the number of images in the grid is 5*5 (25)
            plt.imshow(doG_images[i], cmap='Greys_r')
        plt.tight_layout()
        self.canvas_DOG.draw()

    ###############################
    # SIFT PERFORMANT VERSION #
    ###############################

    def run_sift_performant_version(self):
        self.canvas.figure.clear()
        start = datetime.now()
        # find the keypoints and descriptors with SIFT
        self.train_image = cv2.imread(self.Path_To_Train.text(), 0)
        kp1, des1 = self.sift_performance.detectAndCompute(self.query_image, None)
        kp2, des2 = self.sift_performance.detectAndCompute(self.train_image, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=37)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for _ in range(len(matches))]
        good = set()
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.6 * n.distance:
                matchesMask[i] = [1, 0]
                good.add(m)
        print("Performant ", len(good))
        print(datetime.now() - start)
        draw_params = dict(matchColor=(0, 255, 255),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(self.query_image, kp1, self.train_image, kp2, matches, None, **draw_params)
        # create an axis and draw the images
        plt.figure(num=1)
        plt.imshow(img3)
        plt.title("Matches Obtained Performant Version")
        plt.tight_layout()
        # Move self.canvas.draw() to end to make sure title is rendered every single time
        self.canvas.draw()
        # Extra Stuff
        self.statusbar.showMessage("Matches found!", msecs=10000)
        self.Match_Score.setText("%d" % (len(good)))
        #     Populate some labels
        # Get Match Score Here
        if len(good) > 37:
            self.Match_Score.setStyleSheet("color:blue;")
            # Set Verdict Here
            self.Verdict.setStyleSheet("color:green;")
            self.Verdict.setText("Fingerprints/Images Are A Good Match!")
        elif len(good) > 15:
            self.Match_Score.setStyleSheet("color:orange;")
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

    ###############################
    # Minutiae Algorithm
    ###############################
    def run_minutiae_algorithm(self):
        self.canvas_minutiae_match.figure.clear()
        coor_termination1, coor_bifurcation1,total_bif_term1 = detectAndComputeMinutiae(self.Path_To_Train.text())
        coor_termination2, coor_bifurcation2,total_bif_term2 = detectAndComputeMinutiae(self.Path_To_Query.text())
        # Image Profiles
        img_profile1_term = generate_tuple_profile(coor_termination1)  # Image 1 Termination
        img_profile1_bif = generate_tuple_profile(coor_bifurcation1)  # Image 1 Bifurcation
        # This was created only for display purposes
        self.termin_disp = img_profile1_term
        self.bif_disp = img_profile1_bif
        # Image 2 Profiles
        img_profile2_term = generate_tuple_profile(coor_termination2)
        img_profile2_bif = generate_tuple_profile(coor_bifurcation2)
        # For caluclation process
        calc_bif_term1  = generate_tuple_profile(total_bif_term1)
        calc_bif_term2  = generate_tuple_profile(total_bif_term2)
        # Load Images here (should already be loaded when tranformed into class)
        # Plot Terminations as red and bifurcations as blue
        train_image = load_image(self.Path_To_Train.text())
        query_image = load_image(self.Path_To_Query.text())
        # Plot Termination and Bifurcation Circle
        ax = self.figure_Minutiae_Match.subplots(1, 2)
        self.figure_Minutiae_Match.suptitle('Matches Obtained Minutiae', fontsize=12)
        # Images Here
        for y, x in img_profile1_term.keys():
            termination = plt.Circle((x, y), radius=1, linewidth=2, color='red', fill=False)
            ax[0].add_artist(termination)
            ax[0].imshow(train_image)

        for y, x in img_profile1_bif.keys():
            bifurcation = plt.Circle((x, y), radius=1, linewidth=2, color='blue', fill=False)
            ax[0].add_artist(bifurcation)
            ax[0].imshow(train_image)

        # FOr Query Image
        for y, x in img_profile2_term.keys():
            termination = plt.Circle((x, y), radius=1, linewidth=2, color='red', fill=False)
            ax[1].add_artist(termination)
            ax[1].imshow(query_image)

        for y, x in img_profile1_bif.keys():
            bifurcation = plt.Circle((x, y), radius=1, linewidth=2, color='blue', fill=False)
            ax[1].add_artist(bifurcation)
            ax[1].imshow(query_image)

        # # Common points Termination
        common_points_query_termination, common_points_train_termination = match_tuples(img_profile1_term,
                                                                                        img_profile2_term)
        common_points_query_bifurcation, common_points_train_bifurcation = match_tuples(img_profile1_bif,
                                                                                        img_profile2_bif)

        common_points_both_train,common_points_both_query = match_tuples(calc_bif_term1,calc_bif_term2)

        print(f"common_both_train {len(common_points_both_train)} common_both_query {len(common_points_both_query)}")
        self.minutiae_value  = len(common_points_both_query)

        # Draw Lines to Match points, points with "X" means there was no match on the other image
        for x, y in common_points_query_termination:
            # Reverse points since ConnectPatch is flipped
            xy = (y, x)
            con = ConnectionPatch(xyA=xy, xyB=xy, coordsA="data", coordsB="data",
                                  axesA=ax[0], axesB=ax[1], color="red")
            ax[1].add_artist(con)

            ax[0].plot(x, y, 'rx', markersize=5)
            ax[1].plot(x, y, 'rx', markersize=5)

        for x, y in common_points_query_bifurcation:
            # Reverse points since ConnectPatch is flipped
            xy = (y, x)
            con = ConnectionPatch(xyA=xy, xyB=xy, coordsA="data", coordsB="data",
                                  axesA=ax[0], axesB=ax[1], color="blue")
            ax[1].add_artist(con)
            ax[0].plot(x, y, 'bx', markersize=5)
            ax[1].plot(x, y, 'bx', markersize=5)

        self.canvas_minutiae_match.draw()
        self.generateExtraMinutiae()

    #         set label texts here
    def setMinutiaeLabelText(self):
        ...

    def generateExtraMinutiae(self):
        train_image = load_image(self.Path_To_Train.text(), gray=True)
        # Equalized Image
        equalized_image = histogram_equalisation(train_image)
        ax_equalized = self.figure_equalized.subplots(1, 1)
        ax_equalized.imshow(np.hstack((train_image, equalized_image)), cmap='gray')
        self.figure_equalized.suptitle('Histogram Equalization/Normalization', fontsize=12)

        # Binarized Image
        binarized_image = binarise(equalized_image)
        ax_binarized = self.figure_binarized.subplots(1, 1)
        ax_binarized.imshow(np.hstack((train_image, binarized_image)), cmap='gray')
        self.figure_binarized.suptitle('Binarized Image', fontsize=12)

        # Thinned Image
        thinned_image = thin_image(train_image)
        ax_thinned = self.figure_thinned.subplots(1, 1)
        ax_thinned.imshow(thinned_image, cmap='gray')
        self.figure_thinned.suptitle('Thinned Image', fontsize=12)

        # Enhanced Image
        enhanced_image = enhance_image(train_image, skeletonise=True, min_wave_length=3)
        ax_enhanced = self.figure_enhanced.subplots(1, 1)
        # FOr Query Image
        for y, x in self.termin_disp.keys():
            termination = plt.Circle((x, y), radius=1, linewidth=2, color='red', fill=False)
            ax_enhanced.add_artist(termination)
            ax_enhanced.imshow(enhanced_image)
        for y, x in self.bif_disp.keys():
            bifurcation = plt.Circle((x, y), radius=1, linewidth=2, color='blue', fill=False)
            ax_enhanced.add_artist(bifurcation)
            ax_enhanced.imshow(enhanced_image)
        # Create Legend Here
        patches = [mpatches.Patch(color="red", label="Terminations"),
                   mpatches.Patch(color="blue", label="Bifurcations")]
        # put those patched as legend-handles into the legend
        ax_enhanced.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax_enhanced.imshow(enhanced_image, cmap="gray")
        self.figure_enhanced.suptitle('Enhanced Image With Terminations and Bifurcations', fontsize=12)

        self.canvas_equalized.draw()
        self.canvas_binarized.draw()
        self.canvas_thinned.draw()

        self.setMinutiaeLabelScores()
        # todo: Clean up and put into functions
        # todo: Update labels with information from results

    def setMinutiaeLabelScores(self):
        self.minutiae_terminations.setText(str(len(self.termin_disp.keys())))  # Terminations score here
        self.minutiae_bifurcations.setText(str(len(self.bif_disp.keys())))  # Bifurcation score
        # Verdict logic here
        if self.minutiae_value < 7:
            self.minutiae_verdict.setStyleSheet("color:orange;")
            self.minutiae_verdict.setText("Fingerprints Match With A Low Score")
            self.minutiae_min_match.setStyleSheet("color:orange;")
            self.min_score_value.setText(str(self.minutiae_value))  # Match Score here
        elif self.minutiae_value < 3:
            self.minutiae_verdict.setStyleSheet("color:red;")
            self.minutiae_verdict.setText("Fingerprints Do not Match")
            self.minutiae_min_match.setStyleSheet("color:red;")
            self.min_score_value.setText(str(self.minutiae_value))  # Match Score here
        else:
            self.minutiae_verdict.setStyleSheet("color:green;")
            self.minutiae_verdict.setText("Fingerprints are a good match")
            self.minutiae_min_match.setStyleSheet("color:green;")
            self.min_score_value.setText(str(self.minutiae_value))  # Match Score here


if __name__ == "__main__":
    app = QApplication(sys.argv)
    UI = UiCode()
    UI.showNormal()
    sys.exit(app.exec_())
