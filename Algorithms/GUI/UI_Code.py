import sys

import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog

from AlgorithmExamination import Ui_MainWindow
from Algorithms.SIFT.SIFT_OBJ import SIFT


class UI_Code(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(UI_Code, self).__init__()
        self.kp1 = None
        self.des1 = None
        self.setupUi(self)
        # Create 2 SIFT OBJECTS HERE
        self.sift_query = SIFT()
        self.sift_train = SIFT()
        # Connect functions to UI here
        self.connect_functions()

    def connect_functions(self):
        self.QueryImage.clicked.connect(self.load_image_query)
        self.TrainImage.clicked.connect(self.load_image_train)

    def load_image_query(self, image=None):
        if not image:
            image = QFileDialog.getOpenFileName(self, 'Open Query Image', '', ".BMP(*.bmp)")
        if image[0]:
            try:
                #   Set label to path
                self.Path_To_Query.setText(image[0])
                #   Assign Image to sift_query
                query_image = cv2.imread(image[0], 0)
                self.kp1, self.des1 = self.sift_query.computeKeypointsAndDescriptors(query_image)
                #   Add info on status bar
                self.statusbar.showMessage("Successfully loaded query image",msecs=5000)
            except AttributeError as e:
                print(e)
                self.Path_To_Query.setText("Invalid image file loaded")
                self.statusbar.showMessage("Unable to process image, try again", msecs=5000)

    def load_image_train(self, image=None):
        if not image:
            image = QFileDialog.getOpenFileName(self, 'Open Query Image', '', ".BMP(*.bmp)")
        if image[0]:
            try:
                #   Set label to path
                self.Path_To_Train.setText(image[0])
                #   Assign Image to sift_query
                train_image = cv2.imread(image[0], 0)
                self.kp1, self.des1 = self.sift_train.computeKeypointsAndDescriptors(train_image)
                #   Add info on status bar
                self.statusbar.showMessage("Successfully loaded training image",msecs=5000)
            except AttributeError as e:
                print(e)
                self.Path_To_Train.setText("Invalid image file loaded")
                self.statusbar.showMessage("Unable to process image, try again", msecs=5000)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    UI = UI_Code()
    UI.show()
    sys.exit(app.exec_())
