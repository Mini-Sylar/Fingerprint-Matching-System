# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AlgorithmExamination.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1027, 619)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.Algorithm_Layout = QtWidgets.QGridLayout()
        self.Algorithm_Layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.Algorithm_Layout.setObjectName("Algorithm_Layout")
        self.SIFT = QtWidgets.QTabWidget(self.centralwidget)
        self.SIFT.setObjectName("SIFT")
        self.SIFT_Matches = QtWidgets.QWidget()
        self.SIFT_Matches.setObjectName("SIFT_Matches")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.SIFT_Matches)
        self.verticalLayout.setContentsMargins(-1, -1, -1, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.Final_Image_Container = QtWidgets.QVBoxLayout()
        self.Final_Image_Container.setObjectName("Final_Image_Container")
        self.verticalLayout.addLayout(self.Final_Image_Container)
        self.Info_Container = QtWidgets.QFormLayout()
        self.Info_Container.setContentsMargins(-1, -1, -1, 0)
        self.Info_Container.setObjectName("Info_Container")
        self.label_1 = QtWidgets.QLabel(self.SIFT_Matches)
        self.label_1.setObjectName("label_1")
        self.Info_Container.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_1)
        self.Match_Score = QtWidgets.QLabel(self.SIFT_Matches)
        self.Match_Score.setText("")
        self.Match_Score.setObjectName("Match_Score")
        self.Info_Container.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.Match_Score)
        self.label_6 = QtWidgets.QLabel(self.SIFT_Matches)
        self.label_6.setObjectName("label_6")
        self.Info_Container.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.Min_Match = QtWidgets.QLabel(self.SIFT_Matches)
        self.Min_Match.setText("")
        self.Min_Match.setObjectName("Min_Match")
        self.Info_Container.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.Min_Match)
        self.label_2 = QtWidgets.QLabel(self.SIFT_Matches)
        self.label_2.setObjectName("label_2")
        self.Info_Container.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.Sigma = QtWidgets.QLabel(self.SIFT_Matches)
        self.Sigma.setText("")
        self.Sigma.setObjectName("Sigma")
        self.Info_Container.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.Sigma)
        self.label_3 = QtWidgets.QLabel(self.SIFT_Matches)
        self.label_3.setObjectName("label_3")
        self.Info_Container.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.Assumed_Blur = QtWidgets.QLabel(self.SIFT_Matches)
        self.Assumed_Blur.setText("")
        self.Assumed_Blur.setObjectName("Assumed_Blur")
        self.Info_Container.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.Assumed_Blur)
        self.label_4 = QtWidgets.QLabel(self.SIFT_Matches)
        self.label_4.setObjectName("label_4")
        self.Info_Container.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.Verdict = QtWidgets.QLabel(self.SIFT_Matches)
        self.Verdict.setText("")
        self.Verdict.setObjectName("Verdict")
        self.Info_Container.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.Verdict)
        self.verticalLayout.addLayout(self.Info_Container)
        self.SIFT.addTab(self.SIFT_Matches, "")
        self.Gaussian = QtWidgets.QWidget()
        self.Gaussian.setMaximumSize(QtCore.QSize(16777215, 500))
        self.Gaussian.setObjectName("Gaussian")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.Gaussian)
        self.verticalLayout_9.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.Show_Gaussian_Images = QtWidgets.QVBoxLayout()
        self.Show_Gaussian_Images.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.Show_Gaussian_Images.setObjectName("Show_Gaussian_Images")
        self.verticalLayout_9.addLayout(self.Show_Gaussian_Images)
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setObjectName("formLayout_4")
        self.label = QtWidgets.QLabel(self.Gaussian)
        self.label.setObjectName("label")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.G_Scale_Count = QtWidgets.QLabel(self.Gaussian)
        self.G_Scale_Count.setText("")
        self.G_Scale_Count.setObjectName("G_Scale_Count")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.G_Scale_Count)
        self.label_5 = QtWidgets.QLabel(self.Gaussian)
        self.label_5.setObjectName("label_5")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.G_Octaves = QtWidgets.QLabel(self.Gaussian)
        self.G_Octaves.setText("")
        self.G_Octaves.setObjectName("G_Octaves")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.G_Octaves)
        self.verticalLayout_9.addLayout(self.formLayout_4)
        self.SIFT.addTab(self.Gaussian, "")
        self.DoG = QtWidgets.QWidget()
        self.DoG.setObjectName("DoG")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.DoG)
        self.verticalLayout_5.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.Show_DOG_Image = QtWidgets.QVBoxLayout()
        self.Show_DOG_Image.setObjectName("Show_DOG_Image")
        self.verticalLayout_5.addLayout(self.Show_DOG_Image)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_7 = QtWidgets.QLabel(self.DoG)
        self.label_7.setObjectName("label_7")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.label_9 = QtWidgets.QLabel(self.DoG)
        self.label_9.setObjectName("label_9")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.DOG_Img_Count = QtWidgets.QLabel(self.DoG)
        self.DOG_Img_Count.setText("")
        self.DOG_Img_Count.setObjectName("DOG_Img_Count")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.DOG_Img_Count)
        self.DOG_Octave_Count = QtWidgets.QLabel(self.DoG)
        self.DOG_Octave_Count.setText("")
        self.DOG_Octave_Count.setObjectName("DOG_Octave_Count")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.DOG_Octave_Count)
        self.verticalLayout_5.addLayout(self.formLayout_2)
        self.SIFT.addTab(self.DoG, "")
        self.Algorithm_Layout.addWidget(self.SIFT, 0, 0, 1, 1)
        self.Minutiae = QtWidgets.QTabWidget(self.centralwidget)
        self.Minutiae.setObjectName("Minutiae")
        self.Minutiae_Matches = QtWidgets.QWidget()
        self.Minutiae_Matches.setObjectName("Minutiae_Matches")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.Minutiae_Matches)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.min_matches_layout = QtWidgets.QVBoxLayout()
        self.min_matches_layout.setObjectName("min_matches_layout")
        self.verticalLayout_3.addLayout(self.min_matches_layout)
        self.min_match_info = QtWidgets.QFormLayout()
        self.min_match_info.setObjectName("min_match_info")
        self.min_match_score = QtWidgets.QLabel(self.Minutiae_Matches)
        self.min_match_score.setObjectName("min_match_score")
        self.min_match_info.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.min_match_score)
        self.min_score_value = QtWidgets.QLabel(self.Minutiae_Matches)
        self.min_score_value.setText("")
        self.min_score_value.setObjectName("min_score_value")
        self.min_match_info.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.min_score_value)
        self.label_8 = QtWidgets.QLabel(self.Minutiae_Matches)
        self.label_8.setObjectName("label_8")
        self.min_match_info.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.minutiae_min_match = QtWidgets.QLabel(self.Minutiae_Matches)
        self.minutiae_min_match.setObjectName("minutiae_min_match")
        self.min_match_info.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.minutiae_min_match)
        self.label_11 = QtWidgets.QLabel(self.Minutiae_Matches)
        self.label_11.setObjectName("label_11")
        self.min_match_info.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.minutiae_terminations = QtWidgets.QLabel(self.Minutiae_Matches)
        self.minutiae_terminations.setText("")
        self.minutiae_terminations.setObjectName("minutiae_terminations")
        self.min_match_info.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.minutiae_terminations)
        self.label_13 = QtWidgets.QLabel(self.Minutiae_Matches)
        self.label_13.setObjectName("label_13")
        self.min_match_info.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.minutiae_bifurcations = QtWidgets.QLabel(self.Minutiae_Matches)
        self.minutiae_bifurcations.setText("")
        self.minutiae_bifurcations.setObjectName("minutiae_bifurcations")
        self.min_match_info.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.minutiae_bifurcations)
        self.label_15 = QtWidgets.QLabel(self.Minutiae_Matches)
        self.label_15.setObjectName("label_15")
        self.min_match_info.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.minutiae_verdict = QtWidgets.QLabel(self.Minutiae_Matches)
        self.minutiae_verdict.setText("")
        self.minutiae_verdict.setObjectName("minutiae_verdict")
        self.min_match_info.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.minutiae_verdict)
        self.verticalLayout_3.addLayout(self.min_match_info)
        self.Minutiae.addTab(self.Minutiae_Matches, "")
        self.enhanced_image = QtWidgets.QWidget()
        self.enhanced_image.setObjectName("enhanced_image")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.enhanced_image)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.enhance_layout = QtWidgets.QVBoxLayout()
        self.enhance_layout.setObjectName("enhance_layout")
        self.verticalLayout_12.addLayout(self.enhance_layout)
        self.formLayout_7 = QtWidgets.QFormLayout()
        self.formLayout_7.setObjectName("formLayout_7")
        self.label_12 = QtWidgets.QLabel(self.enhanced_image)
        self.label_12.setObjectName("label_12")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.block_or_bin = QtWidgets.QLabel(self.enhanced_image)
        self.block_or_bin.setText("")
        self.block_or_bin.setObjectName("block_or_bin")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.block_or_bin)
        self.label_21 = QtWidgets.QLabel(self.enhanced_image)
        self.label_21.setObjectName("label_21")
        self.formLayout_7.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_21)
        self.block_fr_min = QtWidgets.QLabel(self.enhanced_image)
        self.block_fr_min.setText("")
        self.block_fr_min.setObjectName("block_fr_min")
        self.formLayout_7.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.block_fr_min)
        self.label_16 = QtWidgets.QLabel(self.enhanced_image)
        self.label_16.setObjectName("label_16")
        self.formLayout_7.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.thresh_bin_enh = QtWidgets.QLabel(self.enhanced_image)
        self.thresh_bin_enh.setText("")
        self.thresh_bin_enh.setObjectName("thresh_bin_enh")
        self.formLayout_7.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.thresh_bin_enh)
        self.label_18 = QtWidgets.QLabel(self.enhanced_image)
        self.label_18.setObjectName("label_18")
        self.formLayout_7.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.skeleton_en = QtWidgets.QLabel(self.enhanced_image)
        self.skeleton_en.setText("")
        self.skeleton_en.setObjectName("skeleton_en")
        self.formLayout_7.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.skeleton_en)
        self.label_19 = QtWidgets.QLabel(self.enhanced_image)
        self.label_19.setObjectName("label_19")
        self.formLayout_7.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.min_wav_min = QtWidgets.QLabel(self.enhanced_image)
        self.min_wav_min.setText("")
        self.min_wav_min.setObjectName("min_wav_min")
        self.formLayout_7.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.min_wav_min)
        self.label_20 = QtWidgets.QLabel(self.enhanced_image)
        self.label_20.setObjectName("label_20")
        self.formLayout_7.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.max_wav_min = QtWidgets.QLabel(self.enhanced_image)
        self.max_wav_min.setText("")
        self.max_wav_min.setObjectName("max_wav_min")
        self.formLayout_7.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.max_wav_min)
        self.verticalLayout_12.addLayout(self.formLayout_7)
        self.Minutiae.addTab(self.enhanced_image, "")
        self.Normalize = QtWidgets.QWidget()
        self.Normalize.setObjectName("Normalize")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.Normalize)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.norm_layout = QtWidgets.QVBoxLayout()
        self.norm_layout.setObjectName("norm_layout")
        self.verticalLayout_4.addLayout(self.norm_layout)
        self.Minutiae.addTab(self.Normalize, "")
        self.binarized = QtWidgets.QWidget()
        self.binarized.setObjectName("binarized")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.binarized)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.bin_layout = QtWidgets.QVBoxLayout()
        self.bin_layout.setObjectName("bin_layout")
        self.verticalLayout_7.addLayout(self.bin_layout)
        self.formLayout_5 = QtWidgets.QFormLayout()
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_10 = QtWidgets.QLabel(self.binarized)
        self.label_10.setObjectName("label_10")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.threshold_bin = QtWidgets.QLabel(self.binarized)
        self.threshold_bin.setText("")
        self.threshold_bin.setObjectName("threshold_bin")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.threshold_bin)
        self.label_14 = QtWidgets.QLabel(self.binarized)
        self.label_14.setObjectName("label_14")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.delta_bin = QtWidgets.QLabel(self.binarized)
        self.delta_bin.setText("")
        self.delta_bin.setObjectName("delta_bin")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.delta_bin)
        self.label_17 = QtWidgets.QLabel(self.binarized)
        self.label_17.setObjectName("label_17")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.window_bin = QtWidgets.QLabel(self.binarized)
        self.window_bin.setText("")
        self.window_bin.setObjectName("window_bin")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.window_bin)
        self.verticalLayout_7.addLayout(self.formLayout_5)
        self.Minutiae.addTab(self.binarized, "")
        self.Thinned = QtWidgets.QWidget()
        self.Thinned.setObjectName("Thinned")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.Thinned)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.thin_layout = QtWidgets.QVBoxLayout()
        self.thin_layout.setObjectName("thin_layout")
        self.verticalLayout_10.addLayout(self.thin_layout)
        self.Minutiae.addTab(self.Thinned, "")
        self.Algorithm_Layout.addWidget(self.Minutiae, 0, 1, 1, 1)
        self.gridLayout.addLayout(self.Algorithm_Layout, 0, 0, 1, 1)
        self.ExtraActionsResearch = QtWidgets.QGroupBox(self.centralwidget)
        self.ExtraActionsResearch.setMinimumSize(QtCore.QSize(700, 0))
        self.ExtraActionsResearch.setObjectName("ExtraActionsResearch")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.ExtraActionsResearch)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.run_sift_research = QtWidgets.QPushButton(self.ExtraActionsResearch)
        self.run_sift_research.setEnabled(False)
        self.run_sift_research.setMinimumSize(QtCore.QSize(200, 0))
        self.run_sift_research.setObjectName("run_sift_research")
        self.gridLayout_2.addWidget(self.run_sift_research, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.generate_gaussian_images = QtWidgets.QPushButton(self.ExtraActionsResearch)
        self.generate_gaussian_images.setEnabled(False)
        self.generate_gaussian_images.setMinimumSize(QtCore.QSize(200, 0))
        self.generate_gaussian_images.setObjectName("generate_gaussian_images")
        self.gridLayout_2.addWidget(self.generate_gaussian_images, 1, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.run_sift_performance = QtWidgets.QPushButton(self.ExtraActionsResearch)
        self.run_sift_performance.setEnabled(False)
        self.run_sift_performance.setObjectName("run_sift_performance")
        self.gridLayout_2.addWidget(self.run_sift_performance, 0, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.generate_DOG_images = QtWidgets.QPushButton(self.ExtraActionsResearch)
        self.generate_DOG_images.setEnabled(False)
        self.generate_DOG_images.setMinimumSize(QtCore.QSize(200, 0))
        self.generate_DOG_images.setObjectName("generate_DOG_images")
        self.gridLayout_2.addWidget(self.generate_DOG_images, 2, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.run_minutiae = QtWidgets.QPushButton(self.ExtraActionsResearch)
        self.run_minutiae.setEnabled(False)
        self.run_minutiae.setMinimumSize(QtCore.QSize(200, 0))
        self.run_minutiae.setObjectName("run_minutiae")
        self.gridLayout_2.addWidget(self.run_minutiae, 0, 2, 1, 1, QtCore.Qt.AlignRight)
        self.gridLayout.addWidget(self.ExtraActionsResearch, 4, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.QueryImage = QtWidgets.QPushButton(self.centralwidget)
        self.QueryImage.setMinimumSize(QtCore.QSize(111, 0))
        self.QueryImage.setObjectName("QueryImage")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.QueryImage)
        self.Path_To_Query = QtWidgets.QLabel(self.centralwidget)
        self.Path_To_Query.setObjectName("Path_To_Query")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.Path_To_Query)
        self.TrainImage = QtWidgets.QPushButton(self.centralwidget)
        self.TrainImage.setObjectName("TrainImage")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.TrainImage)
        self.Path_To_Train = QtWidgets.QLabel(self.centralwidget)
        self.Path_To_Train.setObjectName("Path_To_Train")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.Path_To_Train)
        self.gridLayout.addLayout(self.formLayout, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1027, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExport_All_Images = QtWidgets.QAction(MainWindow)
        self.actionExport_All_Images.setObjectName("actionExport_All_Images")
        self.menuFile.addAction(self.actionExport_All_Images)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())

        self.retranslateUi(MainWindow)
        self.SIFT.setCurrentIndex(0)
        self.Minutiae.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Algorithm Comparison"))
        self.label_1.setText(_translate("MainWindow", "Match Score"))
        self.label_6.setText(_translate("MainWindow", "Min Match"))
        self.label_2.setText(_translate("MainWindow", "Sigma"))
        self.label_3.setText(_translate("MainWindow", "Assumed Blur"))
        self.label_4.setText(_translate("MainWindow", "Verdict"))
        self.SIFT.setTabText(self.SIFT.indexOf(self.SIFT_Matches), _translate("MainWindow", "SIFT Matches"))
        self.label.setText(_translate("MainWindow", "Image Count"))
        self.label_5.setText(_translate("MainWindow", "Octaves"))
        self.SIFT.setTabText(self.SIFT.indexOf(self.Gaussian), _translate("MainWindow", "Gaussian Scale Space"))
        self.label_7.setText(_translate("MainWindow", "Number of Images"))
        self.label_9.setText(_translate("MainWindow", "Octaves"))
        self.SIFT.setTabText(self.SIFT.indexOf(self.DoG), _translate("MainWindow", "Difference of Gaussian"))
        self.min_match_score.setText(_translate("MainWindow", "Match Score"))
        self.label_8.setText(_translate("MainWindow", "Min Match"))
        self.minutiae_min_match.setText(_translate("MainWindow", "5"))
        self.label_11.setText(_translate("MainWindow", "Terminations"))
        self.label_13.setText(_translate("MainWindow", "Bifurcations"))
        self.label_15.setText(_translate("MainWindow", "Verdict"))
        self.Minutiae.setTabText(self.Minutiae.indexOf(self.Minutiae_Matches), _translate("MainWindow", "Minutiae Matches"))
        self.label_12.setText(_translate("MainWindow", "Block Orientation"))
        self.label_21.setText(_translate("MainWindow", "Block Frequency"))
        self.label_16.setText(_translate("MainWindow", "Threshold"))
        self.label_18.setText(_translate("MainWindow", "Skeletonized"))
        self.label_19.setText(_translate("MainWindow", "Max Wave Length"))
        self.label_20.setText(_translate("MainWindow", "Min Wave Length"))
        self.Minutiae.setTabText(self.Minutiae.indexOf(self.enhanced_image), _translate("MainWindow", "Enhanced"))
        self.Minutiae.setTabText(self.Minutiae.indexOf(self.Normalize), _translate("MainWindow", "Normalization"))
        self.label_10.setText(_translate("MainWindow", "Threshold"))
        self.label_14.setText(_translate("MainWindow", "Delta"))
        self.label_17.setText(_translate("MainWindow", "Window Size"))
        self.Minutiae.setTabText(self.Minutiae.indexOf(self.binarized), _translate("MainWindow", "Binarization"))
        self.Minutiae.setTabText(self.Minutiae.indexOf(self.Thinned), _translate("MainWindow", "Thinning"))
        self.run_sift_research.setText(_translate("MainWindow", "Run SIFT Algorithm (Research)"))
        self.generate_gaussian_images.setText(_translate("MainWindow", "Generate Gaussian Images"))
        self.run_sift_performance.setText(_translate("MainWindow", "Run SIFT Algorithm (Performance)"))
        self.generate_DOG_images.setText(_translate("MainWindow", "Generate DoG Images"))
        self.run_minutiae.setText(_translate("MainWindow", "Run Minutiae Algorithm"))
        self.QueryImage.setText(_translate("MainWindow", "Select Query Image"))
        self.Path_To_Query.setText(_translate("MainWindow", "Path to query image will show here"))
        self.TrainImage.setText(_translate("MainWindow", "Select Training Image"))
        self.Path_To_Train.setText(_translate("MainWindow", "Path to training image will show here"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuAbout.setTitle(_translate("MainWindow", "About"))
        self.actionExport_All_Images.setText(_translate("MainWindow", "Export All Images"))
