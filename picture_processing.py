# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'picture_processing.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1481, 675)
        MainWindow.setMaximumSize(QtCore.QSize(1481, 675))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("C:/Users/XIEZIYI/Pictures/Saved Pictures/哆啦A梦.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setDockOptions(QtWidgets.QMainWindow.AllowTabbedDocks|QtWidgets.QMainWindow.AnimatedDocks)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 1471, 631))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(9, 7, 101, 591))
        self.label.setMouseTracking(True)
        self.label.setTabletTracking(False)
        self.label.setAcceptDrops(False)
        self.label.setToolTipDuration(-1)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setLineWidth(2)
        self.label.setText("")
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setScaledContents(False)
        self.label.setWordWrap(False)
        self.label.setOpenExternalLinks(False)
        self.label.setObjectName("label")
        self.home_page = QtWidgets.QPushButton(self.frame)
        self.home_page.setGeometry(QtCore.QRect(14, 50, 91, 41))
        self.home_page.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.home_page.setObjectName("home_page")
        self.function = QtWidgets.QPushButton(self.frame)
        self.function.setGeometry(QtCore.QRect(14, 180, 91, 41))
        self.function.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.function.setObjectName("function")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(110, 0, 1351, 621))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(35)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.stackedWidget = QtWidgets.QStackedWidget(self.frame_2)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 0, 1351, 611))
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.label_21 = QtWidgets.QLabel(self.page)
        self.label_21.setGeometry(QtCore.QRect(10, 0, 581, 331))
        self.label_21.setFrameShape(QtWidgets.QFrame.Box)
        self.label_21.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_21.setLineWidth(3)
        self.label_21.setMidLineWidth(5)
        self.label_21.setText("")
        self.label_21.setPixmap(QtGui.QPixmap("C:/Users/XIEZIYI/Pictures/Saved Pictures/b3.jpg"))
        self.label_21.setObjectName("label_21")
        self.textBrowser = QtWidgets.QTextBrowser(self.page)
        self.textBrowser.setGeometry(QtCore.QRect(160, 140, 261, 131))
        self.textBrowser.setObjectName("textBrowser")
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.label_2 = QtWidgets.QLabel(self.page_2)
        self.label_2.setGeometry(QtCore.QRect(10, 15, 1341, 591))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setLineWidth(2)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.page_2)
        self.label_3.setGeometry(QtCore.QRect(28, 40, 231, 561))
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_3.setMidLineWidth(0)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.toolBox = QtWidgets.QToolBox(self.page_2)
        self.toolBox.setGeometry(QtCore.QRect(38, 60, 211, 461))
        self.toolBox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolBox.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.toolBox.setFrameShadow(QtWidgets.QFrame.Plain)
        self.toolBox.setLineWidth(0)
        self.toolBox.setObjectName("toolBox")
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setGeometry(QtCore.QRect(0, 0, 211, 357))
        self.page_3.setObjectName("page_3")
        self.graying = QtWidgets.QPushButton(self.page_3)
        self.graying.setGeometry(QtCore.QRect(0, 66, 211, 41))
        self.graying.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.graying.setObjectName("graying")
        self.toolBox.addItem(self.page_3, "")
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setGeometry(QtCore.QRect(0, 0, 211, 357))
        self.page_4.setObjectName("page_4")
        self.magnify = QtWidgets.QPushButton(self.page_4)
        self.magnify.setGeometry(QtCore.QRect(0, 16, 211, 41))
        self.magnify.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.magnify.setObjectName("magnify")
        self.lessen = QtWidgets.QPushButton(self.page_4)
        self.lessen.setGeometry(QtCore.QRect(0, 78, 211, 41))
        self.lessen.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.lessen.setObjectName("lessen")
        self.rotate1 = QtWidgets.QPushButton(self.page_4)
        self.rotate1.setGeometry(QtCore.QRect(0, 150, 211, 41))
        self.rotate1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.rotate1.setObjectName("rotate1")
        self.toolBox.addItem(self.page_4, "")
        self.page_6 = QtWidgets.QWidget()
        self.page_6.setGeometry(QtCore.QRect(0, 0, 211, 357))
        self.page_6.setObjectName("page_6")
        self.gaussian = QtWidgets.QPushButton(self.page_6)
        self.gaussian.setGeometry(QtCore.QRect(0, 18, 211, 41))
        self.gaussian.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.gaussian.setObjectName("gaussian")
        self.mid_value = QtWidgets.QPushButton(self.page_6)
        self.mid_value.setGeometry(QtCore.QRect(0, 68, 211, 41))
        self.mid_value.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.mid_value.setObjectName("mid_value")
        self.edge = QtWidgets.QPushButton(self.page_6)
        self.edge.setGeometry(QtCore.QRect(0, 175, 211, 41))
        self.edge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.edge.setObjectName("edge")
        self.bilateral = QtWidgets.QPushButton(self.page_6)
        self.bilateral.setGeometry(QtCore.QRect(0, 122, 211, 41))
        self.bilateral.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.bilateral.setObjectName("bilateral")
        self.toolBox.addItem(self.page_6, "")
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setGeometry(QtCore.QRect(0, 0, 211, 357))
        self.page_5.setObjectName("page_5")
        self.global_binarization = QtWidgets.QPushButton(self.page_5)
        self.global_binarization.setGeometry(QtCore.QRect(0, 0, 211, 41))
        self.global_binarization.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.global_binarization.setObjectName("global_binarization")
        self.black_hat = QtWidgets.QPushButton(self.page_5)
        self.black_hat.setGeometry(QtCore.QRect(0, 155, 211, 41))
        self.black_hat.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.black_hat.setObjectName("black_hat")
        self.hood = QtWidgets.QPushButton(self.page_5)
        self.hood.setGeometry(QtCore.QRect(0, 205, 211, 41))
        self.hood.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.hood.setObjectName("hood")
        self.close_operation = QtWidgets.QPushButton(self.page_5)
        self.close_operation.setGeometry(QtCore.QRect(0, 103, 211, 41))
        self.close_operation.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_operation.setObjectName("close_operation")
        self.open_operation = QtWidgets.QPushButton(self.page_5)
        self.open_operation.setGeometry(QtCore.QRect(0, 53, 211, 41))
        self.open_operation.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.open_operation.setObjectName("open_operation")
        self.gradient = QtWidgets.QPushButton(self.page_5)
        self.gradient.setGeometry(QtCore.QRect(0, 259, 211, 41))
        self.gradient.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.gradient.setObjectName("gradient")
        self.toolBox.addItem(self.page_5, "")
        self.img = QtWidgets.QLabel(self.page_2)
        self.img.setGeometry(QtCore.QRect(272, 50, 781, 551))
        self.img.setFrameShape(QtWidgets.QFrame.Box)
        self.img.setFrameShadow(QtWidgets.QFrame.Raised)
        self.img.setText("")
        self.img.setObjectName("img")
        self.stackedWidgetsx = QtWidgets.QStackedWidget(self.page_2)
        self.stackedWidgetsx.setGeometry(QtCore.QRect(1070, 50, 251, 471))
        self.stackedWidgetsx.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.stackedWidgetsx.setObjectName("stackedWidgetsx")
        self.page_9 = QtWidgets.QWidget()
        self.page_9.setObjectName("page_9")
        self.stackedWidgetsx.addWidget(self.page_9)
        self.page_7 = QtWidgets.QWidget()
        self.page_7.setObjectName("page_7")
        self.label_4 = QtWidgets.QLabel(self.page_7)
        self.label_4.setGeometry(QtCore.QRect(0, 22, 251, 441))
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setLineWidth(2)
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.recently = QtWidgets.QRadioButton(self.page_7)
        self.recently.setGeometry(QtCore.QRect(5, 50, 241, 71))
        self.recently.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.recently.setObjectName("recently")
        self.zwillng = QtWidgets.QRadioButton(self.page_7)
        self.zwillng.setGeometry(QtCore.QRect(5, 197, 241, 81))
        self.zwillng.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.zwillng.setObjectName("zwillng")
        self.stackedWidgetsx.addWidget(self.page_7)
        self.page_8 = QtWidgets.QWidget()
        self.page_8.setObjectName("page_8")
        self.label_6 = QtWidgets.QLabel(self.page_8)
        self.label_6.setGeometry(QtCore.QRect(0, 21, 251, 441))
        self.label_6.setFrameShape(QtWidgets.QFrame.Box)
        self.label_6.setLineWidth(2)
        self.label_6.setMidLineWidth(0)
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.recently2 = QtWidgets.QRadioButton(self.page_8)
        self.recently2.setGeometry(QtCore.QRect(5, 40, 241, 71))
        self.recently2.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.recently2.setObjectName("recently2")
        self.zwillng2 = QtWidgets.QRadioButton(self.page_8)
        self.zwillng2.setGeometry(QtCore.QRect(5, 200, 241, 71))
        self.zwillng2.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.zwillng2.setObjectName("zwillng2")
        self.stackedWidgetsx.addWidget(self.page_8)
        self.page_10 = QtWidgets.QWidget()
        self.page_10.setObjectName("page_10")
        self.label_7 = QtWidgets.QLabel(self.page_10)
        self.label_7.setGeometry(QtCore.QRect(10, 20, 241, 441))
        self.label_7.setFrameShape(QtWidgets.QFrame.Box)
        self.label_7.setLineWidth(2)
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.clockwise90 = QtWidgets.QRadioButton(self.page_10)
        self.clockwise90.setGeometry(QtCore.QRect(15, 30, 231, 61))
        self.clockwise90.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.clockwise90.setObjectName("clockwise90")
        self.anticlockwise90 = QtWidgets.QRadioButton(self.page_10)
        self.anticlockwise90.setGeometry(QtCore.QRect(16, 150, 231, 61))
        self.anticlockwise90.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.anticlockwise90.setObjectName("anticlockwise90")
        self.spin180 = QtWidgets.QRadioButton(self.page_10)
        self.spin180.setGeometry(QtCore.QRect(15, 250, 231, 61))
        self.spin180.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.spin180.setObjectName("spin180")
        self.stackedWidgetsx.addWidget(self.page_10)
        self.page_11 = QtWidgets.QWidget()
        self.page_11.setObjectName("page_11")
        self.label_8 = QtWidgets.QLabel(self.page_11)
        self.label_8.setGeometry(QtCore.QRect(0, 20, 251, 441))
        self.label_8.setFrameShape(QtWidgets.QFrame.Box)
        self.label_8.setLineWidth(2)
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.sobel = QtWidgets.QRadioButton(self.page_11)
        self.sobel.setGeometry(QtCore.QRect(4, 60, 241, 41))
        self.sobel.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.sobel.setObjectName("sobel")
        self.scharr = QtWidgets.QRadioButton(self.page_11)
        self.scharr.setGeometry(QtCore.QRect(4, 155, 241, 41))
        self.scharr.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.scharr.setObjectName("scharr")
        self.laplacian = QtWidgets.QRadioButton(self.page_11)
        self.laplacian.setGeometry(QtCore.QRect(5, 237, 241, 41))
        self.laplacian.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.laplacian.setObjectName("laplacian")
        self.stackedWidgetsx.addWidget(self.page_11)
        self.page_12 = QtWidgets.QWidget()
        self.page_12.setObjectName("page_12")
        self.label_9 = QtWidgets.QLabel(self.page_12)
        self.label_9.setGeometry(QtCore.QRect(0, 20, 251, 451))
        self.label_9.setFrameShape(QtWidgets.QFrame.Box)
        self.label_9.setLineWidth(2)
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.page_12)
        self.label_10.setGeometry(QtCore.QRect(10, 140, 221, 141))
        self.label_10.setObjectName("label_10")
        self.stackedWidgetsx.addWidget(self.page_12)
        self.page_13 = QtWidgets.QWidget()
        self.page_13.setObjectName("page_13")
        self.label_11 = QtWidgets.QLabel(self.page_13)
        self.label_11.setGeometry(QtCore.QRect(0, 20, 251, 441))
        self.label_11.setFrameShape(QtWidgets.QFrame.Box)
        self.label_11.setLineWidth(2)
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.page_13)
        self.label_12.setGeometry(QtCore.QRect(13, 161, 221, 121))
        self.label_12.setObjectName("label_12")
        self.stackedWidgetsx.addWidget(self.page_13)
        self.page_14 = QtWidgets.QWidget()
        self.page_14.setObjectName("page_14")
        self.label_13 = QtWidgets.QLabel(self.page_14)
        self.label_13.setGeometry(QtCore.QRect(10, 20, 241, 451))
        self.label_13.setFrameShape(QtWidgets.QFrame.Box)
        self.label_13.setLineWidth(2)
        self.label_13.setText("")
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.page_14)
        self.label_14.setGeometry(QtCore.QRect(20, 180, 211, 111))
        self.label_14.setObjectName("label_14")
        self.stackedWidgetsx.addWidget(self.page_14)
        self.page_15 = QtWidgets.QWidget()
        self.page_15.setObjectName("page_15")
        self.label_15 = QtWidgets.QLabel(self.page_15)
        self.label_15.setGeometry(QtCore.QRect(10, 30, 241, 441))
        self.label_15.setFrameShape(QtWidgets.QFrame.Box)
        self.label_15.setLineWidth(2)
        self.label_15.setText("")
        self.label_15.setObjectName("label_15")
        self.bimary = QtWidgets.QRadioButton(self.page_15)
        self.bimary.setGeometry(QtCore.QRect(15, 60, 231, 51))
        self.bimary.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.bimary.setObjectName("bimary")
        self.binary_inv = QtWidgets.QRadioButton(self.page_15)
        self.binary_inv.setGeometry(QtCore.QRect(15, 121, 231, 51))
        self.binary_inv.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.binary_inv.setObjectName("binary_inv")
        self.trunc = QtWidgets.QRadioButton(self.page_15)
        self.trunc.setGeometry(QtCore.QRect(16, 185, 231, 51))
        self.trunc.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.trunc.setObjectName("trunc")
        self.tozero = QtWidgets.QRadioButton(self.page_15)
        self.tozero.setGeometry(QtCore.QRect(15, 246, 231, 51))
        self.tozero.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.tozero.setObjectName("tozero")
        self.tozero_inv = QtWidgets.QRadioButton(self.page_15)
        self.tozero_inv.setGeometry(QtCore.QRect(15, 312, 231, 51))
        self.tozero_inv.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.tozero_inv.setObjectName("tozero_inv")
        self.stackedWidgetsx.addWidget(self.page_15)
        self.page_16 = QtWidgets.QWidget()
        self.page_16.setObjectName("page_16")
        self.label_16 = QtWidgets.QLabel(self.page_16)
        self.label_16.setGeometry(QtCore.QRect(0, 20, 251, 451))
        self.label_16.setFrameShape(QtWidgets.QFrame.Box)
        self.label_16.setLineWidth(2)
        self.label_16.setText("")
        self.label_16.setObjectName("label_16")
        self.open = QtWidgets.QRadioButton(self.page_16)
        self.open.setGeometry(QtCore.QRect(20, 160, 221, 121))
        self.open.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.open.setObjectName("open")
        self.stackedWidgetsx.addWidget(self.page_16)
        self.page_17 = QtWidgets.QWidget()
        self.page_17.setObjectName("page_17")
        self.label_18 = QtWidgets.QLabel(self.page_17)
        self.label_18.setGeometry(QtCore.QRect(0, 20, 251, 451))
        self.label_18.setFrameShape(QtWidgets.QFrame.Box)
        self.label_18.setLineWidth(2)
        self.label_18.setText("")
        self.label_18.setObjectName("label_18")
        self.close = QtWidgets.QRadioButton(self.page_17)
        self.close.setGeometry(QtCore.QRect(20, 170, 221, 91))
        self.close.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.close.setObjectName("close")
        self.stackedWidgetsx.addWidget(self.page_17)
        self.page_18 = QtWidgets.QWidget()
        self.page_18.setObjectName("page_18")
        self.label_17 = QtWidgets.QLabel(self.page_18)
        self.label_17.setGeometry(QtCore.QRect(0, 20, 251, 451))
        self.label_17.setFrameShape(QtWidgets.QFrame.Box)
        self.label_17.setLineWidth(2)
        self.label_17.setText("")
        self.label_17.setObjectName("label_17")
        self.blackhat = QtWidgets.QRadioButton(self.page_18)
        self.blackhat.setGeometry(QtCore.QRect(10, 180, 231, 101))
        self.blackhat.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.blackhat.setObjectName("blackhat")
        self.stackedWidgetsx.addWidget(self.page_18)
        self.page_19 = QtWidgets.QWidget()
        self.page_19.setObjectName("page_19")
        self.label_19 = QtWidgets.QLabel(self.page_19)
        self.label_19.setGeometry(QtCore.QRect(10, 30, 241, 441))
        self.label_19.setFrameShape(QtWidgets.QFrame.Box)
        self.label_19.setLineWidth(2)
        self.label_19.setText("")
        self.label_19.setObjectName("label_19")
        self.tophat = QtWidgets.QRadioButton(self.page_19)
        self.tophat.setGeometry(QtCore.QRect(20, 190, 211, 81))
        self.tophat.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.tophat.setObjectName("tophat")
        self.stackedWidgetsx.addWidget(self.page_19)
        self.page_20 = QtWidgets.QWidget()
        self.page_20.setObjectName("page_20")
        self.label_20 = QtWidgets.QLabel(self.page_20)
        self.label_20.setGeometry(QtCore.QRect(10, 20, 241, 451))
        self.label_20.setFrameShape(QtWidgets.QFrame.Box)
        self.label_20.setLineWidth(2)
        self.label_20.setText("")
        self.label_20.setObjectName("label_20")
        self.gradient1 = QtWidgets.QRadioButton(self.page_20)
        self.gradient1.setGeometry(QtCore.QRect(30, 160, 201, 111))
        self.gradient1.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.gradient1.setObjectName("gradient1")
        self.stackedWidgetsx.addWidget(self.page_20)
        self.label_5 = QtWidgets.QLabel(self.page_2)
        self.label_5.setGeometry(QtCore.QRect(1070, 20, 101, 41))
        self.label_5.setObjectName("label_5")
        self.stackedWidget.addWidget(self.page_2)
        self.horizontalLayout.addWidget(self.frame)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1481, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.QQ = QtWidgets.QMenu(self.menu_2)
        self.QQ.setObjectName("QQ")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.charu = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("C:/Users/XIEZIYI/Pictures/Saved Pictures/加号.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.charu.setIcon(icon1)
        self.charu.setObjectName("charu")
        self.baocun = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("C:/Users/XIEZIYI/Pictures/Saved Pictures/保存.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.baocun.setIcon(icon2)
        self.baocun.setObjectName("baocun")
        self.zh = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("C:/Users/XIEZIYI/Pictures/Saved Pictures/QQ (1).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.zh.setIcon(icon3)
        self.zh.setObjectName("zh")
        self.actionck = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("C:/Users/XIEZIYI/Pictures/Saved Pictures/a1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionck.setIcon(icon4)
        self.actionck.setObjectName("actionck")
        self.menu.addAction(self.charu)
        self.menu.addAction(self.baocun)
        self.QQ.addAction(self.zh)
        self.menu_2.addAction(self.QQ.menuAction())
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        self.toolBox.setCurrentIndex(3)
        self.stackedWidgetsx.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "OpenCV基本图像处理"))
        self.home_page.setText(_translate("MainWindow", "首页"))
        self.function.setText(_translate("MainWindow", "功能"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600; font-style:italic;\">此项目基于用OpenCV做一些基本图像处理，灰度化，图片旋转，边缘检测，高斯滤波，绘制轮廓,图像放大没用</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600; font-style:italic;\">等...</span></p></body></html>"))
        self.graying.setText(_translate("MainWindow", "图像灰度化"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_3), _translate("MainWindow", "图像基础类"))
        self.magnify.setText(_translate("MainWindow", "图像放大"))
        self.lessen.setText(_translate("MainWindow", "图像缩小"))
        self.rotate1.setText(_translate("MainWindow", "图像旋转"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_4), _translate("MainWindow", "图像变换类"))
        self.gaussian.setText(_translate("MainWindow", "高斯滤波"))
        self.mid_value.setText(_translate("MainWindow", "中值滤波"))
        self.edge.setText(_translate("MainWindow", "边缘检测"))
        self.bilateral.setText(_translate("MainWindow", "双边滤波"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_6), _translate("MainWindow", "图像滤波类"))
        self.global_binarization.setText(_translate("MainWindow", "全局二值化"))
        self.black_hat.setText(_translate("MainWindow", "黑帽"))
        self.hood.setText(_translate("MainWindow", "顶帽"))
        self.close_operation.setText(_translate("MainWindow", "闭运算"))
        self.open_operation.setText(_translate("MainWindow", "开运算"))
        self.gradient.setText(_translate("MainWindow", "梯度"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_5), _translate("MainWindow", "图像形态学"))
        self.recently.setText(_translate("MainWindow", "最近邻插值"))
        self.zwillng.setText(_translate("MainWindow", "双立方插值"))
        self.recently2.setText(_translate("MainWindow", "最近邻插值"))
        self.zwillng2.setText(_translate("MainWindow", "双立方插值"))
        self.clockwise90.setText(_translate("MainWindow", "顺时针旋转90°"))
        self.anticlockwise90.setText(_translate("MainWindow", "逆时针旋转90°"))
        self.spin180.setText(_translate("MainWindow", "旋转180°"))
        self.sobel.setText(_translate("MainWindow", "索贝尔(Sobel)"))
        self.scharr.setText(_translate("MainWindow", "沙尔(Scharr)"))
        self.laplacian.setText(_translate("MainWindow", "拉普拉斯(Laplacian)"))
        self.label_10.setText(_translate("MainWindow", "无属性呢！"))
        self.label_12.setText(_translate("MainWindow", "无属性呢！"))
        self.label_14.setText(_translate("MainWindow", "无属性呢！"))
        self.bimary.setText(_translate("MainWindow", "THRESH_BINARY"))
        self.binary_inv.setText(_translate("MainWindow", "THRESH_BINARY_INV"))
        self.trunc.setText(_translate("MainWindow", "THRESH_TRUNC"))
        self.tozero.setText(_translate("MainWindow", "THRESH_TOZERO"))
        self.tozero_inv.setText(_translate("MainWindow", "THRESH_TOZERO_INV"))
        self.open.setText(_translate("MainWindow", "MORPH_OPEN"))
        self.close.setText(_translate("MainWindow", "MORPH_CLOSE"))
        self.blackhat.setText(_translate("MainWindow", "MORPH_BLACKHAT"))
        self.tophat.setText(_translate("MainWindow", "MORPH＿TOPHAT"))
        self.gradient1.setText(_translate("MainWindow", "MORPH_GRADIENT"))
        self.label_5.setText(_translate("MainWindow", "属性区"))
        self.menu.setTitle(_translate("MainWindow", "图片"))
        self.menu_2.setTitle(_translate("MainWindow", "帮助"))
        self.QQ.setTitle(_translate("MainWindow", "QQ"))
        self.charu.setText(_translate("MainWindow", "插入"))
        self.baocun.setText(_translate("MainWindow", "保存"))
        self.zh.setText(_translate("MainWindow", "2959979833"))
        self.actionck.setText(_translate("MainWindow", "ck"))