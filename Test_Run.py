import sys
from PyQt5.QtGui import QPixmap, QImage, QPainter, qRgb
import UI.picture_processing
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import cv2
import numpy as np


class Window(UI.picture_processing.Ui_MainWindow, QMainWindow):
    # 初始化
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)

        # 默认页面打开
        self.stackedWidget.setCurrentIndex(0)
        # 设置切换界面
        self.function.clicked.connect(self.qh1)
        self.home_page.clicked.connect(self.qh2)
        self.magnify.clicked.connect(self.amplification)
        # 点击功能时候无属性
        self.function.clicked.connect(self.attribute_free)
        # 当点击缩小显示属性
        self.lessen.clicked.connect(self.lessen1)
        # # 当点击旋转显示属性图
        self.rotate1.clicked.connect(self.spin11)
        # 当点击高斯滤波显示属性图
        self.gaussian.clicked.connect(self.gauss)
        # 当点击中值滤波显示属性图
        self.mid_value.clicked.connect(self.mid_value1)
        # 当点击双边滤波显示属性图
        self.mid_value.clicked.connect(self.bilateral1)
        # 当点击边缘检测显示属性图
        self.edge.clicked.connect(self.edge1)
        # 当点击全局二值化显示属性图
        self.global_binarization.clicked.connect(self.binaryzation1)

        # 当点击开运算显示属性图
        self.open_operation.clicked.connect(self.open_operation1)

        # 当点击闭运算显示属性图
        self.close_operation.clicked.connect(self.close_operation1)

        # 当点击黑帽显示属性图
        self.black_hat.clicked.connect(self.blackhat1)

        # 当点击顶帽显示属性图
        self.hood.clicked.connect(self.tophat1)

        # 当点击梯度显示属性图
        self.gradient.clicked.connect(self.gradient2)

        # 定义点击插入放入图片在窗口
        self.charu.triggered.connect(self.cahru1)
        # 定义保存图像
        self.baocun.triggered.connect(self.baocun1)
        # 单击事件灰度化
        self.graying.clicked.connect(self.graying_cv)
        self.img_path = ""  # 通用初始化图像路径

        # 放大功能
        # 最近邻插值
        # self.recently.clicked.connect(self.recently1)
        # self.zwillng.clicked.connect(self.bicubic1)
        # 缩小功能
        self.recently2.clicked.connect(self.lessen_open)
        self.zwillng2.clicked.connect(self.lessen_open1)

        # 旋转单击事件
        self.clockwise90.clicked.connect(self.clockwise1)  # 顺时针90
        self.anticlockwise90.clicked.connect(self.anticlockwise1)  # 逆时针90
        self.spin180.clicked.connect(self.spin)  # 旋转180

        # 高斯滤波单击事件
        # 单击索贝尔
        self.sobel.clicked.connect(self.sobel1)
        # 单击沙尔
        self.scharr.clicked.connect(self.scharr1)
        # 单击拉普拉斯
        self.laplacian.clicked.connect(self.laplacian1)

        # 单击中值滤波
        self.mid_value.clicked.connect(self.mid_value2)

        # 单击双边滤波
        self.bilateral.clicked.connect(self.bilateral2)

        # 单击边缘检测
        self.edge.clicked.connect(self.edge2)

        # 形态学
        # 点击全局二值化
        # 单击THRESH_BINARY
        self.bimary.clicked.connect(self.binary2)
        # 单击THRESH_BINARY_INV
        self.binary_inv.clicked.connect(self.binary_inv2)
        # 单击THRESH_TRUNC
        self.trunc.clicked.connect(self.trunc2)
        # 单击THRESH_TOZERO
        self.tozero.clicked.connect(self.tozero2)
        # 单击THRESH_TOZERO_INV
        self.tozero_inv.clicked.connect(self.tozero_inv2)

        # 单击开运算
        self.open.clicked.connect(self.open_operation2)
        # 单击闭运算
        self.close.clicked.connect(self.closed_operation2)
        # 单击黑帽
        self.blackhat.clicked.connect(self.blackhat2)
        # 单击顶帽
        self.tophat.clicked.connect(self.tophat2)
        # 单击梯度
        self.gradient1.clicked.connect(self.gradient3)

    # 将OpenCV图像转换为Qt图像几乎通用
    def qpixmap_image(self, image):
        # 将OpenCV图像转换为RGB颜色空间
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取图像尺寸和通道数
        height, width, channels = image_rgb.shape

        # 创建QImage对象
        q_image = QImage(image_rgb.data, width, height, width * channels, QImage.Format_RGB888)

        # 创建QPixmap对象并将QImage设置为其数据
        pixmap = QPixmap.fromImage(q_image)

        return pixmap

    def qh1(self):
        self.stackedWidget.setCurrentIndex(1)

    # 首页切换
    def qh2(self):
        self.stackedWidget.setCurrentIndex(0)

    # 点击功能时候无属性
    def attribute_free(self):
        self.stackedWidgetsx.setCurrentIndex(0)

    # 图像放大属性切换
    def amplification(self):
        self.stackedWidgetsx.setCurrentIndex(1)

    # 图像缩小属性切换
    def lessen1(self):
        self.stackedWidgetsx.setCurrentIndex(2)

    # 旋转属性切换
    def spin11(self):
        self.stackedWidgetsx.setCurrentIndex(3)

    # 高斯滤波属性切换
    def gauss(self):
        self.stackedWidgetsx.setCurrentIndex(4)

    # 中值滤波属性切换
    def mid_value1(self):
        self.stackedWidgetsx.setCurrentIndex(5)

    # 双边滤波属性切换
    def bilateral1(self):
        self.stackedWidgetsx.setCurrentIndex(6)

    # 边缘检测属性切换
    def edge1(self):
        self.stackedWidgetsx.setCurrentIndex(7)

    # 全局二值化属性切换
    def binaryzation1(self):
        self.stackedWidgetsx.setCurrentIndex(8)

    # 开运算属性切换
    def open_operation1(self):
        self.stackedWidgetsx.setCurrentIndex(9)

    # 闭运算属性切换
    def close_operation1(self):
        self.stackedWidgetsx.setCurrentIndex(10)

    # 黑帽属性切换
    def blackhat1(self):
        self.stackedWidgetsx.setCurrentIndex(11)

    # 顶帽属性切换
    def tophat1(self):
        self.stackedWidgetsx.setCurrentIndex(12)

    # 梯度属性切换
    def gradient2(self):
        self.stackedWidgetsx.setCurrentIndex(13)

    # 显示选择文件夹
    def openfile(self):
        path, type = QFileDialog.getOpenFileName(self, '选择文件', '', 'image files(*.png , *.jpg)')
        return path

    # 图像插入到label中
    def cahru1(self):
        img_path = self.openfile()  # 获取图像路径
        if img_path:
            self.img_path = img_path  # 保存图像路径
            img = QPixmap(img_path)  # 获取图像映像QPimax对象
            self.img.setPixmap(img)  # 插入label中
            self.img.setScaledContents(True)  # 设置label内部图片自适应填充

    # 保存文件
    def savefile(self):
        path, type = QFileDialog.getSaveFileName(self, '文件保存', '', 'image files(*.png , *.jpg)')

        return path

    # 获取当前插入的图像
    def baocun1(self):
        pixmap = self.img.pixmap()  # 获取当前插入的图像的QPixmap对象
        save_path = self.savefile()  # 获取保存路径
        if save_path:
            pixmap.save(save_path)  # 保存图像到指定的文件夹
            # 弹窗提示保存成功
            QtWidgets.QMessageBox.about(self, '提示', '保存成功')

    # 灰度化处理函数
    def graying_cv(self):
        if self.img_path:  # 如果存在图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像

            img_pixmap = self.cv_pixmap(gray_img)  # 将灰度图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为灰度图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 将OpenCV图像转换为QPixmap对象
    # QPixmap对象
    def cv_pixmap(self, img):
        height, width = img.shape[:2]  # 获取图像的高度和宽度
        qimg = QImage(img.data, width, height, img.strides[0], QImage.Format_Grayscale8)  # 创建灰度图像的QImage对象
        pixmap = QPixmap.fromImage(qimg)  # 创建灰度图像的QPixmap对象
        return pixmap

    # # 图像放大
    # # 最近邻插值
    # def recently1(self):
    #     if self.img_path:
    #         img1 = cv2.imread(self.img_path)  # 使用OpenCV读取图像
    #
    #         height, width, _ = img1.shape
    #         img_resized = cv2.resize(img1, (width * 2, height * 2), interpolation=cv2.INTER_NEAREST)
    #         img_resized = cv2.convertScaleAbs(img_resized)  # 将结果转换为无符号8位整数
    #         pixmap_resized = self.qpixmap_image(img_resized)  # 将缩小后的图像转换为QPixmap对象
    #         self.img.setPixmap(pixmap_resized)  # 设置标签的显示图像为缩小后的图像
    #         self.img.setScaledContents(True)  # 自适应缩放图像到标签大小
    #         self.img.setAlignment(Qt.AlignCenter)  # 图像居中显示
    #
    # # 双立方插值
    # def bicubic1(self):
    #
    #     pixmap = self.img.pixmap()  # 获取当前插入的图像的QPixmap对象
    #     if pixmap:
    #         img_np = self.qpixmap_image(pixmap)  # 将QPixmap对象转换为numpy数组
    #         img_resized = cv2.pyrUp(img_np)
    #         pixmap_resized = self.qpixmap_image(img_resized)  # 将放大后的图像转换为QPixmap对象
    #         self.img.setPixmap(pixmap_resized)  # 设置标签的显示图像为放大后的图像
    #         self.img.setScaledContents(True)  # 自适应缩放图像到标签大小
    #         self.img.setAlignment(Qt.AlignCenter)  # 图像居中显示

    # 图像缩小
    # 最近邻插值
    def lessen_open(self):
        if self.img_path:
            img1 = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            height, width, _ = img1.shape
            img_resized = cv2.resize(img1, (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)
            pixmap_resized = self.qpixmap_image(img_resized)  # 将缩小后的图像转换为QPixmap对象
            self.img.setPixmap(pixmap_resized)  # 设置标签的显示图像为缩小后的图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小
            self.img.setAlignment(Qt.AlignCenter)  # 图像居中显示

    # 双立方插值
    def lessen_open1(self):
        if self.img_path:    # 如果存在图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            height, width, _ = img.shape
            img_resized = cv2.resize(img, (width // 3, height // 3), interpolation=cv2.INTER_AREA)
            pixmap_resized = self.qpixmap_image(img_resized)  # 将缩小后的图像转换为QPixmap对象
            self.img.setPixmap(pixmap_resized)  # 设置标签的显示图像为缩小后的图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小
            self.img.setAlignment(Qt.AlignCenter)  # 图像居中显示

    # 顺时针旋转90
    def clockwise1(self):
        if self.img_path:  # 如果存在图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像
            image1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img_pixmap = self.qpixmap_image(image1)  # 将灰度图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为放大后的图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 逆时针旋转90
    def anticlockwise1(self):
        if self.img_path:  # 如果存在图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像
            image1 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_pixmap = self.qpixmap_image(image1)  # 将灰度图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为放大后的图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 旋转180
    def spin(self):
        if self.img_path:  # 如果存在图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像
            image1 = cv2.rotate(img, cv2.ROTATE_180)
            img_pixmap = self.qpixmap_image(image1)  # 将灰度图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为放大后的图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 高斯滤波

    # 索贝尔算法
    def sobel1(self):
        if self.img_path:  # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像

            # 使用Sobel算子计算边缘
            sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)  # 计算X方向上的边缘
            sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)  # 计算Y方向上的边缘
            sobel_result = np.sqrt(sobel_x + sobel_y)  # 计算边缘的强度

            sobel_result = cv2.convertScaleAbs(sobel_result)  # 将结果转换为无符号8位整数
            img_pixmap = self.qpixmap_image(sobel_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 沙尔算法
    def scharr1(self):
        if self.img_path:  # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
            # 沙尔算子
            scharr_result = cv2.Scharr(gray_img, cv2.CV_64F, 1, 0)
            scharr_result = cv2.convertScaleAbs(scharr_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(scharr_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 拉普拉斯算子
    def laplacian1(self):
        if self.img_path:  # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
            # 沙尔算子
            laplacian_result = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=5)
            laplacian_result = cv2.convertScaleAbs(laplacian_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(laplacian_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 中值滤波
    def mid_value2(self):
        if self.mid_value.clicked.connect(self.mid_value2):
            QtWidgets.QMessageBox.about(self, '提示', '用于对于胡椒噪音效果明显')
        if self.img_path:  # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
            # 中值算法
            median_result = cv2.medianBlur(gray_img, 27)
            median_result = cv2.convertScaleAbs(median_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(median_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 双边滤波
    def bilateral2(self):
        if self.bilateral.clicked.connect(self.bilateral2):
            QtWidgets.QMessageBox.about(self, '提示', '用于对于美颜效果')
        if self.img_path:  # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            # 中值算法
            bilateral_result = cv2.bilateralFilter(img, 39, 30, 30)
            bilateral_result = cv2.convertScaleAbs(bilateral_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(bilateral_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 边缘检测大法
    def edge2(self):
        if self.img_path:  # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
            # Canny算子
            canny_result = cv2.Canny(gray_img, 80, 130)
            canny_result = cv2.convertScaleAbs(canny_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(canny_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 形态学处理 全局二值化
    # BINARY类型
    def binary2(self):
        if self.img_path:  # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
            # 全局二值化属性binary
            retval, threshold_result = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
            threshold_result = cv2.convertScaleAbs(threshold_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(threshold_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # BINARY_INV
    def binary_inv2(self):
        if self.img_path:
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
            # 全局二值化属性binary_inv
            retval, threshold_result = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)
            threshold_result = cv2.convertScaleAbs(threshold_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(threshold_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # THRESH_TRUNC
    def trunc2(self):
        if self.img_path:    # # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
            # 全局二值化属性THRESH_TRUNC
            retval, threshold_result = cv2.threshold(gray_img, 180, 255, cv2.THRESH_TRUNC)
            threshold_result = cv2.convertScaleAbs(threshold_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(threshold_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # THRESH_TOZERO
    def tozero2(self):
        if self.img_path:    # # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
            # 全局二值化属性THRESH_TRUNC
            retval, threshold_result = cv2.threshold(gray_img, 190, 255, cv2.THRESH_TOZERO)
            threshold_result = cv2.convertScaleAbs(threshold_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(threshold_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # THRESH_TOZERO_INV
    def tozero_inv2(self):
        if self.img_path:    # # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
            # 全局二值化属性THRESH_TRUNC
            retval, threshold_result = cv2.threshold(gray_img, 230, 255, cv2.THRESH_TOZERO_INV)
            threshold_result = cv2.convertScaleAbs(threshold_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(threshold_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 开运算
    def open_operation2(self):
        if self.img_path:    # # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像
            # 获取卷积核
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # 矩阵卷积核5x5的
            dilate_result = cv2.morphologyEx(img, cv2.MORPH_OPEN,  kernel)

            threshold_result = cv2.convertScaleAbs(dilate_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(threshold_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 闭运算
    def closed_operation2(self):
        if self.img_path:    # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像
            # 获取卷积核
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩阵卷积核5x5的
            dilate_result = cv2.morphologyEx(img, cv2.MORPH_CLOSE,  kernel)

            threshold_result = cv2.convertScaleAbs(dilate_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(threshold_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 黑帽
    def blackhat2(self):
        if self.img_path:    # # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像
            # 获取卷积核
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩阵卷积核5x5的
            dilate_result = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT,  kernel)

            threshold_result = cv2.convertScaleAbs(dilate_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(threshold_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 顶帽
    def tophat2(self):
        if self.img_path:  # # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像
            # 获取卷积核
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩阵卷积核5x5的
            dilate_result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

            threshold_result = cv2.convertScaleAbs(dilate_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(threshold_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 梯度
    def gradient3(self):
        if self.img_path:  # # 如果存在图像路径
            print(self.img_path)  # 打印图像路径
            img = cv2.imread(self.img_path)  # 使用OpenCV读取图像
            # 获取卷积核
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩阵卷积核5x5的
            dilate_result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

            threshold_result = cv2.convertScaleAbs(dilate_result)  # 将结果转换为无符号8位整数

            img_pixmap = self.qpixmap_image(threshold_result)  # 将边缘图像转换为QPixmap对象
            self.img.setPixmap(img_pixmap)  # 设置标签的显示图像为边缘图像
            self.img.setScaledContents(True)  # 自适应缩放图像到标签大小

    # 结束
    def finish(self):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = Window()
    main_window.show()

    # 进入程序的主循环，并通过exit函数确保住循环安全结束
    sys.exit(app.exec())
