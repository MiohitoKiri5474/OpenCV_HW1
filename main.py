import sys

import cv2

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

file_name = None
image1 = None
image2 = None


def main():
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Five Blocks GUI")
    window.resize(1000, 1000)

    # ----------------------------------- #
    # Create a main layout to contain the five blocks
    main_layout = QVBoxLayout()

    # ----------------------------------- #
    # Create two horizontal layouts for organizing the blocks

    load_img_btn_over_all = QHBoxLayout()
    top_row_layout = QHBoxLayout()
    bottom_row_layout = QHBoxLayout()

    # ----------------------------------- #
    # Create five blocks as QLabel widgets

    block1 = QWidget()
    block2 = QWidget()
    block3 = QWidget()
    block4 = QWidget()
    block5 = QWidget()

    # ----------------------------------- #
    # Create BTNs

    # For load images over all
    load_img1_btn = QPushButton("Load Image 1")
    load_img2_btn = QPushButton("Load Image 2")

    # For Block1
    Block1_btn_1_1 = QPushButton("1.1 Colors Separation")
    Block1_btn_1_2 = QPushButton("1.2 Color Transformation")
    Block1_btn_1_3 = QPushButton("1.3 Color Extraction")

    # For Block2
    Block2_btn_2_1 = QPushButton("2.1 Gaussian blur")
    Block2_btn_2_2 = QPushButton("2.2 Bilateral blur")
    Block2_btn_2_3 = QPushButton("2.3 Median blur")

    # For Block3
    Block3_btn_3_1 = QPushButton("3.1 Sobel X")
    Block3_btn_3_2 = QPushButton("3.2 Sobel Y")
    Block3_btn_3_3 = QPushButton("3.3 Combination and Threshold")
    Block3_btn_3_4 = QPushButton("3.4 Gradient Angle")

    # For Block4
    Block4_btn_transforms = QPushButton("4. Transforms")
    Block4_input_rotation = QLineEdit()
    Block4_input_scaling = QLineEdit()
    Block4_input_TX = QLineEdit()
    Block4_input_TY = QLineEdit()

    # For Block5
    Block5_btn_load_img = QPushButton("Load Image")
    Block5_btn_5_1 = QPushButton("5.1 Show Agumented Images")
    Block5_btn_5_2 = QPushButton("5.2 Show Model Structure")
    Block5_btn_5_3 = QPushButton("5.3 Show Acc and Loss")
    Block5_btn_5_4 = QPushButton("5.4 Inference")

    # ----------------------------------- #
    # Add BTN into each lable

    # For Block1
    Block1_layout = QVBoxLayout()
    Block1_layout.addWidget(QLabel("1. Image Processing"))
    Block1_layout.addWidget(Block1_btn_1_1)
    Block1_layout.addWidget(Block1_btn_1_2)
    Block1_layout.addWidget(Block1_btn_1_3)
    block1.setLayout(Block1_layout)

    # For Block2
    Block2_layout = QVBoxLayout()
    Block2_layout.addWidget(QLabel("2. Image Smoothing"))
    Block2_layout.addWidget(Block2_btn_2_1)
    Block2_layout.addWidget(Block2_btn_2_2)
    Block2_layout.addWidget(Block2_btn_2_3)
    block2.setLayout(Block2_layout)

    # For Block3
    Block3_layout = QVBoxLayout()
    Block3_layout.addWidget(QLabel("3. Edge Detection"))
    Block3_layout.addWidget(Block3_btn_3_1)
    Block3_layout.addWidget(Block3_btn_3_2)
    Block3_layout.addWidget(Block3_btn_3_3)
    Block3_layout.addWidget(Block3_btn_3_4)
    block3.setLayout(Block3_layout)

    # For Block4
    Block4_l1 = QWidget()
    Block4_l1_layout = QVBoxLayout()
    Block4_l1_layout.addWidget(QLabel("Rotation (deg):"))
    Block4_l1_layout.addWidget(QLabel("Scaling       :"))
    Block4_l1_layout.addWidget(QLabel("Tx     (pixel):"))
    Block4_l1_layout.addWidget(QLabel("Ty     (pixel):"))
    Block4_l1.setLayout(Block4_l1_layout)

    Block4_l2 = QWidget()
    Block4_l2_layout = QVBoxLayout()
    Block4_l2_layout.addWidget(Block4_input_rotation)
    Block4_l2_layout.addWidget(Block4_input_scaling)
    Block4_l2_layout.addWidget(Block4_input_TX)
    Block4_l2_layout.addWidget(Block4_input_TY)
    Block4_l2.setLayout(Block4_l2_layout)

    Block4_top_layout = QHBoxLayout()
    Block4_top_layout.addWidget(Block4_l1)
    Block4_top_layout.addWidget(Block4_l2)

    Block4_down_layout = QHBoxLayout()
    Block4_down_layout.addWidget(Block4_btn_transforms)

    Block4_layout = QVBoxLayout()
    Block4_layout.addWidget(QLabel("4. Transforms"))
    Block4_layout.addLayout(Block4_top_layout)
    Block4_layout.addLayout(Block4_down_layout)
    block4.setLayout(Block4_layout)

    # For Block5
    Block5_layout = QVBoxLayout()
    Block5_layout.addWidget(QLabel("5. VGG19"))
    Block5_layout.addWidget(Block5_btn_load_img)
    Block5_layout.addWidget(Block5_btn_5_1)
    Block5_layout.addWidget(Block5_btn_5_2)
    Block5_layout.addWidget(Block5_btn_5_3)
    Block5_layout.addWidget(Block5_btn_5_4)
    Block5_layout.addWidget(QLabel("Predict = "))
    block5.setLayout(Block5_layout)

    # ----------------------------------- #
    # Define functions of each btn of related

    def get_path():
        global file_name
        file_name = QFileDialog.getOpenFileName ( None, 'open file', '.' )[0]

    # For load images over all
    def load_img1_btn_clicked():
        global image1
        get_path()
        image1 = cv2.imread ( file_name )
        if image1 is None:
            print ( "[ERROR]: Image cannot load" )
        else:
            print("Loaded Image 1", file_name)

    def load_img2_btn_clicked():
        global image2
        get_path()
        image2 = cv2.imread ( file_name )
        if image2 is None:
            print ( "[ERROR]: Image cannot load" )
        else:
            print("Loaded Image 2", file_name)

    # For Block1
    def Block1_btn_1_1_clicked():
        print("Color Separation button clicked")

    def Block1_btn_1_2_clicked():
        print("Color Transformation button clicked")

    def Block1_btn_1_3_clicked():
        print("Color Extraction button clicked")

    # For Block2
    def Block2_btn_2_1_clicked():
        print("Gaussian blur button clicked")

    def Block2_btn_2_2_clicked():
        print("Bilateral blur button clicked")

    def Block2_btn_2_3_clicked():
        print("Median blur button clicked")

    # For Block3
    def Block3_btn_3_1_clicked():
        print("Sobel X button clicked")

    def Block3_btn_3_2_clicked():
        print("Sobel Y button clicked")

    def Block3_btn_3_3_clicked():
        print("Combination and Thresold button clicked")

    def Block3_btn_3_4_clicked():
        print("Gradient Angle button clicked")

    # For Block4
    def Block4_btn_transforms_clicked():
        print("Transforms button clicked")
        print("Rotation: ", Block4_input_rotation.text())
        print("Scaling: ", Block4_input_scaling.text())
        print("Tx: ", Block4_input_TX.text())
        print("Ty: ", Block4_input_TY.text())

    # For Block5
    def Block5_btn_load_img_clicked():
        print("Load Image button clicked")

    def Block5_btn_5_1_clicked():
        print("Show Agumented Images button clicked")

    def Block5_btn_5_2_clicked():
        print("Show Model Structure button clicked")

    def Block5_btn_5_3_clicked():
        print("Show Acc and Loss button clicked")

    def Block5_btn_5_4_clicked():
        print("Inference button clicked")

    # ----------------------------------- #
    # Connect functions and btns

    # For load images over all
    load_img1_btn.clicked.connect(load_img1_btn_clicked)
    load_img2_btn.clicked.connect(load_img2_btn_clicked)

    # For Block1
    Block1_btn_1_1.clicked.connect(Block1_btn_1_1_clicked)
    Block1_btn_1_2.clicked.connect(Block1_btn_1_2_clicked)
    Block1_btn_1_3.clicked.connect(Block1_btn_1_3_clicked)

    # For Block2
    Block2_btn_2_1.clicked.connect(Block2_btn_2_1_clicked)
    Block2_btn_2_2.clicked.connect(Block2_btn_2_2_clicked)
    Block2_btn_2_3.clicked.connect(Block2_btn_2_3_clicked)

    # For Block3
    Block3_btn_3_1.clicked.connect(Block3_btn_3_1_clicked)
    Block3_btn_3_2.clicked.connect(Block3_btn_3_2_clicked)
    Block3_btn_3_3.clicked.connect(Block3_btn_3_3_clicked)
    Block3_btn_3_4.clicked.connect(Block3_btn_3_4_clicked)

    # For Block4
    Block4_btn_transforms.clicked.connect(Block4_btn_transforms_clicked)

    # For Block5
    Block5_btn_load_img.clicked.connect(Block5_btn_load_img_clicked)
    Block5_btn_5_1.clicked.connect(Block5_btn_5_1_clicked)
    Block5_btn_5_2.clicked.connect(Block5_btn_5_2_clicked)
    Block5_btn_5_3.clicked.connect(Block5_btn_5_3_clicked)
    Block5_btn_5_4.clicked.connect(Block5_btn_5_4_clicked)

    # ----------------------------------- #
    # Add the blocks to the layouts

    load_img_btn_over_all.addWidget(load_img1_btn)
    load_img_btn_over_all.addWidget(load_img2_btn)
    top_row_layout.addWidget(block1)
    top_row_layout.addWidget(block2)
    bottom_row_layout.addWidget(block3)
    bottom_row_layout.addWidget(block4)
    bottom_row_layout.addWidget(block5)

    # ----------------------------------- #
    # Add the layouts to the main layout

    main_layout.addLayout(load_img_btn_over_all)
    main_layout.addLayout(top_row_layout)
    main_layout.addLayout(bottom_row_layout)

    # ----------------------------------- #
    # Set the main layout for the window

    window.setLayout(main_layout)

    # ----------------------------------- #
    # Show the window

    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
