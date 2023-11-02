import sys
import time

import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QVBoxLayout, QWidget)
from scipy import signal

file_name = None
image1 = None
image1_gray = None
image2 = None


def update_radius(value):
    global window_radius
    window_radius = value


def _GaussianBlur(img):
    kernel = np.array(
        [
            [0.045, 0.122, 0.045],
            [0.122, 0.332, 0.122],
            [0.045, 0.122, 0.045],
        ]
    )

    return signal.convolve2d(img, kernel, mode="same", boundary="symm").astype(np.uint8)


def Sobel ( img, op ):
    blur = cv2.GaussianBlur ( img, ( 3, 3 ), 0 )
    padding = np.zeros ( ( blur.shape[0] + 2, blur.shape[1] + 2 ) )
    padding[1 : -1, 1 : -1] = blur

    for x in range ( img.shape[0] ):
        for y in range ( img.shape[1] ):
            blur[x, y] = abs ( np.sum ( padding[x : x + 3, y : y + 3] * op ) )

    return blur

# ----------------------------------- #
# Define functions of each btn of related


# General
def get_path():
    global file_name
    file_name = QFileDialog.getOpenFileName(None, "open file", ".")[0]


# For load images over all
def load_img1_btn_clicked():
    global image1
    global image1_gray
    get_path()
    image1 = cv2.imread(file_name)
    image1_gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    if image1 is None:
        print("[ERROR]: Image cannot load")
    else:
        print("Loaded Image 1", file_name)


def load_img2_btn_clicked():
    global image2
    get_path()
    image2 = cv2.imread(file_name)
    if image2 is None:
        print("[ERROR]: Image cannot load")
    else:
        print("Loaded Image 2", file_name)


# For Block1
def Block1_btn_1_1_clicked():
    print("Color Separation button clicked")
    if image1 is None:
        print("[ERROR]: Please load image first")
        return

    b, g, r = cv2.split(image1)

    zeros = np.zeros_like(b)

    cv2.imshow("Blue Image", cv2.merge((b, zeros, zeros)))
    cv2.imshow("Green Image", cv2.merge((zeros, g, zeros)))
    cv2.imshow("Red Image", cv2.merge((zeros, zeros, r)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Block1_btn_1_2_clicked():
    print("Color Transformation button clicked")
    if image1 is None:
        print("[ERROR]: Please load image first")
        return

    b, g, r = cv2.split(image1)
    avg = (r + g + b) // 3

    cv2.imshow("OpenCV function", cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY))
    cv2.imshow("Average weighted", cv2.merge((avg, avg, avg)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Block1_btn_1_3_clicked():
    print("Color Extraction button clicked")

    if image1 is None:
        print("[ERROR]: Please load image first")
        return

    hsv_img = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_not(
        cv2.inRange(hsv_img, np.array([15, 25, 25]), np.array([85, 255, 255]))
    )

    cv2.imshow("I1 mask", cv2.bitwise_not(mask))
    cv2.imshow("Extracted Color", cv2.bitwise_and(image1, image1, mask=mask))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# For Block2
def Block2_btn_2_1_clicked():
    print("Gaussian blur button clicked")

    if image1 is None:
        print("[ERROR]: Please load image first")
        return

    cv2.namedWindow("Gaussian Blur")
    cv2.createTrackbar("m:", "Gaussian Blur", 1, 5, update_radius)

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        current_radius = cv2.getTrackbarPos("m:", "Gaussian Blur")
        sz = (2 * current_radius + 1, 2 * current_radius + 1)

        cv2.imshow("Gaussian Blur", cv2.GaussianBlur(image1, sz, 0))


def Block2_btn_2_2_clicked():
    print("Bilateral Filter button clicked")

    if image1 is None:
        print("[ERROR]: Please load image first")
        return

    cv2.namedWindow("Bilateral Filter")
    cv2.createTrackbar("m:", "Bilateral Filter", 1, 5, update_radius)

    img_list = []
    for i in range(6):
        print("Bilateral Filter img producting: ", i)
        img_list.append(cv2.bilateralFilter(image1, 0, 90, 90, 2 * i + 1))
        cv2.imshow(str(i), img_list[i])

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        current_radius = cv2.getTrackbarPos("m:", "Bilateral Filter")
        cv2.imshow("Bilateral Filter", img_list[current_radius])


def Block2_btn_2_3_clicked():
    print("Median Fliter button clicked")

    if image1 is None:
        print("[ERROR]: Please load image first")
        return

    cv2.namedWindow("Median Filter")
    cv2.createTrackbar("m:", "Median Filter", 1, 5, update_radius)

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        current_radius = cv2.getTrackbarPos("m:", "Median Filter")
        cv2.imshow("Median Filter", cv2.medianBlur(image1, 2 * current_radius + 1))


# For Block3
def Block3_btn_3_1_clicked():
    print("Sobel X button clicked")

    if image1_gray is None:
        print("[ERROR]: Please load image first")
        return

    sobel_operator = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    result = Sobel(image1_gray, sobel_operator)

    cv2.imshow("Sobel X", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Block3_btn_3_2_clicked():
    print("Sobel Y button clicked")

    if image1_gray is None:
        print("[ERROR]: Please load image first")
        return

    sobel_operator = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    result = Sobel(image1_gray, sobel_operator)

    cv2.imshow("Sobel Y", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Block3_btn_3_3_clicked():
    print("Combination and Threshold button clicked")

    if image1_gray is None:
        print("[ERROR]: Please load image first")
        return

    result_x = Sobel(image1_gray, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    result_y = Sobel(image1_gray, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))

    result = (result_x**2 + result_y**2) ** 0.5

    result = (
        (result - np.amin(result)) * 255 / (np.amax(result) - np.amin(result))
    ).astype(np.uint8)

    _, threshold = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY)

    cv2.imshow("Combination of Sobel x and Sobel y", result)
    cv2.imshow("Threshold", threshold)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Block3_btn_3_4_clicked():
    print("Gradient Angle button clicked")

    if image1_gray is None:
        print ( "[ERROR]: Please load image first" )
        return

    blur = cv2.GaussianBlur ( image1_gray, ( 3, 3 ), 0 )
    padding = np.zeros ( ( blur.shape[0] + 2, blur.shape[1] + 2 ) )
    padding[1 : -1, 1 : -1] = blur
    gradient_angle = np.zeros_like ( blur, dtype = 'uint16' )

    for x in range ( blur.shape[0] ):
        for y in range ( blur.shape[1] ):
            gradient_angle[x, y] = ( np.degrees ( np.arctan2 ( np.sum ( padding[x : x + 3, y : y + 3] * np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) ), np.sum ( padding[x : x + 3, y : y + 3] * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) ) ) ) + 360 ) % 360


    mask1 = ( ( gradient_angle > 120 ) & ( gradient_angle <= 180 ) ).astype ( np.uint8 ) * 255
    mask2 = ( ( gradient_angle > 210 ) & ( gradient_angle <= 330 ) ).astype ( np.uint8 ) * 255

    cv2.imshow ( '[120, 180]', cv2.bitwise_and ( blur, mask1 ) )
    cv2.imshow ( '[210, 330]', cv2.bitwise_and ( blur, mask2 ) )
 
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    Block2_btn_2_1 = QPushButton("2.1 Gaussian Blur")
    Block2_btn_2_2 = QPushButton("2.2 Bilateral Filter")
    Block2_btn_2_3 = QPushButton("2.3 Median Filter")

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
