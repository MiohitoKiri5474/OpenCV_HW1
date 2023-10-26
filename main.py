import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

def main():
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Five Blocks GUI")
    window.resize ( 1000, 1000 )


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
    load_img1_btn = QPushButton ( "Load Image 1" )
    load_img2_btn = QPushButton ( "Load Image 2" )

    # For Block1
    Block1_btn_1_1 = QPushButton ( "1.1 Colors Separation" )
    Block1_btn_1_2 = QPushButton ( "1.2 Color Transformation" )
    Block1_btn_1_3 = QPushButton ( "1.3 Color Extraction" )

    # For Block2
    Block2_btn_2_1 = QPushButton ( "2.1 Gaussian blur" )
    Block2_btn_2_2 = QPushButton ( "2.2 Bilateral blur" )
    Block2_btn_2_3 = QPushButton ( "2.3 Median blur" )

    # For Block3
    Block3_btn_3_1 = QPushButton ( "3.1 Sobel X" )
    Block3_btn_3_2 = QPushButton ( "3.2 Sobel Y" )
    Block3_btn_3_3 = QPushButton ( "3.3 Combination and Threshold" )
    Block3_btn_3_4 = QPushButton ( "3.4 Gradient Angle" )


    # ----------------------------------- #
    # Add BTN into each lable

    # For Block1
    Block1_layout = QVBoxLayout()
    Block1_layout.addWidget ( QLabel ( "1. Image Processing" ) )
    Block1_layout.addWidget ( Block1_btn_1_1 )
    Block1_layout.addWidget ( Block1_btn_1_2 )
    Block1_layout.addWidget ( Block1_btn_1_3 )
    block1.setLayout ( Block1_layout )

    # For Block2
    Block2_layout = QVBoxLayout()
    Block2_layout.addWidget ( QLabel ( "2. Image Smoothing" ) )
    Block2_layout.addWidget ( Block2_btn_2_1 )
    Block2_layout.addWidget ( Block2_btn_2_2 )
    Block2_layout.addWidget ( Block2_btn_2_3 )
    block2.setLayout ( Block2_layout )

    # For Block3
    Block3_layout = QVBoxLayout()
    Block3_layout.addWidget ( QLabel ( "3. Edge Detection" ) )
    Block3_layout.addWidget ( Block3_btn_3_1 )
    Block3_layout.addWidget ( Block3_btn_3_2 )
    Block3_layout.addWidget ( Block3_btn_3_3 )
    Block3_layout.addWidget ( Block3_btn_3_4 )
    block3.setLayout ( Block3_layout )

    # For Block4
    Block4_layout = QVBoxLayout()
    Block4_layout.addWidget ( QLabel ( "4. Transforms" ) )
    block4.setLayout ( Block4_layout )

    # For Block5
    Block5_layout = QVBoxLayout()
    Block5_layout.addWidget ( QLabel ( "5. VGG19" ) )
    block5.setLayout ( Block5_layout )


    # ----------------------------------- #
    # Add the blocks to the layouts

    load_img_btn_over_all.addWidget ( load_img1_btn )
    load_img_btn_over_all.addWidget ( load_img2_btn )
    top_row_layout.addWidget(block1)
    top_row_layout.addWidget(block2)
    bottom_row_layout.addWidget(block3)
    bottom_row_layout.addWidget(block4)
    bottom_row_layout.addWidget(block5)


    # ----------------------------------- #
    # Add the layouts to the main layout

    main_layout.addLayout ( load_img_btn_over_all )
    main_layout.addLayout(top_row_layout)
    main_layout.addLayout(bottom_row_layout)


    # ----------------------------------- #
    # Set the main layout for the window

    window.setLayout(main_layout)

   
    # ----------------------------------- #
    # Define functions of each btn

    # For load images over all
    def load_img1_btn_clicked():
        print ( "Loaded Image 1" )

    def load_img2_btn_clicked():
        print ( "Loaded Image 2" )

    # For Block 1
    def Block1_btn_1_1_clicked():
        print ( "Color Separation button clicked" )

    def Block1_btn_1_2_clicked():
        print ( "Color Transformation button clicked" )

    def Block1_btn_1_3_clicked():
        print ( "Color Extraction button clicked" )

    # For Block 2
    def Block2_btn_2_1_clicked():
        print ( "Gaussian blur button clicked" )

    def Block2_btn_2_2_clicked():
        print ( "Bilateral blur button clicked" )

    def Block2_btn_2_3_clicked():
        print ( "Median blur button clicked" )

    # For Block 3
    def Block3_btn_3_1_clicked():
        print ( "Sobel X buttom clicked" )

    def Block3_btn_3_2_clicked():
        print ( "Sobel Y buttom clicked" )

    def Block3_btn_3_3_clicked():
        print ( "Combination and Thresold buttom clicked" )
        
    def Block3_btn_3_4_clicked():
        print ( "Gradient Angle buttom clicked" )


    # ----------------------------------- #
    # Connect functions and btns

    # For load images over all
    load_img1_btn.clicked.connect ( load_img1_btn_clicked )
    load_img2_btn.clicked.connect ( load_img2_btn_clicked )

    # For Block 1
    Block1_btn_1_1.clicked.connect ( Block1_btn_1_1_clicked )
    Block1_btn_1_2.clicked.connect ( Block1_btn_1_2_clicked )
    Block1_btn_1_3.clicked.connect ( Block1_btn_1_3_clicked )

    # For Block 2
    Block2_btn_2_1.clicked.connect ( Block2_btn_2_1_clicked )
    Block2_btn_2_2.clicked.connect ( Block2_btn_2_2_clicked )
    Block2_btn_2_3.clicked.connect ( Block2_btn_2_3_clicked )

    # For Block 3
    Block3_btn_3_1.clicked.connect ( Block3_btn_3_1_clicked )
    Block3_btn_3_2.clicked.connect ( Block3_btn_3_2_clicked )
    Block3_btn_3_3.clicked.connect ( Block3_btn_3_3_clicked )
    Block3_btn_3_4.clicked.connect ( Block3_btn_3_4_clicked )


    # ----------------------------------- #
    # Show the window

    window.show()


    sys.exit(app.exec_())

if __name__ == "__main__":
    main()