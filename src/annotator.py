import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Display in PyQt5")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        button = QPushButton("Load Image", self)
        layout.addWidget(button)
        button.clicked.connect(self.load_image)

    def load_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg)")

        if file:
            pixmap = QPixmap(file)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            self.image_label.adjustSize()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())