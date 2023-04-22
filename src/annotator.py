import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import QRectF

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Display and Zoom in PyQt5")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.view = QGraphicsView(self)
        layout.addWidget(self.view)

        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.view.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.view.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)

        button = QPushButton("Load Image", self)
        layout.addWidget(button)
        button.clicked.connect(self.load_image)

    def load_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg)")

        if file:
            pixmap = QPixmap(file)

            # Clear the previous content in the scene
            self.scene.clear()
            
            # Add the QPixmap to the QGraphicsScene
            self.scene.addPixmap(pixmap)
            self.view.setSceneRect(QRectF(pixmap.rect()))  # Convert QRect to QRectF

            # Fit the view to the scene's bounding rectangle
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        # Zoom in and out using the mouse wheel
        zoom_factor = 1.15

        if event.angleDelta().y() > 0:
            self.view.scale(zoom_factor, zoom_factor)
        else:
            self.view.scale(1 / zoom_factor, 1 / zoom_factor)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())