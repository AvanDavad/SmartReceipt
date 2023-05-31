import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget, QGraphicsView, QGraphicsScene
from PyQt5.QtWidgets import QGraphicsEllipseItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import QRectF
from pathlib import Path

import json

class CustomGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.point_radius = 3
        self.point_color = QColor(255, 0, 0)  # Red color
        self.pen = QPen(self.point_color)
        self.points = []

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene_pos = event.scenePos()
            point_item = QGraphicsEllipseItem(scene_pos.x() - self.point_radius,
                                              scene_pos.y() - self.point_radius,
                                              self.point_radius * 2,
                                              self.point_radius * 2)
            point_item.setPen(self.pen)
            point_item.setBrush(self.point_color)
            self.addItem(point_item)
            self.points.append((scene_pos.x(), scene_pos.y()))
            print(f"Clicked point: ({scene_pos.x()}, {scene_pos.y()})")
            if len(self.points) % 2 == 0:
                # Draw a line
                start_point = self.points[-2]
                end_point = self.points[-1]
                line_width = 5
                self.addLine(start_point[0], start_point[1], end_point[0], end_point[1], self.pen)

        super().mousePressEvent(event)

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

        self.scene = CustomGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.view.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.view.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)

        button = QPushButton("Load Image", self)
        layout.addWidget(button)
        button.clicked.connect(self.load_image)

        save_button = QPushButton("Save", self)
        layout.addWidget(save_button)
        save_button.clicked.connect(self.save_points)

    def load_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg)")
        self.img_filename = Path(file)

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

    def save_points(self):

        json_path = self.img_filename.with_suffix(".json")
        with open(str(json_path), "r") as file:
            annot = json.load(file)

        assert len(self.scene.points) % 2 == 0, len(self.scene.points)
        annot["lines"] = self.scene.points

        with open(str(json_path), 'w') as f:
            json.dump(annot, f, indent=4)

        print(f"saved {json_path}")
        sys.exit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())