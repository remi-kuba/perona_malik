import sys
import numpy as np
from PIL import Image
import io
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, 
                               QPushButton, QFileDialog, QLabel, QSizePolicy,
                               QLineEdit)
from PySide6.QtGui import (QPixmap, QImage)
from PySide6.QtCore import (Qt, QBuffer)
import qdarktheme

# FOR TESTING
import matplotlib.pyplot as plt


def pm_func1(grad_u, k):
    return np.exp(-(np.power(grad_u, 2)) / (np.power(k, 2)))


def pm_func2(grad_u, k):
    return 1 / (1 + np.power((grad_u / k), 2))

class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Perona Malik Diffusion")
        self.setMinimumSize(600, 400)

        self.layout = QVBoxLayout()

        # Window for the image to load and show
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Upload image option
        self.upload_btn = QPushButton("Upload New Image")
        # When moving the window size, button should stretch horizontally
        self.upload_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.upload_btn.clicked.connect(self.upload_image)

        # Perona Malik diffusion button
        self.diffusion_btn = QPushButton("Diffuse Step")
        self.diffusion_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.diffusion_btn.clicked.connect(self.perona_malik_setup)

        # Perona Malik Steps input
        self.steps_input = QLineEdit(self)
        self.steps_input.setPlaceholderText("Enter the number of steps")

        # Reset button
        self.reset_btn = QPushButton("Reset Image")
        self.reset_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.reset_btn.clicked.connect(self.reset_image)


        # Add widgets
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.upload_btn)
        self.layout.addWidget(self.steps_input)
        self.layout.addWidget(self.diffusion_btn)
        self.layout.addWidget(self.reset_btn)
        self.setLayout(self.layout)

        # Default image
        self.remember_img = None
        self.file_path = "images/balloon.png"
        self.pixmap = QPixmap(self.file_path)

    
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg)"
        )

        if file_path:
            self.file_path = file_path
            self.reset_image()
            

    def set_image(self):
        if self.pixmap:
            new_pixmap = self.pixmap.scaled(
                    self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            self.image_label.setPixmap(new_pixmap)


    def reset_image(self):
        self.remember_img = None # should only remember image over iterations of the same image
        self.pixmap = QPixmap(self.file_path)
        self.set_image()
        

    # add to inherited function
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.set_image()


    def perona_malik_setup(self):
        if self.remember_img is not None:
            scaled_img = self.remember_img
        else:
            image = self.get_image_array()
            self.img_min, self.img_max, scaled_img = self.scale_img(image)
        # image = self.get_image_array()
        # self.img_min, self.img_max, scaled_img = self.scale_img(image)

        num_steps = self.steps_input.text()
        steps = 30 # default
        if num_steps.isdigit():
            steps = int(num_steps)

        scaled_result = self.perona_malik(scaled_img, steps)
        result = self.unscale_img(scaled_result, self.img_min, self.img_max)
        self.remember_img = scaled_result
        self.pixmap = self.get_array_image(result)
        # self.pixmap = self.get_array_image(image)
        self.set_image()


    
    def perona_malik(self, img, iterations = 10, k = 0.1, lmbd = 0.25):
        img_frame = np.zeros(img.shape, img.dtype)
        for _ in range(iterations):
            center_pixels, dn, ds, de, dw = self.compute_gradients(img)
            # discretization: https://acme.byu.edu/00000179-afb2-d74f-a3ff-bfbb15700001/anisotropic-pdf
            img_frame[1:-1, 1:-1] = center_pixels + lmbd * ((pm_func1(dn, k) * dn) + (pm_func1(ds, k) * ds) + (pm_func1(de, k) * de) + (pm_func1(dw, k) * dw))
            img = img_frame
        return img
    

    def compute_gradients(self, img):
        # how to compute gradients:
        # https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/
        
        center_pixels = img[1:-1, 1:-1]
        # North: I(x, y - 1) = I[i-1, j] - I[i, j]
        north = img[:-2, 1:-1] - img[1:-1, 1:-1]

        # South: I(x, y + 1) = I[i+1, j] - I[i, j]
        south = img[2:, 1:-1] - img[1:-1, 1:-1]

        # East: I(x + 1, y) = I[i, j + 1] - I[i, j]
        east = img[1:-1, 2:] - img[1:-1, 1:-1]

        # West: I(x - 1, y) = I[i, j - 1] - I[i, j]
        west = img[1:-1, :-2] - img[1:-1, 1:-1]
            
        return center_pixels, north, south, east, west


    def get_image_array(self):
        q_image = self.pixmap.toImage()
        # Convert QImage to PIL Image
        buffer = q_image.bits().tobytes()
        image = Image.frombytes("RGBA", (q_image.width(), q_image.height()), buffer).convert('L')
        # buffer = QBuffer()
        # buffer.open(QBuffer.ReadWrite)
        # q_image.save(buffer, "PNG")
        # image = Image.open(io.BytesIO(buffer.data())).convert('L')
        return np.array(image)
    
    def get_array_image(self, img):
        # Turn numpy array image back to pixmap
        norm_img = 255 * (img - img.min()) / (img.ptp() + 1e-8)  # avoid divide by zero
        norm_img = norm_img.astype(np.uint8)
        pil_image = Image.fromarray(norm_img, mode='L')

        q_image = QImage(
                pil_image.tobytes(),
                pil_image.width,
                pil_image.height,
                pil_image.width,  # 1 byte per pixel
                QImage.Format_Grayscale8
            )
        return QPixmap.fromImage(q_image)
    
    def scale_img(self, img):
        # push pixel range from 0 to 1 (for stability)
        img_min, img_max = img.min(), img.max()
        scaled_img = (img - img_min) / (img_max - img_min)
        return img_min,img_max, scaled_img
    
    def unscale_img(self, img, img_min, img_max):
        return (img * (img_max - img_min)) + img_min
    




if __name__ == "__main__":
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("auto")
    window = ImageWindow()
    window.show()
    sys.exit(app.exec())
