import os

current_dir = os.path.dirname(__file__)
vocab_file_path = os.path.join(current_dir, 'model/clip/bpe_simple_vocab_16e6.txt.gz')
model_file_path = os.path.join(current_dir, 'model/clip/ViT-B-32.pt')



import sys
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QSlider, QPushButton, QLabel, QTextEdit, QScrollArea, QSpinBox, QFileDialog)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap, QPainter, QBrush, QColor
from PIL import Image
import torch
import clip
import os
import subprocess
import platform
from PyQt6.QtGui import QIcon
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
class Spinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.setFixedSize(100, 100)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)  

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(255, 255, 255, 100)))
        painter.setPen(Qt.PenStyle.NoPen)

        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) * 0.8 / 2 
        num_segments = 12  
        angle_step = 360 / num_segments

        for i in range(num_segments):
            angle = angle_step * i + self.angle
            radian = math.radians(angle)
            x = center.x() + radius * math.cos(radian)
            y = center.y() - radius * math.sin(radian)
            painter.drawEllipse(QPoint(x, y), 5, 5)  

        painter.end()
        self.angle = (self.angle + 3) % 360 

def load_clip_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'ViT-B/32'
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device


model, preprocess, device = load_clip_model()
def calculate_similarity_score(image_path, description):
    text_inputs = clip.tokenize([description]).to(device)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        description_features = model.encode_text(text_inputs)
        image_features = model.encode_image(image)
        similarity = torch.cosine_similarity(description_features, image_features)

    return similarity.item()


def process_image_batch(batch, description_features):
    results = []
    for file_path in batch:
        try:
            image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            similarity = torch.cosine_similarity(description_features, image_features)
            results.append((file_path, similarity.item()))
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
    return results

def search_images_by_description(description, folders, batch_size=10, max_threads=2):
    results = []
    text_inputs = clip.tokenize([description]).to(device)
    
    try:
        with torch.no_grad():
            description_features = model.encode_text(text_inputs)
    except Exception as e:
        print(f"Error encoding description: {e}")
        return results

    def chunkify(lst, n):
        """Divide list into chunks of size n."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    all_image_paths = []
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"Directory not found: {folder}")
            continue
        for file_name in os.listdir(folder):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(folder, file_name)
                all_image_paths.append(file_path)


    with ThreadPoolExecutor(max_threads) as executor:
        future_to_batch = {executor.submit(process_image_batch, batch, description_features): batch for batch in chunkify(all_image_paths, batch_size)}
        for future in as_completed(future_to_batch):
            batch_results = future.result()
            if batch_results:
                results.extend(batch_results)

    results.sort(key=lambda x: x[1], reverse=True)
    

    torch.cuda.empty_cache()
    gc.collect()

    return results

def search_images_by_image(uploaded_image_path, folders, batch_size=10, max_threads=2):
    results = []
    image1 = preprocess(Image.open(uploaded_image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image1_features = model.encode_image(image1)

    def process_image_with_features(file_path):
        try:
            image2 = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
            image2_features = model.encode_image(image2)
            similarity = torch.cosine_similarity(image1_features, image2_features)
            return file_path, similarity.item()
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            return None

    all_image_paths = []
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"Directory not found: {folder}")
            continue
        for file_name in os.listdir(folder):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(folder, file_name)
                all_image_paths.append(file_path)


    with ThreadPoolExecutor(max_threads) as executor:
        future_to_file = {executor.submit(process_image_with_features, file_path): file_path for file_path in all_image_paths}
        for future in as_completed(future_to_file):
            result = future.result()
            if result:
                results.append(result)

    results.sort(key=lambda x: x[1], reverse=True)
    
    torch.cuda.empty_cache()
    gc.collect()

    return results

def search_images_combined(uploaded_image_path, description, folders, batch_size=10, max_threads=4):
    results = []
    image1 = preprocess(Image.open(uploaded_image_path)).unsqueeze(0).to(device)
    text_inputs = clip.tokenize([description]).to(device)

    with torch.no_grad():
        image1_features = model.encode_image(image1)
        description_features = model.encode_text(text_inputs)

    def process_combined_image(file_path):
        try:
            image2 = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
            image2_features = model.encode_image(image2)
            image_similarity = torch.cosine_similarity(image1_features, image2_features)
            text_similarity = torch.cosine_similarity(description_features, image2_features)
            combined_similarity = (image_similarity + text_similarity) / 2
            return file_path, combined_similarity.item()
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            return None

    all_image_paths = []
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"Directory not found: {folder}")
            continue
        for file_name in os.listdir(folder):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(folder, file_name)
                all_image_paths.append(file_path)

    with ThreadPoolExecutor(max_threads) as executor:
        future_to_file = {executor.submit(process_combined_image, file_path): file_path for file_path in all_image_paths}
        for future in as_completed(future_to_file):
            result = future.result()
            if result:
                results.append(result)

    results.sort(key=lambda x: x[1], reverse=True)
    

    torch.cuda.empty_cache()
    gc.collect()

    return results

class ImageSearchThread(QThread):
    results_ready = pyqtSignal(list)

    def __init__(self, image_path, description, folders):
        super().__init__()
        self.image_path = image_path
        self.description = description
        self.folders = folders

    def run(self):
        if self.image_path and self.description:
            results = search_images_combined(self.image_path, self.description, self.folders)
        elif self.image_path:
            results = search_images_by_image(self.image_path, self.folders)
        elif self.description:
            results = search_images_by_description(self.description, self.folders)
        else:
            results = []

        self.results_ready.emit(results)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Findmage')
        self.setGeometry(100, 100, 1000, 800)
        self.setWindowIcon(QIcon('icon.png'))  
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.container = QWidget()
        self.container_layout = QVBoxLayout()
        self.container.setLayout(self.container_layout)
        self.container.setStyleSheet('background-color: #2d2d2d; border-radius: 10px;')

        self.layout.addWidget(self.container)

        self.header_layout = QHBoxLayout()
        self.header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label = QLabel()
        self.logo_label.setPixmap(QPixmap('icon.png').scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio))
        self.header_label = QLabel(' Findmage')
        self.header_label.setStyleSheet('font-size: 28px; color: #ffffff; font-weight: bold;')
        self.header_layout.addWidget(self.logo_label)
        self.header_layout.addWidget(self.header_label)
        self.container_layout.addLayout(self.header_layout)

        self.form_layout = QVBoxLayout()
        self.form_layout.setContentsMargins(20, 20, 20, 20)
        self.form_layout.setSpacing(10)
        self.container_layout.addLayout(self.form_layout)

        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        self.file_input.setStyleSheet('background-color: #3c3c3c; color: #dcdcdc; padding: 10px; '
                                     'border-radius: 5px; border: 2px solid #444;')
        self.file_button = QPushButton('Browse')
        self.file_button.setStyleSheet('background-color: #007acc; color: #ffffff; padding: 10px; '
                                       'border-radius: 5px; border: none;')
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.file_button)
        self.form_layout.addLayout(file_layout)

        description_layout = QHBoxLayout()
        self.description_input = QLineEdit()
        self.description_input.setPlaceholderText('Enter description...')
        self.description_input.setStyleSheet('background-color: #3c3c3c; color: #dcdcdc; padding: 10px; '
                                             'border-radius: 5px; border: 2px solid #444;')
        self.max_images_input = QSpinBox()
        self.max_images_input.setStyleSheet('background-color: #3c3c3c; color: #dcdcdc; padding: 10px; '
                                            'border-radius: 5px; border: 2px solid #444;')
        self.max_images_input.setValue(10)
        self.max_images_input.setMinimum(1)
        self.max_images_input.setMaximum(50)

        description_layout.addWidget(self.description_input)
        description_layout.addWidget(self.max_images_input)
        self.form_layout.addLayout(description_layout)

        self.slider_label = QLabel('Current Threshold: 0.8')
        self.slider_label.setStyleSheet('color: #a0a0a0; font-size: 14px; margin-top: 10px;')
        self.form_layout.addWidget(self.slider_label)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(20)
        self.slider.setMaximum(99)
        self.slider.setValue(80)
        self.slider.setSingleStep(1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setStyleSheet('QSlider::groove:horizontal { border: 1px solid #444; background: #555; '
                                  'height: 2px; border-radius: 1px; } '
                                  'QSlider::handle:horizontal { background: #007acc; width: 12px; height: 12px; '
                                  'border-radius: 6px; margin: -6px 0; } '
                                  'QSlider::handle:horizontal:hover { background: #005f8c; } '
                                  'QSlider::sub-page:horizontal { background: #0094ff; border-radius: 1px; } '
                                  'QSlider::add-page:horizontal { background: #555; border-radius: 1px; }')
        self.slider.valueChanged.connect(self.update_slider_label)
        self.form_layout.addWidget(self.slider)

        self.submit_button = QPushButton('Search')
        self.submit_button.setStyleSheet('background-color: #007acc; color: #ffffff; padding: 12px 25px; '
                                         'border-radius: 5px; font-weight: 700; border: none;')
        self.submit_button.setFixedWidth(150)
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(self.submit_button)
        self.form_layout.addLayout(button_layout)

        self.result_box = QWidget()
        self.result_box_layout = QVBoxLayout()
        self.result_box.setLayout(self.result_box_layout)
        self.result_box.setStyleSheet('background-color: #2d2d2d; padding: 10px; '
                                      'border-radius: 5px; border: 2px solid #444;')

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet('background-color: #3c3c3c; border: 2px solid #888;')

        
        self.scroll_widget = QWidget()
        self.scroll_widget.setLayout(QVBoxLayout())
        self.scroll_widget.layout().addWidget(self.result_box)
        
        self.scroll_area.setWidget(self.scroll_widget)
        self.form_layout.addWidget(self.scroll_area)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet('background-color: #3c3c3c; color: #dcdcdc; padding: 10px; '
                                     'border-radius: 5px; border: 2px solid #444;')
        self.text_area.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.text_area.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.result_box_layout.addWidget(self.text_area)

        self.clear_label = QLabel('<a href="#">Click here</a> to clear the upload and description text.')
        self.clear_label.setStyleSheet('color: #a0a0a0; font-size: 14px;')
        self.form_layout.addWidget(self.clear_label)

        self.overlay = QWidget(self)
        self.overlay.setStyleSheet('background-color: rgba(0, 0, 0, 0.5);')
        self.overlay.setVisible(False)

        self.spinner = Spinner()
        self.spinner.setStyleSheet('background-color: rgba(0, 0, 0, 0);')

        self.spinner_layout = QVBoxLayout()
        self.spinner_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinner_layout.addWidget(self.spinner)
        self.overlay.setLayout(self.spinner_layout)

        self.resizeEvent = self.update_overlay_geometry


        self.possible_folders = [
            # Windows
            os.path.join(os.path.expanduser('~'), 'Pictures'),
            os.path.join(os.path.expanduser('~'), 'OneDrive', 'Pictures'),
            os.path.join(os.path.expanduser('~'), 'Documents', 'Pictures'),
            os.path.join(os.path.expanduser('~'), 'Downloads'),
            os.path.join(os.path.expanduser('~'), 'Desktop'),
            'D:\\Pictures',  
            'E:\\Photos',   
            
            # macOS
            os.path.join(os.path.expanduser('~'), 'iCloud Drive', 'Photos'),
            os.path.join(os.path.expanduser('~'), 'iCloud Drive', 'Documents', 'Photos'),
            '/Volumes/ExternalDrive/Pictures',  
            
            # Linux
            os.path.join(os.path.expanduser('~'), 'Pictures'),
            os.path.join(os.path.expanduser('~'), 'Downloads'),
            os.path.join(os.path.expanduser('~'), 'Desktop'),
            os.path.join(os.path.expanduser('~'), 'Documents', 'Pictures'),
            '/mnt/ExternalDrive/Pictures',  
            '/media/username/ExternalDrive/Pictures',  
        ]



        self.file_button.clicked.connect(self.open_file_dialog)
        self.submit_button.clicked.connect(self.start_search)
        self.clear_label.linkActivated.connect(self.clear_upload_and_description)

    def update_overlay_geometry(self, event):
        super().resizeEvent(event)
        self.overlay.setGeometry(self.rect())

    def update_slider_label(self, value):
        self.slider_label.setText(f'Current Threshold: {value / 100.0:.2f}')

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if file_path:
            self.file_input.setText(file_path)

    def start_search(self):
        self.overlay.setVisible(True)
        self.spinner.show()

        image_path = self.file_input.text()
        description = self.description_input.text()


        folders = self.possible_folders

        self.search_thread = ImageSearchThread(image_path, description, folders)
        self.search_thread.results_ready.connect(self.display_results)


        self.search_thread.finished.connect(self.on_search_finished)
        
        self.search_thread.start()

    def on_search_finished(self):
        self.overlay.setVisible(False)
        self.spinner.hide()

    def display_results(self, results):
        max_results = self.max_images_input.value()
        threshold = self.slider.value() / 100.0


        while self.result_box_layout.count():
            child = self.result_box_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if self.file_input.text().strip() and not self.description_input.text().strip():

            filtered_results = [result for result in results if result[1] >= threshold]
        else:

            filtered_results = results

        if not filtered_results:
            no_result_label = QLabel('No results found.')
            self.result_box_layout.addWidget(no_result_label)
        else:
            for file_path, score in filtered_results:
                if max_results <= 0:
                    break
                max_results -= 1
                pixmap = QPixmap(file_path).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio) 


                label = QLabel()
                label.setPixmap(pixmap)
                label.setCursor(Qt.CursorShape.PointingHandCursor)  
                label.mousePressEvent = lambda event, path=file_path: self.open_image(path) 

                score_label = QLabel(f"Score: {score:.2f}")
                score_label.setStyleSheet('color: #dcdcdc; font-size: 16px; margin-top: 5px;')  

                layout = QVBoxLayout()
                layout.addWidget(label)
                layout.addWidget(score_label)  
                layout.setContentsMargins(10, 10, 10, 10) 

                result_widget = QWidget()
                result_widget.setLayout(layout)
                result_widget.setStyleSheet('background-color: #3c3c3c; border-radius: 10px; padding: 10px; margin-bottom: 10px;')  # Single frame style
                self.result_box_layout.addWidget(result_widget)

    def open_image(self, file_path):
        try:
            if platform.system() == 'Darwin':  
                subprocess.run(['open', file_path])
            elif platform.system() == 'Windows':  
                subprocess.run(['start', file_path], shell=True)
            else:  
                subprocess.run(['xdg-open', file_path])
        except Exception as e:
            print(f"Error opening image: {e}")


    def clear_upload_and_description(self):
        self.file_input.clear()
        self.description_input.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
