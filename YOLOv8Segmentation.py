from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
import re
import numpy as np

class YOLOv8Segmentation:
    def __init__(self, data_path="coco8-seg.yaml", device='cpu', img_size=640):
    #    ="coco8-seg.yaml"
        self.model = YOLO('yolov8n-seg.pt')
        self.data_path = data_path
        self.device = device
        self.img_size = img_size

    def train(self, epochs=7, batch_size=16, learning_rate=0.001):
        self.remove_folder("train")
        self.model.train(
            data=self.data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=self.img_size,
            lr0=learning_rate,
            device=self.device
        )

    def evaluate(self):
        self.remove_folder("val")
        metrics = self.model.val(data=self.data_path)
        print(metrics)

    def prediction(self, test_path, save_results=True, save_txt=True):
        self.remove_folder("predict")
        test_imgages = sorted(list(test_path.glob('*.png')))
        test_data = list(test_imgages)
    
        results = self.model.predict(
            source=[str(test_data[idx]) for idx in np.random.randint(0, len(test_data), (20,))],
            save=save_results,
            save_txt=save_txt,
            imgsz=self.img_size,
            device=self.device
        )
        print(results)
    
    def remove_folder(self, folder_name):

        directory = 'runs/segment'

        if os.path.exists(directory):
            if folder_name == "train" : pattern = r"^train\d*$"
            if folder_name == "val": pattern = r"^val\d*$" 
            if folder_name == "predict": pattern = r"^predict\d*$" 

            # List folders matching the pattern
            folders = [f for f in os.listdir(directory) if re.match(pattern, f) and os.path.isdir(f"{directory}/{f}")]

            print("Matching folders:", folders)

            for folder in folders:
                folder_path = f'runs/segment/{folder}'

                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    print(f"Folder '{folder_path}' and its contents removed successfully!")
                else:
                    print(f"Folder '{folder_path}' does not exist!")


    def plot_predicted_images(self, num_of_images=8):
        images_path = Path("runs/segment/predict")

        images = list(images_path.glob('*.jpg')) 

        num_of_images = min(len(images), num_of_images)

        # Ensure that num_of_images is at least 1
        if num_of_images <= 0:
            num_of_images = 1

        rows, cols = num_of_images, 1
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * num_of_images))

        # Plot images
        for idx, ax in enumerate(axes):
            if idx >= len(images):
                break
            img = plt.imread(images[idx])
            ax.imshow(img)
            ax.axis('off')  
            ax.set_title(f"Image {idx + 1}")

        plt.tight_layout()
        plt.show()
