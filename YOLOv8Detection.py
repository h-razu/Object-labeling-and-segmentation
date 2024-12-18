import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import re
import shutil
from pathlib import Path

class YOLOv8Detection:
    def __init__(self, data_path, device='cpu', img_size=640):
        self.model = YOLO('yolov8n.pt')
        self.data_path = data_path
        self.device = device
        self.img_size = img_size


    def train(self, epochs=7, batch_size=16, learning_rate=0.001):
        
        self.model.train(
            data=self.data_path,
            epochs=epochs,
            patience=3,
            batch=batch_size,
            imgsz=self.img_size,
            lr0=learning_rate,
            mixup=0.1,
            project='model_output/YOLO_detection',
            name="Object Labeling",
            device=self.device
        )


    def evaluate(self, model_path):
        self.remove_folder("val")
        val_model = YOLO(model_path)
        metrics = val_model.val(data=self.data_path)
        
        # Evaluation metrics
        Precision = round(metrics.results_dict['metrics/precision(B)'],3)
        Recall= round(metrics.results_dict['metrics/recall(B)'],3)
        mAP_50 = round(metrics.results_dict['metrics/mAP50(B)'],3)
        mAP_50_95 = round(metrics.results_dict['metrics/mAP50-95(B)'],3)

        metrics_data = {
            'Metric': ['Precision', 'Recall', 'mAP@50', 'mAP@50-95'],
            'Value': [Precision, Recall, mAP_50, mAP_50_95]
        }
    
        df = pd.DataFrame(metrics_data)
        print("Evaluation Metrics for Object Detection:")
        print(df)



    #prediction using randomly selected 20 pictures
    def prediction(self, model_path, test_path):
        self.remove_folder("predict")
        test_imgages = sorted(list(test_path.glob('*.png')))
        test_data = list(test_imgages)
        print(len(test_data))

        model = YOLO(model_path)
        
        preds = model.predict(
                [str(test_data[idx]) for idx in np.random.randint(0, len(test_data), (20,))],
                save=True
                )

        print(preds)

    
    def result_visualization(self, base_dir="runs/detect"):
        path = os.path.join(base_dir, "val", "F1_curve.png")
        plt.figure(figsize=(10,20))
        plt.imshow(Image.open(path))
        plt.axis('off')
        plt.show()

        path = os.path.join(base_dir, "val", "PR_curve.png")
        plt.figure(figsize=(10,20))
        plt.imshow(Image.open(path))
        plt.axis('off')
        plt.show()

        #confusion matrix print
        path = os.path.join(base_dir, "val", "confusion_matrix_normalized.png")
        plt.figure(figsize=(10,20))
        plt.imshow(Image.open(path))
        plt.axis('off')
        plt.show()


    def remove_folder(self, folder_name):

        directory = 'runs/detect'
        if folder_name == "val": pattern = r"^val\d*$" 
        if folder_name == "predict": pattern = r"^predict\d*$" 

        # List folders matching the pattern
        folders = [f for f in os.listdir(directory) if re.match(pattern, f) and os.path.isdir(f"{directory}/{f}")]

        print("Matching folders:", folders)

        for folder in folders:
            folder_path = f'runs/detect/{folder}'

            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"Folder '{folder_path}' and its contents removed successfully!")
            else:
                print(f"Folder '{folder_path}' does not exist!")


    def plot_predicted_images(self, num_of_images=5):
        images_path = Path("runs/detect/predict")

        images = list(images_path.glob('*')) 

        num_of_images = min(len(images), num_of_images)

        rows, cols = num_of_images, 1
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * num_of_images))

        # Plot images
        for idx, ax in enumerate(axes):
            if idx >= len(images):  # Ensure we don't go out of bounds
                break
            img = plt.imread(images[idx])
            ax.imshow(img)
            ax.axis('off')  # Hide axis ticks
            ax.set_title(f"Image {idx + 1}")

        plt.tight_layout()
        plt.show()

        
