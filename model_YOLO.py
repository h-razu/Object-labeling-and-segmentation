from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import re
import shutil

class ModelYOLO:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')


    def yolo_training(self, data, epochs, batch_size, image_size, project_name, exp_name, device):
        
        self.model.train(
            data=data,
            epochs=epochs,
            patience=3,
            batch=batch_size,
            imgsz=image_size,
            mixup=0.1,
            project=project_name,
            name=exp_name,
            device=device
        )

    def result_visualization(self, base_dir):
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

    #prediction using randomly selected 20 pictures
    def prediction(self, model_path, test_path):
        self.remove_folder("predict")
        test_imgages = sorted(list(test_path.glob('*.png')))
        test_data = list(test_imgages)
        print(len(test_data))

        model = YOLO(model_path)
        
        preds = self.model.predict(
                [str(test_data[idx]) for idx in np.random.randint(0, len(test_data), (20,))],
                save=True
                )

        print(preds)

    def plot_predicted_images(self, images_path, num_of_images=8):
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


    def evaluation_on_validate(self,model_path, val_data_path, save_dir='runs/val_evaluation'):
        self.remove_folder("val")
        model = YOLO(model_path)
    
        # Evaluate the model using the validation dataset
        metrics = model.val(
            data=val_data_path,
            save_dir=save_dir  
        )

        print(metrics)

        
