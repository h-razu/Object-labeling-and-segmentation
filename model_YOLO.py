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
        if folder_name is "val": pattern = r"^val\d*$" 
        if folder_name is "predict": pattern = r"^predict\d*$" 

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
    def prediction(self, test_path):
        self.remove_prediction_folder("predict")
        test_imgages = sorted(list(test_path.glob('*.png')))
        test_data = list(test_imgages)
        print(len(test_data))

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

        # Print evaluation results
        # print("Evaluation Results:")
        # print(f"Precision: {results.preds['precision']}")
        # print(f"Recall: {results.preds['recall']}")
        # print(f"mAP@0.5: {results.preds['mAP50']}")
        # print(f"mAP@0.5:0.95: {results.preds['mAP50_95']}")
        print("KKKK")
        print(metrics)
        print("KKKK")

        # # Average Precision at IoU=0.5
        # print(f"AP (Average Precision at IoU=0.5): {metrics.boxmaps['AP50']}")

        # # mAP at IoU from 0.5 to 0.95 (mean of all IoU thresholds)
        # print(f"mAP (mean Average Precision at IoU=0.5:0.95): {metrics.boxmaps['mAP_0.5:0.95']}")

        # # Precision: Percentage of correct predictions among all positive predictions
        # print(f"Precision: {metrics.preds['precision']}")

        # # Recall: Percentage of correct predictions among all ground-truth objects
        # print(f"Recall: {metrics.preds['recall']}")

        # # F1-Score: Harmonic mean of precision and recall
        # print(f"F1-Score: {metrics.preds['f1']}")

        # # Confusion matrix (matrix showing true positive, false positive, false negative, etc.)
        # print(f"Confusion Matrix: {metrics.confusion_matrix}")

        # # Per-class APs (Average Precision for each class)
        # for class_name, ap in metrics.boxmaps['AP_class'].items():
        #     print(f"AP for class {class_name}: {ap}")
