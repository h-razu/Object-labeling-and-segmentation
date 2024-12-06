import matplotlib.pyplot as plt
from pathlib import Path
import prettyprinter as pp
import cv2
import os

class DataExploration:    
    def __init__(self, image_path, label_path):
        self.image_path = Path(image_path)
        self.label_path = Path(label_path)

    def visualize_image(self, num_of_image=3):
        plt.figure(figsize=(30, 20))

        print("Some Sample Images: ")
        img = sorted(list(self.image_path.glob('*')))
        for i, img_path in enumerate(img[:num_of_image]):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Image {img_path} is not readable.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(1, num_of_image, i+1)
            plt.imshow(img)
            plt.title(os.path.basename(img_path))
            plt.axis('off')
        plt.show()


    def class_distribution(self):
        # labels = sorted(list(self.label_path.glob('*')))
        print()