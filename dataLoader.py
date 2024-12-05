import os
import json
import numpy
import cv2

class DataLoader:
    def __init__(self, json_path, images_path, output_path, categories):
        self.json_path = json_path
        self.images_path = images_path
        self.output_path = output_path
        self.categories = {category: idx for idx, category in enumerate(categories)}
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)


    def make_annotaion(self):
        
        counter = 0
        #loading the json files
        with open(self.json_path, 'r') as file:
            data = json.load(file)
            
        #create annotation text file for each image
        for image_data in data:
            image_name = image_data['name']
            image_path = os.path.join(self.images_path, image_name)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Image {image_name} not found.")
                counter = counter +1
                continue
            
            height, width, _ = image.shape
            
            # Prepare the annotation file
            annotation_file = os.path.join(self.output_path, f'{os.path.splitext(image_name)[0]}.txt')
            with open(annotation_file, 'w') as ann_file:
                for label in image_data['labels']:
                    category = label['category']
                    if category not in self.categories:
                        continue
                    
                    category_id = self.categories[category]
                    box = label['box2d']
                    
                    # Normalize the bounding box coordinates
                    x_center = (box['x1'] + box['x2']) / 2 / width
                    y_center = (box['y1'] + box['y2']) / 2 / height
                    w = (box['x2'] - box['x1']) / width
                    h = (box['y2'] - box['y1']) / height
                    
                    # Write the annotation in YOLOv8 format
                    ann_file.write(f"{category_id} {x_center} {y_center} {w} {h}\n")
        print(counter)
