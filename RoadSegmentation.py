import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import glob
import random

class RoadSegmentation:
    def __init__(self, image_dir, mask_dir, target_size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        self.model = None

    def resize_image(self, image):
        return cv2.resize(image, self.target_size)

    def resize_mask(self, mask):
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        resized_mask = cv2.resize(mask_gray, self.target_size, interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(resized_mask, axis=2)

    def prepare_data(self):
        image_list = []
        mask_list = []

        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            # Load and preprocess images and masks
            image = cv2.imread(img_path).astype(np.float32) / 255.0
            mask = cv2.imread(mask_path).astype(np.float32) / 255.0
            resized_image = self.resize_image(image)
            resized_mask = self.resize_mask(mask)

            image_list.append(resized_image)
            mask_list.append(resized_mask)

        # Convert lists to numpy arrays
        images = np.array(image_list)
        masks = np.array(mask_list)
        
        # Split data into train and test sets
        return train_test_split(images, masks, test_size=0.3, random_state=42)

    def create_deeplabv3(self, input_shape=(512, 512, 3), num_classes=1):
        # Base model (ResNet50) with DeepLabv3 adjustments
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(input_shape[0] * input_shape[1], activation='sigmoid')(x)
        outputs = tf.keras.layers.Reshape((input_shape[0], input_shape[1], num_classes))(x)

        self.model = Model(inputs=base_model.input, outputs=outputs)

    def train_model(self, X_train, y_train, X_val, y_val, batch_size=16, epochs=5):
        if not self.model:
            raise ValueError("Model is not created yet. Call create_deeplabv3() first.")

        # Compile and train the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
        return history

    def predict_and_visualize(self, X_test, y_test, num_samples=3):
        if not self.model:
            raise ValueError("Model is not trained yet.")

        figure, axes = plt.subplots(num_samples, 3, figsize=(20, 20))

        for i in range(num_samples):
            rand_idx = random.randint(0, len(X_test) - 1)
            original_img = X_test[rand_idx]
            axes[i, 0].imshow(original_img)
            axes[i, 0].title.set_text("Original Image")
            axes[i, 0].axis("off")

            original_mask = y_test[rand_idx]
            axes[i, 1].imshow(original_mask.squeeze(), cmap="gray")
            axes[i, 1].title.set_text("Original Mask")
            axes[i, 1].axis("off")

            # pred_mask = self.model.predict(original_img[np.newaxis, ...]).squeeze()
            original_img = np.expand_dims(original_img, axis=0)
            pred_mask = self.model.predict(original_img).reshape(512,512)
            axes[i, 2].imshow(pred_mask, cmap="gray")
            axes[i, 2].title.set_text("Predicted Mask")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.show()

