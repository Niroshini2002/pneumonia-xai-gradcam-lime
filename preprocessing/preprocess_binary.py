import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# --- Config ---
IMG_SIZE = 224  # standard input size for ResNet-50
RAW_DIR = "data/raw/kermany"
PROCESSED_DIR = "data/processed/binary"

def load_images(folder, label):
    """Load images from a folder and assign a label."""
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # normalize to [0,1]
            images.append(img)
            labels.append(label)
    return images, labels

def preprocess():
    normal_imgs, normal_labels = load_images(f"{RAW_DIR}/normal", 0)
    pneumonia_imgs, pneumonia_labels = load_images(f"{RAW_DIR}/pneumonia", 1)

    X = np.array(normal_imgs + pneumonia_imgs)
    y = np.array(normal_labels + pneumonia_labels)

    # Stratified split to preserve class balance
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(f"{PROCESSED_DIR}/X_train.npy", X_train)
    np.save(f"{PROCESSED_DIR}/y_train.npy", y_train)
    np.save(f"{PROCESSED_DIR}/X_val.npy", X_val)
    np.save(f"{PROCESSED_DIR}/y_val.npy", y_val)
    np.save(f"{PROCESSED_DIR}/X_test.npy", X_test)
    np.save(f"{PROCESSED_DIR}/y_test.npy", y_test)

    print(f"Done. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

if __name__ == "__main__":
    preprocess()