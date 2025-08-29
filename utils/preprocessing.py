import os
import cv2
import numpy as np
import pandas as pd

# Set dataset and output paths
DATASET_PATH = "YOUR_GOOGLE_DRIVE_PATH/Harmony"  # <-- Change this
OUTPUT_FILE = "Output/features.xlsx"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def extract_color_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))

    # Mean & Std of RGB
    mean_r, std_r = np.mean(img[:,:,0]), np.std(img[:,:,0])
    mean_g, std_g = np.mean(img[:,:,1]), np.std(img[:,:,1])
    mean_b, std_b = np.mean(img[:,:,2]), np.std(img[:,:,2])

    # Mean & Std of HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean_h, std_h = np.mean(hsv[:,:,0]), np.std(hsv[:,:,0])
    mean_s, std_s = np.mean(hsv[:,:,1]), np.std(hsv[:,:,1])
    mean_v, std_v = np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])

    return [mean_r,std_r,mean_g,std_g,mean_b,std_b,
            mean_h,std_h,mean_s,std_s,mean_v,std_v]

# Extract features from dataset
data = []
labels = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root, file)
            features = extract_color_features(path)
            data.append(features)
            # Combine parent folder (Good/Bad) + subfolder
            parent = os.path.basename(os.path.dirname(os.path.dirname(path)))  # Good/Bad
            sub = os.path.basename(os.path.dirname(path))                       # subclass
            label = f"{parent}/{sub}"
            labels.append(label)

# Save to Excel
columns = ["mean_r","std_r","mean_g","std_g","mean_b","std_b",
           "mean_h","std_h","mean_s","std_s","mean_v","std_v","label"]

df = pd.DataFrame(data, columns=columns[:-1])
df["label"] = labels
df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Features saved to {OUTPUT_FILE}")