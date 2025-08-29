import cv2
import numpy as np
import pandas as pd
import os

def extract_color_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))

    # Mean & Std of RGB
    mean_r = np.mean(img[:, :, 0]); std_r = np.std(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1]); std_g = np.std(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2]); std_b = np.std(img[:, :, 2])

    # Mean & Std of HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean_h = np.mean(hsv[:, :, 0]); std_h = np.std(hsv[:, :, 0])
    mean_s = np.mean(hsv[:, :, 1]); std_s = np.std(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2]); std_v = np.std(hsv[:, :, 2])

    return [mean_r, std_r, mean_g, std_g, mean_b, std_b,
            mean_h, std_h, mean_s, std_s, mean_v, std_v]

def save_features_to_excel(data, labels, output_file="features.xlsx"):
    columns = [
        "mean_r","std_r","mean_g","std_g","mean_b","std_b",
        "mean_h","std_h","mean_s","std_s","mean_v","std_v","label"
    ]
    df = pd.DataFrame(data, columns=columns)
    df["label"] = labels
    df.to_excel(output_file, index=False)
    print(f"✅ Features saved to {output_file}")
