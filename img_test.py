import os, json, cv2, numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from img_preprocess import model, class_names, match_charge_cnn  # assuming you expose these

y_true, y_pred = [], []
for label in sorted(os.listdir("data/val")):
    folder = os.path.join("data/val", label)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        crop = cv2.imread(img_path)
        pred = match_charge_cnn(crop, None, (0,0,crop.shape[1],crop.shape[0]))
        y_true.append(label)
        y_pred.append(pred)

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred, labels=class_names)
print(f"Val accuracy: {acc:.2%}")
print("Confusion matrix:")
print(cm)
