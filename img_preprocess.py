import cv2, numpy as np
import tensorflow as tf
import json

# load once at module import
model = tf.keras.models.load_model("slot_classifier.h5")
with open("class_names.json") as f:
    class_names = json.load(f)

# at top of file
_, H, W, _ = model.input_shape   # H=128, W=32 in your current model

def match_charge_cnn(screenshot, _, bbox):
    x, y, w, h = bbox
    crop = screenshot[y:y+h, x:x+w]
    # cv2.resize wants (width, height):
    img = cv2.resize(crop, (W, H))          # --> (32,128) -> result is (128,32,3)
    img = img.astype("float32") / 255.0
    preds = model.predict(np.expand_dims(img, 0))[0]
    idx = np.argmax(preds)
    return class_names[idx]
