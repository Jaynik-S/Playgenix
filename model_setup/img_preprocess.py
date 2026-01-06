# img_preprocess.py  
import cv2, numpy as np, tensorflow as tf, json

# 1) Load the best‐performing model & class names once
model = tf.keras.models.load_model("assets/models/best_slot_classifier.h5")
with open("assets/models/class_names.json") as f:
    class_names = json.load(f)

# 2) Grab height (H) and width (W) from model.input_shape
#    model.input_shape is (None, 32, 128, 3)
_, H, W, _ = model.input_shape

def match_charge_cnn(screenshot, _, bbox):
    x, y, w, h = bbox
    crop = screenshot[y:y+h, x:x+w]

    # A) Convert BGR→RGB
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # B) Resize to (width, height) = (W, H)
    img = cv2.resize(img, (W, H))

    # C) Cast to float32 *without* dividing by 255
    img = img.astype("float32")

    # D) Predict
    preds = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    idx = int(np.argmax(preds))

    return class_names[idx]
