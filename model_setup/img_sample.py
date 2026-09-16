import cv2
from img_preprocess import match_charge_cnn

# load one example:
img = cv2.imread("data_crop/val/1-3/200033.png")
result = match_charge_cnn(img, None, (0,0,img.shape[1], img.shape[0]))
print(f"Predicted: {result}")
