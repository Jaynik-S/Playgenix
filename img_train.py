import json
import tensorflow as tf
from tensorflow.keras import layers, models

# 1) Load the data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data_crop/train",
    labels="inferred",
    label_mode="categorical",
    image_size=(32,128),    
    batch_size=32,
    shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data_crop/val",
    labels="inferred",
    label_mode="categorical",
    image_size=(32,128),
    batch_size=50
)

# 2) Capture class names & count
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# 3) Build the model
base = tf.keras.applications.MobileNetV2(
    input_shape=(32,128,3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation="softmax")
])

# 4) Compile & train
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=35
)

# 5) Save both model and class list
model.save("slot_classifier.h5")
with open("class_names.json","w") as f:
    json.dump(class_names, f)
print("✅ Training complete — slot_classifier.h5 and class_names.json written.")
