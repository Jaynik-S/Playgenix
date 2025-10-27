import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# 1) Load the data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data_crop/train",
    labels="inferred",
    label_mode="categorical",
    image_size=(32, 128),    # height, width
    batch_size=32,
    shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data_crop/val",
    labels="inferred",
    label_mode="categorical",
    image_size=(32, 128),
    batch_size=50
)

# 2) Capture class names & count BEFORE augmentation
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# 3) Data augmentation (inline)
data_augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.1, 0.1),       # ±10% shift
    layers.RandomBrightness(0.2),             # ±20% brightness
    layers.RandomContrast(0.2),               # ±20% contrast
    layers.RandomRotation(0.05),              # ±5% rotation
])

def augment(x, y):
    return data_augment(x), y

train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# 4) Build the model
base = tf.keras.applications.MobileNetV2(
    input_shape=(32, 128, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False  # freeze base

model = models.Sequential([
    layers.Rescaling(1.0 / 255.0, input_shape=(32, 128, 3)),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),              # reduce overfitting
    layers.Dense(num_classes, activation="softmax")
])

# 5) Compile & initial training
es = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)
mc = callbacks.ModelCheckpoint(
    "best_slot_classifier.h5", monitor="val_accuracy", save_best_only=True
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[es, mc]
)

# 6) Fine-tune: unfreeze last conv layers
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[es, mc]
)

# 7) Save both final model and class list
model.save("slot_classifier.h5")
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("✅ Training complete — slot_classifier.h5 and class_names.json written.")
