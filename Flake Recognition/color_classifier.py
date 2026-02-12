import tensorflow as tf
import numpy as np
from tkinter import filedialog
from sklearn.model_selection import train_test_split

file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
data = np.loadtxt(file_path, skiprows=1)

X = data[:, :6]
y = data[:, 6] 

X = X.astype(np.float32) / 255.0
y = y.astype(np.int32)

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.1,
    stratify=y,
    random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(6,)),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, activation="relu"),

    tf.keras.layers.Dense(5, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

batch_size = 1024

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(100_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

epochs = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

model.save("color_classifier_tf.keras")

# Prediction
sample = np.array([[144, 157, 117, 164, 133, 96]], dtype=np.float32) / 255.0

pred = model.predict(sample)
class_id = np.argmax(pred, axis=1)[0]

print("Predicted class:", class_id)