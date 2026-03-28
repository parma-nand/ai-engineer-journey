import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Normalize (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0


model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # convert 2D → 1D
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 digits (0–9)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32
)

loss,accuracy=model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)



print("UpTo Date")