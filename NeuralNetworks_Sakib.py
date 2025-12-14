import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# -------------------------------------------------------------
# VISUALIZE DATA

plt.figure(figsize=(5, 2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
plt.show()

# -------------------------------------------------------------
# PREPROCESSING

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# -------------------------------------------------------------
# MODEL DEFINITION

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# -------------------------------------------------------------
# MODEL SUMMARY

model.summary()

# -------------------------------------------------------------
# MODEL COMPILATION / TRAINING

model.compile(
    optimizer=keras.optimizers.SGD(),
    loss="sparse_categorical_crossentropy"
)

model.fit(x_train, y_train, epochs=20)


# -------------------------------------------------------------
# MODEL PREDICTION

plt.close('all')

y_pred = model.predict(x_test[:10])

plt.figure(figsize=(5, 2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.title("Predicted label: " + str(np.argmax(y_pred[i])))
    plt.imshow(x_test[i], cmap='gray')
    plt.axis('off')
plt.show()
