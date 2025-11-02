import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt


num_samples = 1000
img_height, img_width = 28, 28
num_classes = 10


xt = np.random.rand(num_samples, img_height, img_width, 1).astype('float32')
yt = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes)
xts = np.random.rand(200, img_height, img_width, 1).astype('float32')
yts = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 200), num_classes)


def create_model():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # smaller model
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

optimizers = ['adam', 'adagrad']
history = {}

for optim in optimizers:
    print(f"\nTraining with optimizer: {optim}")
    model = create_model()
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    history[optim] = model.fit(
        xt, yt,
        epochs=3, batch_size=32,  
        validation_data=(xts, yts),
        verbose=1
    )


for optim in optimizers:
    plt.figure(figsize=(5, 3))
    plt.plot(history[optim].history['accuracy'], label='Training Accuracy')
    plt.plot(history[optim].history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Optimizer: {optim}")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

plt.show()
