import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])



def train_model(optimizer, learning_rate, momentum=None):
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate, momentum=momentum)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train_cat, batch_size=128, epochs=5, validation_split=0.2, verbose=0)

    return history



optimizers = ['adam', 'rmsprop', 'sgd']
learning_rates = [0.001, 0.01, 0.1]
momentums = [0.9, 0.95]


for optimizer in optimizers:
    for lr in learning_rates:
        for momentum in momentums if optimizer == 'sgd' else [None]:
            history = train_model(optimizer, lr, momentum)


            loss, accuracy = model.evaluate(x_test, y_test_cat)

            print(f"Optimizer: {optimizer}, LR: {lr}, Momentum: {momentum}, Test Accuracy: {accuracy}")


            plt.plot(history.history['accuracy'], label='train_accuracy')
            plt.plot(history.history['val_accuracy'], label='val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'Optimizer: {optimizer}, LR: {lr}, Momentum: {momentum}')
            plt.legend()
            plt.show()
