import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras
from keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#input data
x_train=x_train/255
x_test= x_test/255

y_train_cat=keras.utils.to_categorical(y_train, 10)
y_test_cat=keras.utils.to_categorical(y_test, 10)

#analyze with different functions
activation_functions = ['linear', 'sigmoid', 'tanh', 'relu']

#show first 25 pic from train
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()
#creating model and new layers
model=keras.Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
#compilig modelwith opt loss
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
#trainig model
model.fit(x_train, y_train_cat, batch_size=32, epochs=1, validation_split=0.2)
#shows loss and information
model.evaluate(x_test, y_test_cat)

#predict
n=2
x=np.expand_dims(x_test[n], axis=0)
res=model.predict(x)
print(res)
print(f"Answer: {np.argmax((res))}")

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()
#analyze pridict and show with argmax
pred=model.predict(x_test)
pred=np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])
print(y_test[:20])

mask=pred==y_test
print(mask[:10])

x_false=x_test[~mask]
p_false=pred[~mask]

for i in range(5):
    print("network meaning: "+str(p_false[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show(0)







