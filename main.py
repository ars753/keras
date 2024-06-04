import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random

#creating model
model=Sequential()

#add layer
model.add(Dense(1,input_shape=(3,), activation='linear'))#enter layer with 3 neuroon
#model.add(Dense(1))#enter layer with 1 neuroon
opt=keras.optimizers.Adam(learning_rate=0.1)
#compile model
model.compile(optimizer=opt,loss='mse')


X_train= np.array([(random.randint(0,1000),random.randint(0,1000),random.randint(0,1000)) for _ in range(100)])
y_train= np.array([(X_train[i][0] + 2 * X_train[i][1] + 3 * X_train[i][2]) for i in range(100)])


learn=model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

print(model.predict(np.array([(2,1,1)])))
print(y_train[1],X_train[1])

