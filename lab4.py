import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

url = "http://tk.ulstu.ru/lib/info/usdrub.txt"
data = pd.read_csv(url, sep=";", header=None, names=["TICKER", "PER", "DATE", "TIME", "CLOSE"])
print(data.head())

quotes = data["CLOSE"].tolist()

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length+1])
    return sequences

N = 50
sequences = create_sequences(quotes, N)

train_data, test_data = train_test_split(sequences, test_size=0.2, shuffle=False)
train_data, val_data = train_test_split(train_data, test_size=0.2, shuffle=False)

X_train, y_train = zip(*train_data)
X_val, y_val = zip(*val_data)
X_test, y_test = zip(*test_data)

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(N,)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)

loss_without_dropout, accuracy_without_dropout = model.evaluate(X_test, y_test)

model_dropout = Sequential([
    Dense(64, activation='relu', input_shape=(N,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model_dropout.compile(optimizer='adam', loss='mean_squared_error')

history_dropout = model_dropout.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)

loss_with_dropout, accuracy_with_dropout = model_dropout.evaluate(X_test, y_test)

print("without Dropout:", accuracy_without_dropout)
print("with Dropout:", accuracy_with_dropout)



