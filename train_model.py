import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.constraints import MaxNorm

def train_model(features, targets):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    # Reshape les données pour LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(1, X_train.shape[2]), kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    y_pred = model.predict(X_test)
    for i, column in enumerate(targets.columns):
        accuracy = np.mean(np.round(y_pred[:, i]) == y_test[column])
        print(f"Accuracy for {column}: {accuracy * 100:.2f}%")
    
    return model

if __name__ == "__main__":
    from prepare_data import prepare_data
    features, targets = prepare_data("euromillions_202002.csv")
    model = train_model(features, targets)