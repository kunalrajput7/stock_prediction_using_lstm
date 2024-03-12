# Stock Prediction Using LSTM Algorithm
This project demonstrates how to use Long Short-Term Memory (LSTM) neural networks to predict stock prices. The LSTM algorithm is a type of recurrent neural network (RNN) architecture that is well-suited for time series prediction tasks.

## Overview
The code provided in this repository fetches historical stock price data from the Tiingo API, preprocesses it, and trains an LSTM model to predict future stock prices. The model architecture includes multiple LSTM layers followed by dropout layers for regularization.

### Key Components:
- Data Collection: Historical stock price data for a chosen stock (e.g., Apple - AAPL) is fetched from the Tiingo API using the Pandas DataReader library.
- Preprocessing: The fetched data is preprocessed, including scaling the values to a range of [0, 1] using MinMaxScaler, and dividing it into training and testing datasets.
- Model Architecture: The LSTM model is built using the Sequential API from TensorFlow Keras. The model consists of multiple LSTM layers followed by dropout layers to prevent overfitting. The final layer is a dense layer with one output unit.
- Training: The model is trained using the training dataset with a specified number of epochs and batch size. The Adam optimizer is used, and the loss function is mean squared error.
- Prediction: After training, the model is used to predict future stock prices using the testing dataset. The predictions are then evaluated using root mean squared error (RMSE) metrics.

## Model Architechture
The LSTM model architecture used in this project is as follows:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
```
This architecture consists of multiple LSTM layers with dropout regularization between them. The input shape is specified as (time_steps, features), where time_steps represents the number of previous time steps considered for prediction.

## Training
The model is trained using the training dataset (X_train and y_train) for a specified number of epochs and batch size. Additionally, a portion of the training data is used for validation during training to monitor the model's performance.

## Results
The trained model's performance is evaluated using root mean squared error (RMSE) metrics on both training and testing datasets. Additionally, the predictions are visualized to compare them with actual stock prices.

---
Feel free to customize the README further with additional information, usage instructions, or any other relevant details specific to your project.
