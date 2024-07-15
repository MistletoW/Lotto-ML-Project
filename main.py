import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # Check if data is a 1D array or list, set n_vars accordingly
    n_vars = 1 if type(data) is list or len(data.shape) == 1 else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# Load and preprocess data
def load_data(file_path):
    return pd.read_excel(file_path, engine='openpyxl')

# Define and train LSTM model
def train_lstm(X, y, n_input):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    return model

# Define and train GRU model
def train_gru(X, y, n_input):
    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    return model

# Train and predict with ARIMA
def train_predict_arima(series, order=(5,1,0)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    prediction = model_fit.forecast()[0]
    return prediction

# Train and predict with Ensemble method
def train_ensemble(X, y):
    model1 = RandomForestRegressor(n_estimators=100)
    model2 = GradientBoostingRegressor()
    ensemble = VotingRegressor(estimators=[('rf', model1), ('gb', model2)])
    ensemble.fit(X, y)
    return ensemble

# Adjusted parts within the main function

def main(file_path):
    df = load_data(file_path)
    n_input = 3  # Look back period
    predictions = {'LSTM': [], 'GRU': [], 'ARIMA': [], 'Ensemble': []}
    
    for i, column in enumerate(df.columns):
        series = df[column].values
        supervised = series_to_supervised(series, n_in=n_input)
        X, y = supervised.values[:, :-1], supervised.values[:, -1]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # LSTM
        lstm_model = train_lstm(X, y, n_input)
        lstm_pred = np.round(lstm_model.predict(X[-1].reshape((1, n_input, 1)))[0][0]).astype(int)
        predictions['LSTM'].append(lstm_pred)
        
        # GRU
        gru_model = train_gru(X, y, n_input)
        gru_pred = np.round(gru_model.predict(X[-1].reshape((1, n_input, 1)))[0][0]).astype(int)
        predictions['GRU'].append(gru_pred)
        
        # ARIMA
        arima_pred = np.round(train_predict_arima(series)).astype(int)
        predictions['ARIMA'].append(arima_pred)
        
        # Ensemble
        X_flat = X.reshape(X.shape[0], X.shape[1])
        ensemble_model = train_ensemble(X_flat, y)
        ensemble_pred = np.round(ensemble_model.predict(X_flat[-1].reshape(1, -1))[0]).astype(int)
        predictions['Ensemble'].append(ensemble_pred)

    # Print predictions in grid format
    print("Predictions:\n")
    print(f"{' ':<10} {'Number 1':<10} {'Number 2':<10} {'Number 3':<10} {'Number 4':<10} {'Number 5':<10} {'PP Number':<10}")
    for model in predictions:
        print(f"{model:<10} ", end="")
        for pred in predictions[model]:
            print(f"{pred:<10} ", end="")
        print()  # New line for next model

if __name__ == "__main__":
    file_path = './parsed_lotto_data.xlsx'  # Update this to your Excel file path
    main(file_path)

