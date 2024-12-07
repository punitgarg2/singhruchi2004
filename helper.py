import pandas as pd
import numpy as np
import streamlit as st
import pickle
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta



@st.cache_data(show_spinner=False)
def return_ma_df(company):
    with open('moving_average_dict_1.pkl', 'rb') as file:
        moving_average_dict = pickle.load(file)
    return moving_average_dict[company]

@st.cache_data(show_spinner=False)
def create_train_test_split(moving_average_dataframe,date_df):
    y = moving_average_dataframe['Adj_Close_Price']

    # st.write(moving_average_dataframe.columns)

    x = moving_average_dataframe.drop(['Adj_Close_Price'], axis=1)

    # Date corresponding to moving averages

    dates_curr = date_df.loc[moving_average_dataframe.index[0]:]

    # Defining splitting length --> 70% (train)
    splitting_length = int(len(x) * 0.7)
    x_train = x[:splitting_length]
    y_train = y[:splitting_length]

    x_test = x[splitting_length:]
    y_test = y[splitting_length:]

    # Preserving dates for x_test,y_test
    dates_curr = dates_curr[splitting_length:]
    # Reforming x_train,x_test,y_train,y_test to LSTM compatible input

    x_train = np.asarray(x_train).astype(np.float64)

    y_train = np.asarray(y_train).astype(np.float64)

    x_test = np.asarray(x_test).astype(np.float64)

    y_test = np.asarray(y_test).astype(np.float64)

    return x_train, x_test, y_train, y_test, dates_curr


@st.cache_data(show_spinner=False)
def create_plotting_data(moving_average_dataframe,date_df,min_in_selected_stock,max_in_selected_stock):
    splitting_length = int(len(moving_average_dataframe)*0.7)
    adj_close_price = moving_average_dataframe['Adj_Close_Price']

    plot_train = adj_close_price[:splitting_length]
    plot_test = adj_close_price[splitting_length:]

    plot_train = plot_train*(max_in_selected_stock-min_in_selected_stock) + min_in_selected_stock
    plot_test = plot_test*(max_in_selected_stock-min_in_selected_stock) + min_in_selected_stock

    date_df = date_df.loc[moving_average_dataframe.index[0]:]
    dates_train = date_df[:splitting_length]
    dates_test = date_df[splitting_length:-1]


    return plot_train,plot_test,dates_train,dates_test



@st.cache_data(show_spinner=False)
def train_lstm(x_train,y_train,x_test):
    model = Sequential()
    model.add(LSTM(100, return_sequences=False, input_shape=(x_train.shape[1], 1)))



    # model.add(LSTM(16, return_sequences=False))

    # model.add(Dense(16, activation='relu'))

    model.add(Dense(16, activation='relu'))


    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(x_train, y_train, epochs=4, batch_size=32, validation_split=0.2, verbose=1)

    y_pred = model.predict(x_test)

    return y_pred,model


@st.cache_data(show_spinner=False)
def forecast_next_60_days(last_value,model,cols):
    df_pred_60_days = pd.DataFrame(index=[i for i in range(60)], columns=cols)
    df_pred_60_days['Adj_Close_Price'] = None

    start_future_pred = last_value.values

    start_future_pred = np.asarray(start_future_pred).astype(np.float64)

    future_pred = model.predict(start_future_pred)

    start_future_pred = np.append(start_future_pred, future_pred[0])

    df_pred_60_days.loc[0] = start_future_pred

    for i in range(1, 60):
        next_values = df_pred_60_days.iloc[i - 1:i, 1:].values
        next_values = np.asarray(next_values).astype(np.float64)
        pred = model.predict(next_values)
        next_values = np.append(next_values, pred[0])
        df_pred_60_days.loc[i] = next_values
    return df_pred_60_days



@st.cache_data(show_spinner=False)
def forecasting_dates(start_date):
    dates = []
    current_date = start_date

    while len(dates) < 60:
        # Only weekdays
        if current_date.weekday() < 5:
            dates.append(current_date.strftime('%Y-%m-%d'))

        current_date += timedelta(days=1)
    return dates

