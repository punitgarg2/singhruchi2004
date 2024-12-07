import time
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from flask import Flask
import os

import tensorflow as tf


from helper import *
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go

import os
sp500_adj_close_data = pd.read_csv('sp500_adj_close.csv')
#yha hmne ek table create kiya h 100 colounm ka
cols = []
for i in range(100):
    cols.append(f'MA_{i+1}')

# Sidebar for stock selection
with st.sidebar:

  company_list_for_user_selection = sp500_adj_close_data.columns[1:]

  selected_stock = st.sidebar.selectbox(
    "Select a stock", company_list_for_user_selection,

  )

# Display current selected stock
st.header(f"Current selected stock is of :red[{selected_stock}]")

# Maintaining a dates dataframe
date_df = sp500_adj_close_data['Date']

# Drop Dates columns from sp500_adj_close_data
sp500_adj_close_data = sp500_adj_close_data.drop(['Date'],axis = 1)

# Min-Max in selected stock for inverse transform
min_in_selected_stock = np.nanmin(sp500_adj_close_data[selected_stock])
max_in_selected_stock = np.nanmax(sp500_adj_close_data[selected_stock])
st.write(min_in_selected_stock)
st.write(max_in_selected_stock)



# Create Moving Averages dataframe for Past 100 days

moving_average_dataframe = return_ma_df(selected_stock)

if len(moving_average_dataframe)<1000:
    st.write('Selected stock has been recently listed, our model will be unable to predict')

# creating train test split from moving_average_dataframe use function

x_train, x_test, y_train, y_test, dates_curr = create_train_test_split(moving_average_dataframe,date_df)
moving_average_dataframe




#Generating Plotting Data
plot_train,plot_test,dates_train,dates_test = create_plotting_data(moving_average_dataframe,date_df,min_in_selected_stock,max_in_selected_stock)




# Training  LSTM
y_pred,model = train_lstm(x_train,y_train,x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"R-squared (R²) Score: {r2}")



# inverse transforming y_pred

y_pred = y_pred.ravel()
y_pred = y_pred*(max_in_selected_stock - min_in_selected_stock) + min_in_selected_stock

# Forecast Price for next 60 working days

start_date = datetime.strptime('2024-07-11', '%Y-%m-%d')
next_60_dates = forecasting_dates(start_date)

last_value = moving_average_dataframe[-1:]
last_value = last_value.iloc[:,1:]

pred_df = forecast_next_60_days(last_value,model,cols)

forecasted_values = pred_df['Adj_Close_Price'].values
forecasted_values = forecasted_values*(max_in_selected_stock-min_in_selected_stock) + min_in_selected_stock



# Historical Date Plot

fig = go.Figure(
        data=[
            go.Scatter(
                x = dates_train,
                y = plot_train,
                name="Train",
                mode="lines",
                line=dict(color="red"),
            ),
            go.Scatter(
                x = dates_test,
                y = plot_test,
                name="Test",
                mode="lines",
                line=dict(color="blue"),
            ),
            go.Scatter(
                x = dates_test,
                y = y_pred,
                name="Predictions",
                mode="lines",
                line=dict(color="orange"),
            ),
            go.Scatter(
                x = next_60_dates,
                y = forecasted_values,
                name="forecasting",
                mode="lines",
                line=dict(color="green"),
            )


        ]
    )
fig.update_layout(xaxis_rangeslider_visible=False,width=800)
st.plotly_chart(fig, use_container_width=True,width=800)




