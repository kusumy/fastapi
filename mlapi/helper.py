# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'
import matplotlib
import numpy as np
import pandas as pd
import arrow
import time

# Plotly
import plotly.graph_objects as go
#pd.options.plotting.backend = "plotly"

from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


def apply_styles():
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['figure.figsize'] = (20, 10)
    matplotlib.rcParams['lines.linewidth'] = 1


def filterDatabyDate(df, start_date, end_date):
    # start_date = strptime(start_date, format='%Y-%mm-%d %H:%M:%S')
    # end_date = strptime(end_time, format='%Y-%mm-%dd %H:%M:%S')

    df = df[start_date:end_date]
    df = df.copy(deep=False)
    return df


# create future forecast dates
def create_dates(start, end, freq):
    v = pd.date_range(start=start, end=end, freq=freq, closed=None)
    datetime_forecast = pd.DataFrame(index=v)
    return datetime_forecast


# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Function to get metrics model (R2, MAE, MSE dan MAPE)
def model_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return r2, mae, mse, mape


# feature builder - This section creates feature set with lag number of predictors--Creating features using lagged data
def timeseries_feature_builder(df, lag=1, lead=0, dropnan=True):
    df_copy = df.copy()
    for i in range(1, lag + 1):
        df_copy['lag' + str(i)] = df.shift(i) 
    
    if lead > 0:
        for j in range(1, lead + 1):
            df_copy['lead' + str(j)] = df.shift(-j) 
    
    # drop rows with NaN values
    if dropnan:
        df_copy.dropna(axis=0, inplace=True)
        
    return df_copy


# train-test split for a user input ratio
def train_test_split(value, ratio):
    nrow = len(value)
    print('Total samples: ', nrow)
    split_row = int((nrow) * ratio)
    print('Training samples: ', split_row)
    print('Testing samples: ', nrow - split_row)
    train = value.iloc[:split_row]
    test = value.iloc[split_row:]
    return train, test, split_row


# data transformation
def data_transformation(train, test):
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.fit_transform(test)         
    train_scaled_df = pd.DataFrame(train_scaled, index=train.index, columns=[train.columns])
    test_scaled_df = pd.DataFrame(test_scaled, index=test.index, columns=[test.columns])
    return train_scaled_df, test_scaled_df, scaler


# preprocessing -- drop null values and make arrays
def make_arrays(train, test):
    X_train_array = train.dropna().drop(train.columns[0], axis=1).values
    y_train_array = train.dropna()[train.columns[0]].values
    X_test_array = test.dropna().drop(test.columns[0], axis=1).values
    y_test_array = test.dropna()[test.columns[0]].values
    return X_train_array, y_train_array, X_test_array, y_test_array


# fitting & Validating using SVR
def fit_model(modelkind, X_train_array, y_train_array, X_test_array, y_test_array):
    if modelkind == 'svr':
        model_ml = svm.SVR(kernel='rbf', gamma='auto', tol=0.001, C=10.0, epsilon=0.001)
    if modelkind == 'rf':
        model_ml = RandomForestRegressor(max_depth=6, n_estimators=50)
    # if modelkind == 'xgb':     
    #   model_ml = xgb.XGBRegressor(n_estimators=1000)
     
    model_ml.fit(X_train_array, y_train_array)
    y_pred_train = model_ml.predict(X_train_array)
    y_pred_test = model_ml.predict(X_test_array)        
    #print('r-square_SVR_Test: ', round(model_svr.score(X_test_array,y_test_array),2))
    r2, mae, mse, mape = model_metrics(y_test_array, y_pred_test)
    #print("MAPE  (test): {:0.4f} %".format(mape))
    print("MAE  (test): {:0.4f}".format(mae))
    print("MSE  (test): {:0.4f}".format(mse))
    print("R2   (test): {:0.4f}".format(r2))

    return model_ml, y_pred_test, y_pred_train


# validation result
def valid_result_scaled(scaler, y_pred_train, y_pred_test, df, split_row, lag):
    new_train = df.iloc[:split_row].copy()
    train_pred = new_train.iloc[lag:].copy()
    y_pred_train_transformed = scaler.inverse_transform([y_pred_train])
    y_pred_train_transformed_reshaped = np.reshape(y_pred_train_transformed, (y_pred_train_transformed.shape[1],-1))
    train_pred['predicted'] = np.array(y_pred_train_transformed_reshaped)
    print('Number of training samples:', new_train.shape)
    print('Number of training samples after building timeseries features:', train_pred.shape)
    print('Number of prediction samples on training dataset:', y_pred_train_transformed_reshaped.shape)
    
    new_test = df.iloc[split_row:]
    test_pred = new_test.iloc[lag:].copy()
    y_pred_test_transformed = scaler.inverse_transform([y_pred_test])
    y_pred_test_transformed_reshaped = np.reshape(y_pred_test_transformed, (y_pred_test_transformed.shape[1],-1))
    #print(y_pred_test_transformed_reshaped.shape)
    test_pred['predicted'] = np.array(y_pred_test_transformed_reshaped)
    
    print('Number of testing samples:', new_test.shape)
    print('Number of testing samples after building timeseries features:',test_pred.shape)
    print('Number of prediction samples on testing  dataset:', y_pred_test_transformed_reshaped.shape)
    
    return train_pred, test_pred


# multi-step future forecast
def forecast_scaled(X_test_array, days, model, lag, scaler):
    # Take last test sample
    last_test_sample = X_test_array[-1]                                                          # Array shape (M, )
    X_last_test_sample = np.reshape(last_test_sample, (-1, X_test_array.shape[1]))               # Convert to shape (1, M)
    y_pred_last_sample = model.predict(X_last_test_sample)                                       # Predict y+1 from last observation
    new_array = X_last_test_sample
    new_predict = y_pred_last_sample

    seven_days_svr = []                                                                          # Create list which hold predicted value
    for i in range(0, days): 
        # Insert last predicted value to feature matrix
        new_array = np.insert(new_array, 0, new_predict)                                         # Matrix dimension is (M+1, )
        # Delete last array item, that is y(t-lag). 
        # Because new predicted value is become y(t-1)
        # The new array is shifted
        new_array = np.delete(new_array, -1)                                                     # Matrix dimension is (M, )
        new_array_reshape = np.reshape(new_array, (-1,lag))                                      # Reshape Matrix to (1, M)
        new_predict = model.predict(new_array_reshape)
        temp_predict = scaler.inverse_transform([new_predict])                                   # Matrix predict is (1,1)
        seven_days_svr.append(temp_predict[0][0].round(2))                                       # Append predicted value to forecasting list
    return seven_days_svr


def plot_train_test(train, test, title, xtitle, ytitle, height):
    fig_tt = go.Figure()
    fig_tt.add_trace(go.Scatter(x=train.index, y=train.y, mode='lines', name='Train'))
    fig_tt.add_trace(go.Scatter(x=test.index, y=test.y, mode='lines', name='Test'))

    # Edit the layout
    fig_tt.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle, height=height)
    return fig_tt


# Function for plotting
# It as assumed the training data frame has column name y and predicted has column predicted
def plot_prediction(train_pred, test_pred, title, xtitle, ytitle):
    
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_pred.index, y=train_pred.y, mode='lines', name='Train (actual)'))
    fig.add_trace(go.Scatter(x=train_pred.index, y=train_pred.predicted, mode='lines', name='Predicted (train)'))
    fig.add_trace(go.Scatter(x=test_pred.index, y=test_pred.y, mode='lines', name='Test (actual)'))
    fig.add_trace(go.Scatter(x=test_pred.index, y=test_pred.predicted, mode='lines', name='Predicted (test)'))

    # Edit the layout
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle)
    
    return fig


# Function for plotting
# It as assumed the prediction data frame has column name y and predicted
# And forecast data frame has column name forecast
def plot_forecasting(test_pred, forecast, title, xtitle, ytitle):     
    #import plotly.express as px
    #import plotly.graph_objects as go

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_pred.index, y=test_pred.y, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=test_pred.index, y=test_pred.predicted, mode='lines', name='Predicted'))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.forecast, mode='lines', name='Forecast'))

    # Edit the layout
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle, height=600)
    
    return fig


# Plot threshold
def add_threshold(plot_fig, x0, y0, x1, y1):
    '''
    plot_fig.add_trace(go.Scatter(x=[xths], y=[yths], mode="markers", showlegend=False, opacity=1, text=['xths', str(yths)], textposition='top center', 
                        marker=dict(symbol='x-dot', color='red', opacity=1, size=10) ))
    '''

    # Create horizonal threshold
    plot_fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="purple", width=2, dash="dashdot", ))

    '''
    annotations = []
    annotations.append(dict(x=first_ex_ths_dt, y=ths, yanchor='bottom',
                                  text=f'{xths}, {yths}',
                                  font=dict(family='Arial',
                                            size=12,
                                            color='rgb(37,37,37)'),
                                  showarrow=False))

    plot_fig.update_layout(annotations=annotations)
    '''
    return plot_fig


# Function to add minutes to current date. Return date in format string
def add_minutes(start_date, minutes):
    start_date = str(start_date)
    st = arrow.get(start_date, 'YYYY-MM-DD HH:mm:ss')
    end_date = st.shift(minutes=minutes).strftime('%Y-%m-%d %H:%M:%S')
    return end_date


def add_seconds(start_date, seconds):
    start_date = str(start_date)
    st = arrow.get(start_date, 'YYYY-MM-DD HH:mm:ss')
    end_date = st.shift(seconds=seconds).strftime('%Y-%m-%d %H:%M:%S')
    return end_date


def add_hours(start_date, hours):
    start_date = str(start_date)
    st = arrow.get(start_date, 'YYYY-MM-DD HH:mm:ss')
    end_date = st.shift(hours=hours).strftime('%Y-%m-%d %H:%M:%S')
    return end_date


def add_days(start_date, days):
    start_date = str(start_date)
    st = arrow.get(start_date, 'YYYY-MM-DD HH:mm:ss')
    end_date = st.shift(days=days).strftime('%Y-%m-%d %H:%M:%S')
    return end_date


intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
    ('miliseconds', 0.001)
    )

def display_time(seconds, granularity=2):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
        if value == 0.0:
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])