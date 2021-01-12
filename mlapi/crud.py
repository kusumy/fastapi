from sqlalchemy.orm import Session
from sqlalchemy import text, bindparam
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from pathlib import Path, PurePath
import pandas as pd
import numpy as np
import mlapi.helper as helper
import time
import os
import json

from  . import models
from .database import engine

from humanfriendly import format_timespan

# Scikit-learn
from sklearn.preprocessing import MinMaxScaler
# PyCaret
from pycaret.anomaly import *



def get_model_configuration(db: Session, target: str):
    return db.query(models.ModelConfiguration).filter(models.ModelConfiguration.target_name == target).first()

def get_allmodel_configuration(db: Session):
    return db.query(models.ModelConfiguration).all()

def get_forecast(db: Session, target: str, startdate: str, enddate: str, horizon: int=36):
    # Get model configuration
    result_model_conf = db.query(models.ModelConfiguration).filter(models.ModelConfiguration.target_name == target).first()

    lag = result_model_conf.lag_features
    forecasting_horizon = horizon
    target_column = result_model_conf.target_column
    features_column = result_model_conf.features_column
    startdate = startdate
    enddate = enddate
    file_model = result_model_conf.model_file
    forecasting_table = result_model_conf.forecasting_table
    metric_table =  result_model_conf.metric_table

    # Join target_column and features column
    if len(features_column) > 0:
        columns_all = target_column + "," + features_column
    else:
        columns_all = target_column

    # Construct sql
    sql = text("SELECT datestamp, " + columns_all + " FROM hourly_log WHERE datestamp BETWEEN :x AND :y")
    sql = sql.bindparams(
        bindparam("x", type_=String),
        bindparam("y", type_=String)
    )

    # get date
    result = db.execute(sql, {"x": startdate, "y": enddate}).fetchall()

    d, a = {}, []
    for rowproxy in result:
        # rowproxy.items() returns an array like [(key0, value0), (key1, value1)]
        for column, value in rowproxy.items():
            # build up the dictionary
            d = {**d, **{column: value}}
        a.append(d)

    # Convert to dataframe
    df = pd.json_normalize(a)

    # Set datetime as index
    df['datestamp'] = pd.to_datetime(df['datestamp'])
    df = df.set_index('datestamp')
    df.fillna(method='backfill', inplace=True)

    # By convention, target column is named 'y', so rename target column to y
    df = df.rename(columns={target_column: 'y'})

    # Create Features
    # Set how many lag as time series features
    ds_features = helper.timeseries_feature_builder(df[['y']], lag, dropnan=True)
    test_samples = len(ds_features)

    # Add other features (predictor)
    if len(features_column) > 0:
        predictors = features_column.split(",")
        ds_features[predictors] = df[predictors]
    else:
        predictors = []

    # Convert Pandas series to Array/List, and also drop records which has NaN value
    X_test = ds_features.dropna().drop(ds_features.columns[0], axis=1).values           # Remove y column
    y_test = ds_features.dropna()[ds_features.columns[0]].values.reshape(-1, 1)

    # data transformation
    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    y_test_scaled = scaler.fit_transform(y_test)

    # Note start date for test data
    start_date_test = ds_features.iloc[:1].index[0].strftime('%Y-%m-%d %H:%M:%S')
    end_date_test = ds_features.iloc[-1:].index[0].strftime('%Y-%m-%d %H:%M:%S')

    ###################################################################################
    ## Get ML Model, and do predicting
    ###################################################################################
    import pickle

    # Data folder
    current_dir = Path(__file__).resolve()
    current_dir_parent = current_dir.parent.parent
    model_folder = current_dir_parent / "model"
    file_model = model_folder / file_model

    # load the model from disk
    model = pickle.load(open(file_model, 'rb'))

    # Perform prediction (using lag as feature)
    t1 = time.process_time()
    y_pred_test = model.predict(X_test_scaled)
    t2 = time.process_time()
    execution_time = t2 - t1

    # Create new scaler to invers prediction data
    # The scaler attribute is copied from scaler that was used to scale original data
    new_scaler = MinMaxScaler()
    new_scaler.min_, new_scaler.scale_ = scaler.min_[0], scaler.scale_[0]

    # Reshape the data
    y_pred_test = np.reshape(y_pred_test, (-1, 1))

    # Invers transform of predicted data
    y_pred_test = new_scaler.inverse_transform(y_pred_test)

    # Append predicted value to original dataset
    ds_features['predicted'] = y_pred_test

    r2, mae, mse, mape = helper.model_metrics(ds_features.y, ds_features.predicted)

    # Create dataset for test metrics
    dict_metrics = {"Metric": 'Prediction', "MAPE": mape, "MAE": mae, "MSE": mse, "R2": r2}
    df_metrics = pd.DataFrame([dict_metrics], index=[0])


    ###################################################################################
    ## Do forecasting
    ###################################################################################
    # Set start date forecast
    start_date_forecast = helper.add_seconds(end_date_test, 0)

    # Set end date forecast
    end_date_forecast = helper.add_hours(start_date_forecast, horizon)

    # Create forecasting for several hours ahead
    forecast_dates = helper.create_dates(start_date_forecast, end_date_forecast, '1H')
    forecast_dates.index.rename('datestamp', inplace=True)

    # Take features from last test sample
    X_last_test_sample = X_test_scaled[-1]  # Array shape (M, )

    # Reshape array to (1, M)
    X_last_test_sample = np.reshape(X_last_test_sample, (-1, X_last_test_sample.shape[0]))  # Convert to shape (1, M)

    # Perform one-step ahead forecasting. That is predicting data at t+1, from data at (t, t-1, t-2, .....t-N)
    y_pred_last_sample = model.predict(X_last_test_sample)

    # Set predicted value at t+1 as feature for next forecasting
    new_features = X_last_test_sample
    new_predict = y_pred_last_sample

    # Create empty list to hold forecasted value
    forecast = []

    # Loop through forecasted date, and perfom multi-step forecasting
    for i in range(0, len(forecast_dates)):
        # Insert last predicted value to feature matrix (t+i, t, t-1, ....)
        new_features = np.insert(new_features, 0, new_predict)
        # Delete last features, so the array shape not changed (shape (M,))
        new_features = np.delete(new_features, -1)
        # Reshape array to (1, M)
        new_features_reshape = np.reshape(new_features, (-1, lag + len(predictors)))
        # Do forecast
        new_predict = model.predict(new_features_reshape)
        # Insert forecast value to a list
        forecast.append(new_predict[0])

    forecast_array = np.reshape(forecast, (-1, 1))
    # Invers forecast value
    forecast_array = new_scaler.inverse_transform(forecast_array)
    # Append forecast value to dataframe
    forecast_dates['forecast'] = np.array(forecast_array)

    df_prediction = ds_features[['y', 'predicted']].copy()
    df_forecast = forecast_dates[['forecast']].copy()

    # Merge prediction and forecasting dataframe
    df_merge = pd.merge(df_prediction.reset_index(), df_forecast.reset_index(), on='datestamp', how='outer')

    # This is just a hack for Google Data Studio
    # The first of null value of forecasting value needs to be set to zero in order to be able to read
    df_merge.loc[0, 'forecast'] = 0

    # Create dictionaries to store test logs
    if len(features_column) > 0:
        all_predictors = target_column + ", " + features_column.replace(",", ", ")
    else:
        all_predictors = target_column

    dict_test = {"Number of samples": test_samples,
                 "Predictors:": all_predictors,
                 "Execution time": format_timespan(execution_time, True),
                 "Model": str(model)}

    # Merge dictionary
    log_metric = {**dict_metrics, **dict_test}

    #df_all = pd.DataFrame({"table": df_merge, "metrics":dict_metrics, "log":dict_test})
    table = json.loads(df_merge.to_json(orient='records', date_format='iso', indent=4))
    #log_metric = json.loads(pd.DataFrame.from_dict(log_metric, orient='index').to_json(orient='columns'))
    log_metric = json.loads(json.dumps(log_metric))
    #log = json.loads(pd.DataFrame.from_dict(dict_test, orient='index').to_json(orient='columns'))
    data = { "table" : table ,
             "log_metric": log_metric
           }
    result = data

    # Save json to file
    folder_target = "/var/www/html/pipeline/json"
    folder_target = folder_target + "/" + target + "/"
    target_output_file = folder_target + target + ".json"
    target_metric_file = folder_target + target + "_metric.json"

    with open(target_output_file, 'w', encoding='utf8') as json_file:
        json.dump(table, json_file, allow_nan=True)

    with open(target_metric_file, 'w', encoding='utf8') as json_file:
        json.dump(log_metric, json_file, allow_nan=True)
    ###################################################################################
    ## Create forecast metrics
    ###################################################################################
    #r2, mae, mse, mape = helper.model_metrics(df_f[target].values, forecast_dates['forecast'].values)

    # Create dataset for test metrics
    #dict_metrics = {"Metric": 'Forecasting', "MAPE": mape, "MAE": mae, "MSE": mse}
    #df_metrics = df_metrics.append([dict_metrics], ignore_index=True)

    # Save forecast and metric data to database
    save_to_database(db, forecasting_table, df_merge)
    save_to_database(db, metric_table, df_metrics)

    # Query data
    return result

def get_anomaly(target: str, startdate: str, enddate: str):
    sql = "SELECT target_column, anomaly_model FROM model_configuration WHERE target_name = '{}'".format(target)
    df_config = pd.read_sql(sql, con=engine)

    # Get target column
    target_column = df_config['target_column'].values[0]
    anomaly_model = df_config['anomaly_model'].values[0]

    # Get data
    sql = "SELECT datestamp, {} FROM hourly_log WHERE datestamp BETWEEN '{}' AND '{}'".format(target_column, startdate,
                                                                                              enddate)
    df_target = pd.read_sql(sql, con=engine)
    df_target['datestamp'] = pd.to_datetime(df_target['datestamp'])
    df_target = df_target.set_index('datestamp')
    data = df_target

    # Setup Environment for Anomaly Detection
    exp_ano101 = setup(
        data=data,
        session_id=123,
        verbose=False,
        silent=True
    )

    # Data folder
    current_dir = Path(__file__).resolve()
    current_dir_parent = current_dir.parent.parent
    model_folder = current_dir_parent / "model"
    anomaly_model = os.path.splitext(anomaly_model)[0]
    anomaly_model = model_folder / anomaly_model

    # load the model from disk
    loaded_ano_model = load_model(anomaly_model)

    # Predict anomaly
    new_data_anomaly = predict_model(loaded_ano_model, data=df_target)
    new_data_anomaly = new_data_anomaly.query('Anomaly == 1')

    table_data = json.loads(df_target.to_json(orient='records', date_format='iso', indent=4))
    anomaly_data = json.loads(new_data_anomaly.to_json(orient='records', date_format='iso', indent=4))
    data = {"table_data": table_data,
            "anomaly_data": anomaly_data
            }
    result = data

    # Save json to file
    target_output_file = target + "_data.json"
    target_metric_file = target + "_anomaly.json"

    with open(target_output_file, 'w', encoding='utf8') as json_file:
        json.dump(table_data, json_file, allow_nan=True)

    with open(target_metric_file, 'w', encoding='utf8') as json_file:
        json.dump(anomaly_data, json_file, allow_nan=True)

    # Query data
    return result

    #return json.loads(df_config.to_json(orient='records'))
    #return anomaly_model

def save_to_database(db: Session, table_name: str, df: pd.DataFrame):
    df.to_sql(table_name, db.connection(), if_exists='replace')
