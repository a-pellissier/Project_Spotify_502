from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

@app.get("/")
def hello():
    return {"key": "value"}

@app.get("/predict_fare/{pickup_datetime}/{pickup_longitude}/{pickup_latitude}/{dropoff_longitude}/{dropoff_latitude}/{passenger_count}/")
def predict_fare(pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count):
    X_pred =  pd.DataFrame({
        'key': ['2013-07-06 17:18:00.000000119'],
        "pickup_datetime": [pickup_datetime],
        "pickup_longitude": [float(pickup_longitude)],
        "pickup_latitude": [float(pickup_latitude)],
        "dropoff_longitude": [float(dropoff_longitude)],
        "dropoff_latitude": [float(dropoff_latitude)],
        "passenger_count": [int(passenger_count)]})

    X_pred[['passenger_count']].astype('int8', copy=False)
    
    model = joblib.load('model.joblib')
    prediction = model.predict(X_pred)
    result = dict(prediction=prediction.item(0))

    return result