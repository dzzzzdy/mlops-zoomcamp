import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
from pathlib import Path
from prefect import flow, task

@task(retries=3, retry_delay_seconds=2, log_prints=True)
def read_dataframe(filename):
    df = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet")
    print(f"There are {df.shape[0]} records.")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    print(f"There are {df.shape[0]} records after data preprocessing.")
    return df

@task(retries=3, retry_delay_seconds=2)
def vectorize_features(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical+numerical].to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

@task(retries=3, retry_delay_seconds=2,log_prints=True)
def train(X_train, y_train, dv):
    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-experiment")
    with mlflow.start_run() as run:
        model = LinearRegression()
        model.fit(X_train, y_train)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.sklearn.log_model(model, artifact_path="models")
        print(model.intercept_)
    return model.intercept_, run.info.run_id

@task
def register_model(run_id):
    model_uri = f"runs:/{run_id}/models"

    print(f"Registering model from: {model_uri}")
    registered_model = mlflow.register_model(model_uri=model_uri, name="nyc_taxi_production")


@flow
def run():
    file_name = "data/yellow_tripdata_2023-03.parquet"
    df = read_dataframe(file_name)
    X_train, dv = vectorize_features(df)
    target = 'duration'
    y_train = df[target].values
    intercept, run_id = train(X_train, y_train, dv)
    register_model(run_id)


if __name__ == "__main__":
    run()







    