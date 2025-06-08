import pandas as pd

url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
df = pd.read_parquet(url, engine="pyarrow")

print("原始行数 (Q3):", len(df))

df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df['duration'] = df.duration.dt.total_seconds() / 60
df = df[(df.duration >= 1) & (df.duration <= 60)]

print("清洗后行数 (Q4):", len(df))