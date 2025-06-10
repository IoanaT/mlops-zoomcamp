import pandas as pd

@data_loader
def execute(**kwargs):
    train_year, train_month = 2025, 2
    val_year, val_month = 2025, 3

    def read_dataframe(year, month):
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
        df = pd.read_parquet(url)
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df['duration'] = df['duration'].dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
        return df

    df_train = read_dataframe(train_year, train_month)
    df_val = read_dataframe(val_year, val_month)

    return {
        'df_train': df_train,
        'df_val': df_val
    }
