@data_loader
def execute(**kwargs):
    import pandas as pd
  
    url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    df = pd.read_parquet(url)

    print(f'Loaded records: {len(df)}')  # For Question 3
    return df
