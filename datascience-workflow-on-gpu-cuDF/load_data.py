import cudf

taxi_data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"

df_gpu = cudf.read_parquet(taxi_data_url)
df_gpu.info()
df_gpu.write_parquet("data/yellow_tripdata_2023-01.parquet")
