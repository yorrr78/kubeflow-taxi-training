from google.cloud import aiplatform
import json
import pandas as pd
import numpy as np
import os
import requests
import logging
import tqdm


region = "asia-northeast3"
url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/'
file_name = 'green_tripdata_2021-01.parquet'
endpoint_resource_name = 'projects/616906371504/locations/asia-northeast3/endpoints/4598307160940609536'


def download_to_local(download_url, file):
    """Download the target file from internet to local"""

    if os.path.isfile(f'./data/{file}'):
        logging.info('Already exist')
        pass

    else:
        if not os.path.exists('data'):
            os.mkdir('data')

        file_url = download_url + file
        response = requests.get(file_url, stream=True)
        logging.info(f'downloading.. {file}')

        with open(f'./data/{file}', 'wb') as f_in:
            for chunk in tqdm(response.iter_content()):
                f_in.write(chunk)
        logging.info('Download finished!')
    
    return f'./data/{file}'
        
    
def preprocessing(df):
    columns = [
        'PULocationID',
        'DOLocationID',
        'lpep_dropoff_datetime',
        'lpep_pickup_datetime',
        'trip_distance',
    ]
    df = df[columns]

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)] # Get rid of outliers

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df
        
    
def prepare_dictionaries(df: pd.DataFrame):
    """Composite"""
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts
    

def chunk_data(data, chunk_size):
    """Chunk input data into smaller batches."""
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]


def get_predictions(endpoint_resource_name, instances):
    """Get predictions from online serving model."""
    client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    responses = []
    for instance_batch in instances:
        response = client.predict(
            endpoint=endpoint_resource_name, 
            instances=instance_batch
        )
        responses.append(response)
    
    return responses


def upload_to_gcs(prediction_file, bucket_name, prediction_blob):
    """Upload the file to Google Cloud Storage"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(prediction_blob)
    blob.upload_from_filename(prediction_file)

    print(f"Predictions uploaded to gs://{bucket_name}/{prediction_blob} successfully!")
    

def run():
    # Download input data & processing for predictions
    file_loc = download_to_local(url, file_name)
    df = pd.read_parquet(file_loc)
    print(len(df))
    
    # Preprocess data and chunk it into smaller batches
    # Because one month file is too large for Vertex AI onlien prediction
    df = preprocessing(df)
    input_data = prepare_dictionaries(df)
    chunk_size = 1000 
    
    # Chunk the input data and make predictions for each batch
    input_data_batches = chunk_data(input_data, chunk_size)
    responses = get_predictions(endpoint_resource_name, input_data_batches)
    
    for response in responses:
        response.predictions
        
    print(len(responses))
#     for response in responses:
#         print(response.predictions)

#     # Upload to Google Cloud Storage
#     yorrr78-dev-111111-mlops-bucket/predictions/batch
    
    
if __name__ == '__main__':
    run()