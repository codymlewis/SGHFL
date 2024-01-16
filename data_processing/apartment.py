"""
Smart* Data Set for Sustainability https://traces.cs.umass.edu/index.php/Smart/Smart
"""
import pandas as pd
import os
import requests
import io
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import logging
import tarfile
from safetensors.numpy import save_file


def download_and_extract(dload_url, extract_folder):
    logging.info(f"Downloading raw data files from {dload_url}...")
    r = requests.get(dload_url)
    tar = tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz")
    tar.extractall(extract_folder)
    tar.close()
    logging.info("Raw data files have been saved to ../data")


def download():
    os.makedirs("../data/", exist_ok=True)
    logging.info("Processing weather data...")
    download_and_extract("https://lass.cs.umass.edu/smarttraces/2017/apartment-weather.tar.gz", "../data/")
    weather_2014 = pd.read_csv("../data/apartment-weather/apartment2014.csv")
    weather_2015 = pd.read_csv("../data/apartment-weather/apartment2015.csv")
    weather_2016 = pd.read_csv("../data/apartment-weather/apartment2016.csv")
    weather = pd.concat([weather_2014, weather_2015, weather_2016])
    weather = weather.drop(columns=["icon", "summary"])
    weather = SimpleImputer(missing_values=pd.NA, strategy='mean').set_output(transform="pandas").fit_transform(weather)
    weather['date'] = pd.to_datetime(weather.time, unit='s')
    weather = weather.drop(columns="time")

    logging.info("Processing electrical data...")
    download_and_extract("https://lass.cs.umass.edu/smarttraces/2017/apartment-electrical.tar.gz", "../data")
    years = [2015, 2016]

    for year in years:
        data_folder = f"../data/apartment/{year}"
        csv_files = [f for f in os.listdir(data_folder) if ".csv" in f]
        client_data = {}

        for csv_file in tqdm(csv_files):
            df = pd.read_csv(f"{data_folder}/{csv_file}", header=None)
            df.columns = ["date", "consumption"]
            df.date = pd.to_datetime(df.date).dt.floor('h')
            df = df.groupby("date").mean().reset_index()
            full_df = pd.merge(df, weather[weather.date.isin(df.date)])
            client_id = csv_file[3:csv_file.find("_")]
            client_data[client_id] = full_df.drop(columns="date").to_numpy()

        data_fn = f"../data/apartment_{year}.safetensors"
        save_file(client_data, data_fn)
        logging.info(f"Data saved to {data_fn}")


if __name__ == "__main__":
    download()
