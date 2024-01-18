"""
Solar home dataset https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
"""
from datetime import datetime
import os
import io
import zipfile
import pandas as pd
import numpy as np
import requests
import json
import sklearn.cluster as skc
import logging
from tqdm import tqdm
from safetensors.numpy import save_file


def download_and_extract(dload_url, extract_folder):
    logging.info(f"Downloading raw data files from {dload_url}...")
    r = requests.get(dload_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(extract_folder)
    logging.info("Raw data files have been saved to ../data")


def combine_half_hour(data):
    time_columns = data.columns[4:]
    for half_hour, hour in zip(time_columns[::2], time_columns[1::2]):
        data[hour] += data[half_hour]
    data = data.drop(columns=time_columns[::2])
    return data


def to_long_form(data):
    id_columns = data.columns[:4].tolist()
    time_columns = data.columns[4:]
    data = pd.melt(data, id_vars=id_columns, value_vars=time_columns)
    return data


def format_data(data, value_name):
    data = data.rename(columns={"value": value_name, "date": "Datetime"})
    data['Datetime'] += " " + data['variable']
    data = data.drop(columns='variable')

    def to_datetime(dt_str):
        if dt_str.find('-') >= 0:
            return datetime.strptime(dt_str, "%d-%b-%y %H:%M").isoformat(timespec="minutes")
        if "/" in dt_str[:2]:
            dt_str = "0" + dt_str
        if dt_str[-5] == " ":
            dt_str = dt_str[:-4] + "0" + dt_str[-4:]
        return datetime.strptime(dt_str, "%d/%m/%Y %H:%M").isoformat(timespec="minutes")

    data['Datetime'] = data['Datetime'].map(to_datetime)
    return data


def process_data(filename):
    if "2010-2011" in filename:
        year_range = "2010-2011"
        start_date = "2010-07-01"
        end_date = "2011-06-30"
    else:
        year_range = "2011-2012"
        start_date = "2011-07-01"
        end_date = "2012-06-30"
    logging.info(f"Processing {year_range} data")

    solar_home_data = pd.read_csv(filename, header=1)

    # Each customer has either two or three types of recordings for each day of the year.
    # The recordings correspond to the consumption categories.
    # We convert them to net energy
    has_cl = solar_home_data.Customer.unique()[[
        ((solar_home_data.Customer == c) & (solar_home_data['Consumption Category'] == "CL")).any()
        for c in solar_home_data.Customer.unique()
    ]]
    time_columns = solar_home_data.columns[5:]
    gc_data = solar_home_data[
        (solar_home_data['Consumption Category'] == "GC") & solar_home_data.Customer.isin(has_cl)
    ].reset_index(drop=True)
    cl_data = solar_home_data[
        (solar_home_data['Consumption Category'] == "CL") & solar_home_data.Customer.isin(has_cl)
    ].reset_index(drop=True)
    gg_data = solar_home_data[
        (solar_home_data['Consumption Category'] == "GG") & solar_home_data.Customer.isin(has_cl)
    ].reset_index(drop=True)

    gc_data.loc[:, time_columns] = cl_data[time_columns] + gc_data[time_columns]
    consumption_data = gc_data.drop(columns=["Consumption Category"])
    generation_data = gg_data.drop(columns=["Consumption Category"])
    gc_data = solar_home_data[
        (solar_home_data['Consumption Category'] == "GC") & ~solar_home_data.Customer.isin(has_cl)
    ].reset_index(drop=True)
    gg_data = solar_home_data[
        (solar_home_data['Consumption Category'] == "GG") & ~solar_home_data.Customer.isin(has_cl)
    ].reset_index(drop=True)
    consumption_data = pd.concat((consumption_data, gc_data.drop(columns=["Consumption Category"])))
    generation_data = pd.concat((generation_data, gg_data.drop(columns=["Consumption Category"])))

    # Get rid of half hour intervals
    consumption_data = combine_half_hour(consumption_data)
    generation_data = combine_half_hour(generation_data)
    # Make long form
    consumption_data = to_long_form(consumption_data)
    generation_data = to_long_form(generation_data)
    # Clean up the formatting of the rows and save the processed datase
    consumption_data = format_data(consumption_data, "Consumed Energy")
    generation_data = format_data(generation_data, "Generated Energy")
    solar_home_data = consumption_data.merge(generation_data)
    solar_home_data.to_csv(f"../data/solar_home_{year_range}.csv", index=False)

    # Get postal data for using the weather data
    if not os.path.exists("../data/AU.txt"):
        download_and_extract("https://download.geonames.org/export/zip/AU.zip", '../data')
    postcode_data = pd.read_csv(
        "../data/AU.txt",
        sep='\t',
        names=[
            "country code",
            "postal code",
            "place name",
            "admin name1",
            "admin code1",
            "admin name2",
            "admin code2",
            "admin name3",
            "admin code3",
            "latitude",
            "longitude",
            "accuracy"
        ]
    )
    postcode_data = postcode_data.drop(postcode_data[postcode_data['postal code'].duplicated()].index)

    # Then we download the weather data from https://open-meteo.com and categorise by postcode
    if os.path.exists(f"../data/weather_{year_range}.json"):
        with open(f"../data/weather_{year_range}.json", "r") as f:
            weather_data = json.load(f)
    else:
        weather_data = {
            "structure": {"postcode": {"time": ["temperature", "precipitation", "cloudcover"]}},
            "hourly units": {'time': 'iso8601', 'temperature': 'Celcius', 'precipitation': 'mm', 'cloudcover': '%'},
        }
        logging.info("Getting weather data...")
        for postcode in (pbar := tqdm(solar_home_data.Postcode.unique())):
            pbar.set_postfix_str(f"{postcode=}")
            postcode_row = postcode_data[postcode == postcode_data['postal code']]
            latitude, longitude = postcode_row.latitude.item(), postcode_row.longitude.item()
            r = requests.get(f"https://archive-api.open-meteo.com/v1/era5?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,precipitation,cloudcover&timezone=Australia%2FSydney")
            if r.ok:
                postcode_weather = r.json()
                weather_data[str(postcode)] = {
                    time: other_data for time, *other_data in zip(*[v for v in postcode_weather['hourly'].values()])
                }
            else:
                tqdm.write(f"Postcode {postcode} weather data request failed")
        with open(f"../data/weather_{year_range}.json", 'w') as f:
            json.dump(weather_data, f)
        logging.info(f"weather data saved to ../data/weather_{year_range}.json")

    solar_home_data = solar_home_data[solar_home_data.Customer != 300]
    # For the final steps of processing, we first divide the clients into regions based on postcode
    # if os.path.exists("../data/customer_regions.json"):
    if False:
        with open("../data/customer_regions.json", 'r') as f:
            customer_region_map = json.load(f)
    else:
        postcodes = solar_home_data.Postcode
        clusters = skc.KMeans(10, n_init='auto', random_state=42).fit_predict(postcodes.to_numpy().reshape(-1, 1))
        postcode_region_map = {postcode: i.item() for postcode, i in zip(postcodes, clusters)}
        customers = solar_home_data.Customer.unique()
        customer_region_map = {}
        for customer in customers:
            region = postcode_region_map[solar_home_data.query(f"`Customer` == {customer}").Postcode.unique().item()]
            customer_region_map[str(customer)] = region
        with open("../data/customer_regions.json", 'w') as f:
            json.dump(customer_region_map, f)

    # Then we sort and format the rest of the data for use in the machine learning model
    sorted_full_data = {}
    for customer in tqdm(solar_home_data.Customer.unique()):
        customer_data = solar_home_data.query(f"`Customer` == {customer}")
        postcode_weather_data = weather_data[str(customer_data.Postcode.unique().item())]
        net_energy_data = {
            time: [con_energy, gen_energy]
            for time, con_energy, gen_energy
            in zip(customer_data.Datetime, customer_data["Consumed Energy"], customer_data["Generated Energy"])
        }

        idx = np.argsort([datetime.fromisoformat(k).timestamp() for k in net_energy_data.keys()])
        sorted_energy = np.array(list(net_energy_data.values()))[idx]
        idx = np.argsort([datetime.fromisoformat(k).timestamp() for k in postcode_weather_data.keys()])
        sorted_weather_data = np.array(list(postcode_weather_data.values()))[idx]

        sorted_full_data[str(customer)] = np.concatenate((sorted_energy.reshape(-1, 2), sorted_weather_data), axis=1)

    data_fn = f"../data/solar_home_{year_range}.safetensors"
    save_file(sorted_full_data, data_fn)
    logging.info(f"Full data saved to {data_fn}")


def download():
    os.makedirs("../data/", exist_ok=True)
    data_urls = [
        "https://cdn.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2010-to-30-June-2011.zip?rev=3ba8aee669294858a27cda3f2214aba5",
        "https://cdn.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2011-to-30-June-2012.zip?rev=938d7e42fe0f43969fc4144341dacfac",
    ]
    raw_data_fns = [
        "../data/2010-2011 Solar home electricity data.csv", "../data/2011-2012 Solar home electricity data v2.csv"
    ]
    for data_url, filename in zip(data_urls, raw_data_fns):
        if not os.path.exists(filename):
            download_and_extract(data_url, "../data")
        process_data(filename)


if __name__ == "__main__":
    download()
