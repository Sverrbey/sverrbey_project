import openmeteo_requests
from openmeteo_sdk.Variable import Variable
import pandas as pd
import requests_cache
from retry_requests import retry
import asyncio

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
def get_UKMO_data(params):

	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Elevation {response.Elevation()} m asl")
	print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
	print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["temperature_2m"] = hourly_temperature_2m
	hourly_data["precipitation"] = hourly_precipitation

	hourly_dataframe = pd.DataFrame(data = hourly_data)
	#print(hourly_dataframe)

	# Process daily data only if "daily" is present in params and contains values
	daily_dataframe = None
	if "daily" in params and params["daily"]:
		# Process daily data. The order of variables needs to be the same as requested.
		daily = response.Daily()
		daily_precipitation_sum = daily.Variables(0).ValuesAsNumpy()

		daily_data = {"date": pd.date_range(
			start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
			end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
			freq = pd.Timedelta(seconds = daily.Interval()),
			inclusive = "left"
		)}

		daily_data["precipitation_sum"] = daily_precipitation_sum

		daily_dataframe = pd.DataFrame(data = daily_data)
	#print(daily_dataframe)

	# Write the dataframes to JSON files
	hourly_file_path = f"data/satellite_data/UKMO_hourly_data_{params['longitude']}_{params['latitude']}.json"
	hourly_dataframe.to_json(hourly_file_path, orient='values')

	daily_file_path = f"data/satellite_data/UKMO_daily_data_{params['longitude']}_{params['latitude']}.json"
	if "daily" in params:
		if daily_dataframe is not None:
			daily_dataframe.to_json(daily_file_path, orient='values')

	return hourly_dataframe, daily_dataframe

# Basic Usage

async def main():
    om = openmeteo_requests.AsyncClient()
    params = {
        "latitude": 52.54,
        "longitude": 13.41,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
        "current": ["temperature_2m", "relative_humidity_2m"]
    }

    responses = await om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Current values
    current = response.Current()
    current_variables = list(map(lambda i: current.Variables(i), range(0, current.VariablesLength())))
    current_temperature_2m = next(filter(lambda x: x.Variable() == Variable.temperature and x.Altitude() == 2, current_variables))
    current_relative_humidity_2m = next(filter(lambda x: x.Variable() == Variable.relative_humidity and x.Altitude() == 2, current_variables))

    print(f"Current time {current.Time()}")
    print(f"Current temperature_2m {current_temperature_2m.Value()}")
    print(f"Current relative_humidity_2m {current_relative_humidity_2m.Value()}")

#asyncio.run(main())