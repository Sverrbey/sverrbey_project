import csv
import getopt
import json
import sys
import pandas as pd
from urllib.request import Request, urlopen  # Python 3


def get_observations(argv, stnrs, filepath=None):
    try:
        opts, args = getopt.getopt(argv, "a:s:p:r:ht:")
    except getopt.GetoptError as err:
        print(str(err))  # will print something like "option -a not recognized"
        sys.exit(2)

    station = None
    parameter = None
    resolution_time = None
    api_key = None
    reference_time = None

    for opt, arg in opts:
        if opt == "-s":
            station = arg
        elif opt == "-p":
            parameter = arg
        elif opt == "-r":
            resolution_time = arg
        elif opt == "-a":
            api_key = arg
        elif opt == "-t":
            reference_time = arg
        else:
            assert False, "unhandled option"


    baseurl = "https://hydapi.nve.no/api/v1/Observations?StationId={station}&Parameter={parameter}&ResolutionTime={resolution_time}"

    url = baseurl.format(station=station, parameter=parameter,
                         resolution_time=resolution_time)

    if reference_time is not None:
        url = "{url}&ReferenceTime={reference_time}".format(
            url=url, reference_time=reference_time)

    request_headers = {
        "Accept": "application/json",
        "X-API-Key": api_key
    }

    request = Request(url, headers=request_headers)

    response = urlopen(request)
    content = response.read().decode('utf-8')

    parsed_result = json.loads(content)
    df_insitu = pd.DataFrame()
    st_ids = stnrs.split(',')
    for station in range(len(parsed_result["data"])):
        observations = parsed_result["data"][station]["observations"]
        df_obs = pd.DataFrame(observations)
        # Convert 'time' to datetime and format as standard string
        df_obs['time'] = pd.to_datetime(df_obs['time'], utc=True)
        df_obs['time'] = df_obs['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_obs = df_obs.set_index('time')[['value']]
        df_insitu[st_ids[station]] = df_obs
    df_insitu.to_json(filepath, orient='index')

if __name__ == "__main__":
    get_observations(sys.argv[1:])