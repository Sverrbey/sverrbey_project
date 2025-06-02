import eumdac
import datetime
import shutil
import fnmatch
import requests
import time
import zipfile
import json

import sys, os
import re
from datetime import datetime
import urllib.parse

sys.path.append(os.path.dirname(os.getcwd()))
from authorisation_functions import authorisation_functions as auth
from data_tailor_support_functions import data_tailor_functions as dtf

credentials = auth.import_credentials('authorisation_functions/credentials.json')


# API base endpoint
apis_endpoint= "https://api.eumetsat.int"

# Data Store searching endpoint
service_search = apis_endpoint + "/data/search-products/os"

# Data Store downloading endpoint
service_download = apis_endpoint + "/data/download"

# Data Tailor products endpoint
service_products = apis_endpoint + "/epcs/products"

# Data Tailor chains endpoint
service_chains = apis_endpoint + "/epcs/chains"

# Data Tailor rois endpoint
service_rois = apis_endpoint + "/epcs/rois"

# Data Tailor customisations endpoint
service_customisations = apis_endpoint + "/epcs/customisations"

# Data Tailor download endpoint
service_DT_download = apis_endpoint + "/epcs/download"

# Data Tailor projections endpoint
service_projections = apis_endpoint + "/epcs/projections"

# Data Tailor formats endpoint
service_formats = apis_endpoint + "/epcs/formats"

# Data Tailor filters endpoint
service_filters = apis_endpoint + "/epcs/filters"

#Climate-quality Advanced Microwave Radiometer Level 2 Products (baseline version F06) - Sentinel-6 - Reprocessed
selected_collection_id  = 'EO:EUM:DAT:MSG:HRSEVIRI'
productID               = 'HRSEVIRI'
key                     = "hPRQifGzs_zj1hLmXFz516UrlDAa"
secret                  = "iygKOao6fr9BcD8hIkqm_SmZUhUa"


# Define our start and end dates for temporal subsetting
start_date = datetime(2021, 4, 30, 9, 0)
end_date = datetime(2021, 4, 30, 9, 15)

# Format our paramters for searching
dataset_parameters = {'format': 'json', 'pi': selected_collection_id}
dataset_parameters['dtstart'] = start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
dataset_parameters['dtend'] = end_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

# Retrieve datasets that match our filter
url = service_search
response = requests.get(url, dataset_parameters)
auth.assert_response(response)
found_data_sets = response.json()
total_data_sets = found_data_sets['totalResults']

print('Found Datasets:'+str(total_data_sets))

download_urls = []
if found_data_sets:
    for selected_data_set in found_data_sets['features']:
        product_id = selected_data_set['properties']['identifier']
        download_url = service_download + '/collections/{}/products/{}'\
              .format(urllib.parse.quote(selected_collection_id),urllib.parse.quote(product_id))
        download_urls.append(download_url)
else:
    print('No data sets found')

for download_url in download_urls:
    print(download_url)

""" Customising products with the Data Tailor API """

# check product IDs
access_token = auth.generate_token(consumer_key=key,
                                   consumer_secret=secret)

response = requests.get(service_products,
                        headers={'Authorization': 'Bearer {}'.format(access_token)})

for collection in response.json()['data']:
    print(collection['id'].ljust(20) + '\t '+collection['name'])
    if productID in collection['id'] and '_' not in collection['id']:
        productID = collection['id']

print("Our productID is:" + productID)

# Now we have the product ID, we can retrieve the available service chains"""

# List available formats
response = requests.get(service_formats, headers={'Authorization': f'Bearer {access_token}'})
#print(response.json())

# List available filters
parameters = {"product": productID}
url = dtf.build_url_string(service_filters, parameters)
response = requests.get(url, headers={'Authorization': 'Bearer {}'.format(access_token)})
print(response.json())


parameters = {"product" : productID}
url = dtf.build_url_string(service_chains, parameters)

response = requests.get(url, headers={'Authorization': 'Bearer {}'.format(access_token)})

count = 0
if response.status_code != 200:
    print(f"Error: Server returned status {response.status_code}")
    print("Response text:", response.text)
else:
    try:
        data = response.json()
        for chain in data['data']:
            count = count + 1
            print('Config ('+str(count)+'):')
            print(json.dumps(chain))
            print('\n')
            if chain['name'] == 'Projection Plate-Carree with quick-look':
                chain_config = chain
    except Exception as e:
        print("Error decoding JSON:", e)
        print("Response text:", response.text)

"""
# add new ROI to the predefined list
access_token = auth.generate_token(consumer_key=key,
                                   consumer_secret=secret)
params= {
  "id": "stryn",
  "name": "Stryn",
  "NSWE": [61.75932639, 62.06398778, 6.45811333,  7.33417667]
}

response = requests.post(service_rois,
                         headers={'Authorization': 'Bearer {}'.format(access_token),
                                  'Accept': 'application/json',
                                  'Content-Type': 'application/json'
                                 },
                         data=json.dumps(params))
"""
# get available ROIS
access_token = auth.generate_token(consumer_key=key,
                                   consumer_secret=secret)

response = requests.get(service_rois, headers={'Authorization': 'Bearer {}'.format(access_token)})

for collection in response.json()['data']:
    print(collection['id'].ljust(20) + '\t '+"".join(str(collection['NSWE'])))

# get available formats
access_token = auth.generate_token(consumer_key=key,
                                   consumer_secret=secret)

response = requests.get(service_formats, headers={'Authorization': 'Bearer {}'.format(access_token)})

for collection in response.json()['data']:
    print(collection['id'].ljust(20) + '\t '+collection['name'])


# to define you own chain
chain_config={"product": "HRSEVIRI", 
              "format": "netcdf4", 
              "filter": "hrseviri_natural_color", 
              "projection": "geographic", 
              "roi": "norway", 
              "quicklook": 
                  {"format": "netcdf4", 
                   "filter": "hrseviri_natural_color", 
                   "stretch_method": "min_max", 
                   "x_size": 500}
              }

# get available filters
access_token = auth.generate_token(consumer_key=key,
                                   consumer_secret=secret)

parameters = {"product" : productID}
url = dtf.build_url_string(service_filters, parameters)
response = requests.get(url, headers={'Authorization': 'Bearer {}'.format(access_token)})
found_filters = response.json()

print(str(found_filters['total']) + " filter(s) available for " + productID + " products :")

for collection in response.json()['data']:
    print(collection['id'].ljust(20) + '\t '+ collection['name'])

# We can then adapt the filter for the channel configuration
"""
# to change the filter for the channel configuration
chain_config['filter']= {'name': 'Natural color', 
                         'product': 'HRSEVIRI', 
                         'bands': ['channel_3', 'channel_2', 'channel_1'],  
                         'id': 'hrseviri_natural_color'}
"""

# Launch the customisation job
access_token = auth.generate_token(consumer_key=key,
                                   consumer_secret=secret)

parameters = {"product_paths" : download_urls[0],
              "chain_config" : json.dumps(chain_config),
              "access_token" : access_token}

response = requests.post(service_customisations, params=parameters, 
                         headers={'Authorization': 'Bearer {}'.format(access_token)})
jobID = response.json()['data'][0]

print('URL response: '+str(response.status_code))
print('Your job id is: '+jobID)

#Get log; this needs to be interactive, and only launch when done
status = 'RUNNING'
sleep_time = 10 # seconds
while status == 'RUNNING':
    #print(status)
    access_token = auth.generate_token(consumer_key=key,
                                        consumer_secret=secret)
    url = service_customisations+'/'+jobID
    response = requests.get(url, headers={'Authorization': 'Bearer {}'.format(access_token)})
    print(f"Polling job status: HTTP {response.status_code}")
    if response.status_code != 200:
        print("Error: Server returned status", response.status_code)
        print("Response text:", response.text)
        break
    try:
        status = response.json()[jobID]['status']
    except Exception as e:
        print("Error decoding JSON:", e)
        print("Response text:", response.text)
        break
    print('Status: ' + status)
    # After the polling loop, if status is FAILED
    if status == "FAILED":
        print("Fetching job log for details...")
        access_token = auth.generate_token(consumer_key=key,
                                        consumer_secret=secret)
        url = service_customisations + '/' + jobID + '/log'
        response = requests.get(url, headers={'Authorization': f'Bearer {access_token}'})
        print("Job log:")
        print(response.text)
    if "DONE" in status:
        print('Job successful.')
        break
    elif "ERROR" in status or 'KILLED' in status:
        print('Job unsuccessful, exiting')
        break
    elif 'QUEUED' in status:
        status = 'RUNNING'
    elif "INACTIVE" in status:
        print('Job inactive; doubling status polling time (max 10 mins)')
        sleep_time = max(60*10, sleep_time*2)
    time.sleep(sleep_time)

# Once the box above registers as <DONE>, you can check your customised products for this job
if status == 'DONE':
    access_token = auth.generate_token(consumer_key=key,
                                       consumer_secret=secret)
    url = service_customisations+'/'+jobID
    response = requests.get(url, headers={'Authorization': 'Bearer {}'.format(access_token)})
    results = response.json()[jobID]['output_products']
    for result in results:
        print(result)

# Retrieve the customised products
access_token = auth.generate_token(consumer_key=key,
                                   consumer_secret=secret)

url = service_DT_download+'?path='
download_folder = "data\satellite_data"
for result in results:
    print('Downloading: ' + result)
    response = requests.get(url+os.path.basename(result), headers={'Authorization': 'Bearer {}'.format(access_token)})
    open(os.path.join(download_folder,os.path.basename(result)), 'wb').write(response.content)
    # display images only if image available in the output products (ie no compression)
    if 'zip' not in result and 'aux' not in result:
        if 'png' in result or 'jpg' in result:
            img_output = result
print('Done!')

# Clear custimisation
access_token = auth.generate_token(consumer_key=key,
                                   consumer_secret=secret)
url = service_customisations + "/delete"
response = requests.patch(url,\
  headers={'Authorization': 'Bearer {}'.format(access_token),
  'Content-Type': 'application/json','Accept': 'application/json'},
  data=json.dumps({'uuids':jobID}))

