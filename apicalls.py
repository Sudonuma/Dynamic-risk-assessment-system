import requests
import os
import json

#Specify a URL that resolves to your workspace

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['test_data_path']) 

URL = "http://127.0.0.1:8000/"

# Call each API endpoint and store the responses
response1 = requests.get(URL+'/prediction?filename=data.csv').content
# response1 = requests.post(
#     f'{URL}/prediction',
#     json={
#         'filename': './testdata/testdata.csv'}).text


response2 = requests.get(URL+'/scoring').content
response3 = requests.get(URL+'/summarystats').content
response4 = requests.get(URL+'/diagnostics').content

#combine all API responses
responses = [response1, response2, response3, response4]

#write the responses to your workspace

with open(os.path.join(os.getcwd(), config['output_model_path'], "apireturns.txt"), "w") as f:
    f.write(str(responses))