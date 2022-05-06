
from traceback import print_tb
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

##################Function to get model predictions
def model_predictions(X):
    #read the deployed model and a test dataset, calculate predictions
    
    # read test dataset
    # X = pd.read_csv(os.path.join(test_data_path, 'testdata.csv')) 
    # X = X.drop(['corporation'], axis=1)
    # X = X.drop(['exited'], axis=1)
    # load model from the deployment directory
    model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))

    # compute predictions
    predictions = model.predict(X)
    return predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    # read data
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    df = df.drop(['exited'], axis=1)
    df = df.drop(['corporation'], axis=1)

    stats = {}
    for column in df.columns:
        mean = df[column].mean()
        median = df[column].median()
        stdv = df[column].std()
        # stats.append()
        stats[column] = {'mean': mean, 'median': median, 'stdv': stdv}

    return stats

def missing_data():
    # compute percentage of missing values for each column
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    missing_data = df.isna().sum()
    percentage = missing_data / df.shape[0] *100
    return percentage

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py

    ingestion_time = []
    training_time = []
    for _  in range(20):

        starttime = timeit.default_timer()
        os.system('python3 ingestion.py')
        timing = timeit.default_timer() - starttime
        ingestion_time.append(timing)

        starttime = timeit.default_timer()
        os.system('python3 training.py')
        timing = timeit.default_timer() - starttime
        training_time.append(timing)
    
    mean_of_ingest_time = sum(ingestion_time) / len(ingestion_time)
    mean_of_train_time = sum(training_time) / len(training_time)
    
    return [mean_of_ingest_time, mean_of_train_time]

##################Function to check dependencies
def outdated_packages_list():
    #get a list of outdated packages
    # dependencies = subprocess.check_output(
    #     ['pip-outdated', 'requirements.txt']
    #     )
    # dependencies = subprocess.check_output(
    #     ['pip', 'list', '--outdated']
    #     )

    dependencies = subprocess.run(
        ['pip-outdated','requirements.txt'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8')

    dep = dependencies.stdout
    return dep

if __name__ == '__main__':
    X = pd.read_csv(os.path.join(test_data_path, 'testdata.csv')) 
    X = X.drop(['corporation'], axis=1)
    X = X.drop(['exited'], axis=1)
    # model_predictions()
    # dataframe_summary()
    # missing_data()
    # print(type(execution_time()))
    outdated_packages_list()





    
