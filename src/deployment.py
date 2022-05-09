import pickle
import os
import sys
import shutil
import json




##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])



####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    #copy ingestedfiles.txt 
    shutil.copy(os.path.join(dataset_csv_path, 'ingestedfiles.txt'), prod_deployment_path)

    #copy model 
    shutil.copy(os.path.join(model_path, 'trainedmodel.pkl'), prod_deployment_path)

    #copy latestscore.txt
    shutil.copy(os.path.join(model_path, 'latestscore.txt'), prod_deployment_path)

if __name__ == '__main__':
    store_model_into_pickle()


