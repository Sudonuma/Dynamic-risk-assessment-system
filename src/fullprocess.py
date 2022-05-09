

import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os
import glob
from ingestion import merge_multiple_dataframe
import re
import pandas as pd
from sklearn.metrics import f1_score

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

# dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])
source_data_path = os.path.join(config['input_folder_path'])
ingested_data_path = os.path.join(config["output_folder_path"])


##################Check and read new data
#first, read ingestedfiles.txt
with open(os.path.join(prod_deployment_path, "ingestedfiles.txt")) as file:
    ingested_files = file.readlines()
    ingested_files = ingested_files[1:]
    ingested_files = list(map(lambda s: s.strip(), ingested_files))
    ingested_files_basenames = []
    for _ in ingested_files:
        ingested_files_basenames.append(os.path.basename(_))

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_data = []
for name in glob.glob(source_data_path + '/*.csv'):
    source_data.append(os.path.basename(name))

check_new = set(source_data) != set(ingested_files_basenames)



##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if check_new:
    merge_multiple_dataframe()



##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(prod_deployment_path, "latestscore.txt")) as file:
    latest_score = re.findall("\d+\.\d+", file.read())
    latest_score = float(latest_score[0])

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
dataset = pd.read_csv(os.path.join(ingested_data_path, 'finaldata.csv'))
target = dataset.pop('exited')
X = dataset.drop(['corporation'], axis=1)

prediction = diagnostics.model_predictions(X)
new_score = f1_score(prediction, target)
print(new_score)
if (new_score < latest_score):
    training.train_model()
    scoring.score_model()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
    deployment.store_model_into_pickle()
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
    
    reporting.score_model()

    # diagnostics
    os.system("python apicalls.py")





