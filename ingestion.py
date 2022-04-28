import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob



#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file

    df = pd.DataFrame()
    file_names = []
    for file in glob.glob(input_folder_path+'/*.csv'):
        temp_df = pd.read_csv(file)
        
        # save filenmaes
        file_names.append(file)
        # merge datasets
        df = df.append(temp_df, ignore_index=True)
        
    # drop duplicates
    df = df.drop_duplicates().reset_index(drop=1)


    # save ingested data
    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)
    
    # save ingested filenames to ingestedfiles.txt
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as file:
        file.write(
            f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("\n".join(file_names))


if __name__ == '__main__':
    merge_multiple_dataframe()
