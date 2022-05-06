import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path'])




##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    
    # call model prediction function from diagnostics.py to obtain a list of predicted values
    # this function will use test data
    # compute prediction values

    # load test dataset
    test_data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    target = test_data.pop('exited')

    # load model
    # model = pickle.load(open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb'))
    
    X = test_data.drop(['corporation'], axis=1)
    prediction = model_predictions(X)
    cf_matrix = confusion_matrix(target, prediction)
    # print(cf_matrix)
    sns.heatmap(cf_matrix, annot=True)
    plt.savefig("cf_matrix.png")
    # plot confusion matrix

    
    




if __name__ == '__main__':
    score_model()
