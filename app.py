from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis 
# import predict_exited_from_saved_model
import json
import os
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


@app.route('/')
def index():
    user = request.args.get('user')
    return "Hello " + user

#######################Prediction Endpoint
@app.route("/prediction", methods=['GET', 'POST'])
def predict():        
    filename = request.args.get('filename')
    thedata=pd.read_csv(filename)
    thedata = thedata.drop(['corporation', 'exited'], axis=1)

    #call the prediction function you created in Step 3
    predictions = model_predictions(thedata)
    return jsonify(predictions.tolist())

######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    #check the score of the deployed model
    f1 = score_model()
    
    return str(f1)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def statistics():        
    #check means, medians, and modes for each column
    stat = dataframe_summary()
    return stat

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    timing = execution_time()
    NApercentage = missing_data()
    dependancies = outdated_packages_list()

    diagnostics = {'execution_time': timing,
                    'NApercentage': NApercentage,
                    'dep': dependancies}
    return str(diagnostics)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
