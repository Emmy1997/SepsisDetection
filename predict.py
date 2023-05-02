import json

import pandas as pd
import os
import numpy as np
import sys
import pickle
from preprocess_data import PreProcess


def main():
    args = sys.argv[1:]
    path = args[0]
    X_test = []
    patient_list = []
    for file in os.listdir(f"{path}/"):
        patient_test = pd.read_csv(f"{path}/{file}", sep='|')
        patient_list.append(file)
        X_test.append(patient_test)

    test_df = pd.concat(X_test).reset_index(drop=True)
    pre_obj_val = PreProcess(test_df, sample=False)
    train_pipe_line_dict = json.load(open(f'train_pipe_line_dict.json', 'r'))
    train_df = pd.read_csv('train_df.csv')
    test_df_transformed = pre_obj_val.run_pipeline(train=False, train_df=train_df, pipeline_dict=train_pipe_line_dict)
    test_df_transformed_clean = test_df_transformed.drop(['patient', 'y'])

    model = pickle.load(open("final_model.pkl", "rb"))
    y_pred = model.predict(test_df_transformed_clean)

    patient_list_id = [int(i.replace('patient_', '').replace('.psv', '')) for i in patient_list]

    df = pd.DataFrame({'Id': patient_list_id, 'SepsisLabel': y_pred})
    df.to_csv('prediction.csv', index=False)

if __name__ == '__main__':
    main()