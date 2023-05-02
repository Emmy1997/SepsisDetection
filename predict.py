import json
import pandas as pd
import os
import sys
import pickle
from preprocess_data import PreProcess
import numpy as np


def main():
    args = sys.argv[1:]
    path = args[0]
    X_test = []
    patient_list = []
    for file in os.listdir(f"{path}/"):
        df = pd.read_csv(f"{path}/{file}", sep='|')
        p_name = file.split('/')[-1].split('.')[0]
        patient_list.append(p_name)
        sepsislabel = list(df['SepsisLabel'].values)
        y = 1 if np.sum(sepsislabel) > 0 else 0
        df['patient'] = p_name
        df['timestamp'] = np.arange(df.shape[0])
        df['y'] = [y for _ in range(df.shape[0])]
        X_test.append(df)

    test_df = pd.concat(X_test).reset_index(drop=True)
    pre_obj_val = PreProcess(test_df, sample=False)
    train_pipe_line_dict = json.load(open(f'train_pipe_line_dict.json', 'r'))
    train_df = pd.read_csv('train_df.csv')
    test_df_transformed = pre_obj_val.run_pipeline(train=False, train_df=train_df, pipeline_dict=train_pipe_line_dict)
    test_df_transformed_clean = test_df_transformed.drop(['patient', 'y'])

    model = pickle.load(open("final_model.pkl", "rb"))
    y_pred = model.predict(test_df_transformed_clean)

    df = pd.DataFrame({'id': patient_list, 'prediction': y_pred})
    df.to_csv('prediction.csv', index=False)

if __name__ == '__main__':
    main()