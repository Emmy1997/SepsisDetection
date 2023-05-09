import json
import pandas as pd
import os
import sys
import pickle
from preprocess_data import PreProcess
import numpy as np
from sklearn.metrics import f1_score
import re

def main():
    args = sys.argv[1:]
    path = args[0]
    X_test = []
    patient_list = []
    files = sorted([(int(re.findall(r'\d+', s)[-1]), s) for s in os.listdir(path)])
    files = [s[1] for s in files]
    for file in files:
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
    ## filter only first 1
    ## taking only the relevant rows (6 hours before the sepsis occured)
    healthy_df = test_df[test_df['y'] == 0]
    sick_df1 = test_df[test_df['y'] == 1]
    sick_df_no_sepsis = sick_df1[sick_df1['SepsisLabel'] == 0]
    sick_df_sepsis = sick_df1[sick_df1['SepsisLabel'] == 1]
    sick_df_sepsis = sick_df_sepsis.groupby('patient').apply(
        lambda x: x[x['timestamp'] == x['timestamp'].min()]).reset_index(drop=True)
    sick_df_for_training = pd.concat([sick_df_no_sepsis, sick_df_sepsis])
    sick_df_training = sick_df_for_training.sort_values(['patient', 'timestamp'])
    test_df_final = pd.concat([sick_df_training, healthy_df], axis=0).reset_index(drop=True)
    test_df_final = test_df_final.sort_values(['patient', 'timestamp'])
    ###### preprocess test based on pipeline
    pre_obj_val = PreProcess(test_df_final, sample=False)
    train_pipe_line_dict = json.load(open(f'train_pipeline_dict.json', 'r'))
    train_df = pd.read_csv('train_df_filtered.csv')
    test_df_transformed = pre_obj_val.run_pipeline(train=False, train_df=train_df, pipeline_dict=train_pipe_line_dict)
    test_df_transformed.to_csv("test_df_transformed.csv")
    ### start prediction
    y_true = test_df_transformed['y'].values.ravel()
    cols_to_drop = ['patient', 'y', 'index', "level_0", 'SepsisLabel']
    patient_list = test_df_transformed['patient'].values
    cols_to_drop = list(set(test_df_transformed.columns) & set(cols_to_drop))
    test_df_transformed_clean = test_df_transformed.drop(columns=cols_to_drop)
    model = pickle.load(open("xgb_model.pickle", "rb"))
    cols = model.get_booster().feature_names
    y_pred = model.predict(test_df_transformed_clean[cols])
    ## compute result
    score = f1_score(y_true, y_pred)
    print(f"F1 score is :{score}")
    df = pd.DataFrame({'id': patient_list, 'prediction': y_pred})
    df.to_csv('prediction.csv', index=False)

if __name__ == '__main__':
    main()