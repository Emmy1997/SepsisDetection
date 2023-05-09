import json
import pandas as pd
import os
import sys
import pickle
from preprocess_data import PreProcess
import numpy as np
from sklearn.metrics import f1_score


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
    val_df = pd.read_csv('val_df.csv')
    train_val_df =  pd.concat([train_df, val_df], axis=0)
    test_df_transformed = pre_obj_val.run_pipeline(train=False, train_df=train_val_df, pipeline_dict=train_pipe_line_dict)
    y_true = test_df_transformed[['y']].values.ravel()
    test_df_transformed_clean = test_df_transformed.drop(['patient', 'y', 'index'])

    model = pickle.load(open("xgb_model.pickle", "rb"))
    y_pred = model.predict(test_df_transformed_clean)
    score = f1_score(y_true, y_pred)
    print(f"F1 score is :{score}")
    df = pd.DataFrame({'id': patient_list, 'prediction': y_pred})
    df.to_csv('prediction.csv', index=False)

if __name__ == '__main__':
    main()