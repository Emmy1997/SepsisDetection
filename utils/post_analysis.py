import json
import pandas as pd
import os
import sys
import pickle

from matplotlib import pyplot as plt

from preprocess_data import PreProcess
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
# import shap


def main():
    pass
    # test_df_transformed_clean = pd.read_csv("../test_df_transformed.csv")
    # model = pickle.load(open("../xgb_model.pickle", "rb"))
    # cols = model.get_booster().feature_names
    # # cols_to_drop = ['patient', 'y', 'index', "level_0", 'SepsisLabel']
    # # cols_to_drop = list(set(test_df_transformed_clean.columns) & set(cols_to_drop))
    # # test_df_transformed_clean = test_df_transformed_clean.drop(columns=cols_to_drop)
    # y_pred = model.predict(test_df_transformed_clean[cols])
    # # Fits the explainer
    # explainer = shap.Explainer(model.predict, test_df_transformed_clean[cols], approximate=True, nsamples=100)
    # # Calculates the SHAP values - It takes some time
    #
    # shap_values = explainer(test_df_transformed_clean[cols])
    #
    # # Dump SHAP values to a pickle file
    # with open('shap_values.pickle', 'wb') as handle:
    #     pickle.dump(shap_values, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ## load shap
    # Load SHAP values from a pickle file
    # with open('shap_values.pickle', 'rb') as handle:
    #     shap_values = pickle.load(handle)
    # # Plot SHAP values with larger figure size
    # fig = plt.figure()
    # shap.plots.bar(shap_values, show=False)
    # plt.gcf().set_size_inches(20, 6)
    # plt.savefig("shap_values.jpg")
    # plt.show()
    #
    # with open("shap_value.txt", "w") as file:
    #     for value in shap_values:
    #         file.write(str(value) + "\n")
    #
    # with open("y_pred.txt", "w") as file:
    #     for value in y_pred:
    #         file.write(str(value) + "\n")


if __name__ == '__main__':
    main()