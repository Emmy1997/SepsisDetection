import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set the directory path
directory_path = r"C://Users//ASUS//Downloads//data//train"

# create an empty list to hold the dataframes
dfs = []

# loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".psv"):
        p_name = filename.split('/')[-1].split('.')[0]
        # read the file into a dataframe
        df = pd.read_csv(os.path.join(directory_path, filename), sep="|")
        sepsislabel = list(df['SepsisLabel'].values)
        y = 1 if np.sum(sepsislabel)>0 else 0
        df['patient'] = p_name
        df['timestamp'] = np.arange(df.shape[0])
        df['y'] = [y for _ in range(df.shape[0])]
        # append the dataframe to the list
        dfs.append(df)

# merge all the dataframes into one
train_df = pd.concat(dfs, ignore_index=True)
train_df.to_csv('train_df.csv')

# set the directory path
# set the directory path
directory_path = r"C://Users//ASUS//Downloads//data//test"

# create an empty list to hold the dataframes
dfs = []

# loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".psv"):
        p_name = filename.split('/')[-1].split('.')[0]
        # read the file into a dataframe
        df = pd.read_csv(os.path.join(directory_path, filename), sep="|")
        sepsislabel = list(df['SepsisLabel'].values)
        y = 1 if np.sum(sepsislabel)>0 else 0
        df['patient'] = p_name
        df['timestamp'] = np.arange(df.shape[0])
        df['y'] = [y for _ in range(df.shape[0])]
        # append the dataframe to the list
        dfs.append(df)

# merge all the dataframes into one
test_df = pd.concat(dfs, ignore_index=True)
test_df.to_csv('test_df.csv')