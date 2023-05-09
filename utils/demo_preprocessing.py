from utils.preprocess_data import PreProcess, train_val_split
import os
import json

path_to_save = '../transformed_files'

"""
###############################
feature set:
- more than 90% missing values - TroponinI, Fibrinogen, EtCO2, and Bilirubin_direct 
- dropping unit2 and unit1
- with SIRS column
###############################
1. until temp
2. all 
3. hypothesis was rejected
###############################
must features - demog:
ICULOSS
HospAdmTime
Age, Gender
###############################
filtering:
1. patients with more than than x% null rows
2. filter all rows per patient if all from some point is null

normalization per patient:
1. mean 
2. window and then median/mean
"""


# Find last experiment number
experiment_number = 0
for filename in os.listdir(path_to_save):
    if filename.startswith('experiment_'):
        exp_num = int(filename.split('_')[1])
        if len(os.listdir(os.path.join(path_to_save, filename))) == 0:
            continue
        if exp_num > experiment_number:
            experiment_number = exp_num

# Increment experiment number by 1
experiment_number += 1

# Create folder for experiment number and save files inside
experiment_folder = os.path.join(path_to_save, f"experiment_{experiment_number}")
print(f'experiment folder = {experiment_folder}')
os.makedirs(experiment_folder, exist_ok=True)

set_B = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                         'Alkalinephos',
                         'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
                         'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct',
                         'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'HR', 'O2Sat', 'Temp', 'SBP',
                         'MAP', 'DBP', 'Resp', 'EtCO2']

## impute_type - Mean, 'WindowsMeanBucket', 'PatientBucket'.
## normalization_type - 'Mean', 'WindowsMeanBucket', 'WindowsMedianBucket'.
train_pipe_line_dict = {"impute_type": 'Mean', 'normalization_type': 'WindowsMedianBucket'}
train_path = '../train_df_filtered.csv'
train_df, val_df = train_val_split(train_path)

#################### TRANSFORM TRAIN ####################
pre_obj_train = PreProcess(train_df, sample=False)
train_df_transformed = pre_obj_train.run_pipeline(pipeline_dict=train_pipe_line_dict)
train_pipe_line_dict = pre_obj_train.pipeline_dict

#################### TRANSFORM VAL ####################
pre_obj_val = PreProcess(val_df, sample=False)
val_df_transformed = pre_obj_val.run_pipeline(train=False, train_df=train_df, pipeline_dict=train_pipe_line_dict)

###########################################################
# Save transformed train data
train_df_file = os.path.join(experiment_folder, 'train_transformed.csv')
train_df_transformed.to_csv(train_df_file)

# Save pipeline dictionary for train
train_pipeline_dict_file = os.path.join(experiment_folder, '../train_pipeline_dict.json')
with open(train_pipeline_dict_file, 'w') as f:
    json.dump(train_pipe_line_dict, f)


# Save transformed validation data
val_df_file = os.path.join(experiment_folder, 'val_transformed.csv')
val_df_transformed.to_csv(val_df_file)




