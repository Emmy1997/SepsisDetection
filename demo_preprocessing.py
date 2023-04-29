from preprocess_data import PreProcess, train_val_split
import os
import json

path_to_save = 'transformed_files'

# Find last experiment number
experiment_number = 0
for filename in os.listdir(path_to_save):
    if filename.endswith('.csv') and filename.startswith('experiment_'):
        exp_num = int(filename.split('_')[1])
        if exp_num > experiment_number:
            experiment_number = exp_num

# Increment experiment number by 1
experiment_number += 1

# Create folder for experiment number and save files inside
experiment_folder = os.path.join(path_to_save, f"experiment_{experiment_number}")
os.makedirs(experiment_folder, exist_ok=True)

set_A = ['Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium']
train_pipe_line_dict = {"impute_type": 'Mean', 'normalization_type': 'Mean', "feature_set": set_A}
train_path = 'train_df_filtered.csv'
train_df, val_df = train_val_split(train_path)

#################### TRANSFORM TRAIN ####################
pre_obj_train = PreProcess(train_df, sample=False)
train_df_transformed = pre_obj_train.run_pipeline(pipeline_dict=train_pipe_line_dict)


#################### TRANSFORM VAL ####################
pre_obj_val = PreProcess(val_df, sample=False)
val_pipe_line_dict = {"impute_type": 'Mean', 'normalization_type': 'Mean', "feature_set": set_A}
val_df_transformed = pre_obj_val.run_pipeline(train=False, train_df=train_df, pipeline_dict=val_pipe_line_dict)


###########################################################
# Save transformed train data
train_df_file = os.path.join(experiment_folder, 'train_transformed.csv')
train_df_transformed.to_csv(train_df_file)

# Save pipeline dictionary for train
train_pipeline_dict_file = os.path.join(experiment_folder, 'train_pipeline_dict.json')
with open(train_pipeline_dict_file, 'w') as f:
    json.dump(train_pipe_line_dict, f)


# Save transformed validation data
val_df_file = os.path.join(experiment_folder, 'val_transformed.csv')
val_df_transformed.to_csv(val_df_file)

# Save pipeline dictionary for validation
val_pipeline_dict_file = os.path.join(experiment_folder, 'val_pipeline_dict.json')
with open(val_pipeline_dict_file, 'w') as f:
    json.dump(val_pipe_line_dict, f)



