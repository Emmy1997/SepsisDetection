from preprocess_data import Imputations, PreProcess

# impute_obj = Imputations('train_df_filtered.csv')
# # imputed_df_mean = impute_obj.impute_by('Mean')
# # print(imputed_df_mean.head())
#
# imputed_df_windows = impute_obj.impute_by('WindowsMeanBucket')
# print(imputed_df_windows.head())

# imputed_df_patient_buckets= impute_obj.impute_by('PatientBucket')
# print(imputed_df_patient_buckets.head())
train_path = 'train_df_filtered.csv'
pre_obj = PreProcess(train_path)
set_A = ['Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
                         'Magnesium']
# set_B = []
# set_C = []
pipe_line_dict = { "impute_type": 'Mean', 'normalization_type': 'WindowsMeanBucket',
                              "feature_set": set_A}
train_df_transformed = pre_obj.run_pipeline(pipe_line_dict)