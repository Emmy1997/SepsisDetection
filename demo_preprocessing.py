from preprocess_data import Imputations

impute_obj = Imputations('train_df_filtered.csv')
# imputed_df_mean = impute_obj.impute_by('Mean')
# print(imputed_df_mean.head())

imputed_df_windows = impute_obj.impute_by('WindowsMeanBucket')
print(imputed_df_windows.head())

# imputed_df_patient_buckets= impute_obj.impute_by('PatientBucket')
# print(imputed_df_patient_buckets.head())