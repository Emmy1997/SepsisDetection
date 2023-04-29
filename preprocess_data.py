import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Imputations:
    def __init__(self, train_df: pd.DataFrame, patients_ids: list):
        self.train_df = train_df
        self.patients_ids = patients_ids
        self.create_patient_group_mapping()
        self.labs = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium',
                'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
                'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
                'Fibrinogen', 'Platelets']
        # list out vital signal features for imputation
        self.vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
        # list out demographic features for imputation
        self.demogs = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
        # List of continuous features for imputation
        self.continuous_features = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
                               'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
                               'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct',
                               'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP',
                               'DBP', 'Resp', 'EtCO2']

    def create_patients_dict(self):
        self.patients_dfs = dict(list(self.train_df.groupby('patient')))

    def collect_patients_df(self, df_dict):
        dfs = []
        for patient, df in df_dict.items():
            dfs.append(df)

        train_df = pd.concat(dfs, axis=0).reset_index()
        return train_df


    def impute_by(self, impute_name: str):
        """
        impute main function
        :param impute_name: one of 'mean', "WindowsMeanBucket" or "PatientBucket"
        :return:
        """
        train_df = copy.deepcopy(self.train_df)
        train_df = self.impute_demog_features(train_df)
        if impute_name == 'Mean':
            return self.impute_mean(train_df)
        elif impute_name == 'WindowsMeanBucket':
            return self.impute_windows(train_df)
        elif impute_name == 'PatientBucket':
            return self.impute_on_patient_group(train_df)

    def impute_mean(self, train_df):
        # Perform mean imputation for continuous features
        train_df[self.continuous_features] = train_df[self.continuous_features].fillna(
            train_df[self.continuous_features].mean())
        return train_df

    def impute_windows(self, train_df):
        train_patients = dict(list(train_df.groupby('patient')))
        for feature in self.continuous_features:
            mean_val_b1_list, mean_val_b2_list, mean_val_b3_list = [], [], []
            ### calculate mean of each bucket
            for patient_df in train_patients.values():
                values = patient_df[feature].dropna()
                if len(values) > 5:
                    bucket_size = int(len(values) / 3)
                    values1, values2, values3 = values[:bucket_size], values[bucket_size:2 * bucket_size], values[
                                                                                                           2 * bucket_size:]
                    mean_val_b1_list.extend(values1)
                    mean_val_b2_list.extend(values2)
                    mean_val_b3_list.extend(values3)

            mean_val_b1, mean_val_b2, mean_val_b3 = np.mean(mean_val_b1_list), np.mean(mean_val_b2_list), np.mean(
                mean_val_b3_list)

            for patient, patient_df in train_patients.items():
                values = patient_df[feature].dropna()
                final_values = {}
                if len(values) > 5:
                    bucket_size = int(len(values) / 3)
                    values1, values2, values3 = values[:bucket_size], values[bucket_size:2 * bucket_size], values[
                                                                                                           2 * bucket_size:]
                    med1, med2, med3 = np.median(values1), np.median(values2), np.median(values3)
                    final_values[f'{feature}_b1'] = med1
                    final_values[f'{feature}_b2'] = med2
                    final_values[f'{feature}_b3'] = med3
                else:
                    final_values[f'{feature}_b1'] = mean_val_b1
                    final_values[f'{feature}_b2'] = mean_val_b2
                    final_values[f'{feature}_b3'] = mean_val_b3


                mask = patient_df[feature].isna()
                b_size = len(mask)//3
                mask_b1 = np.zeros_like(mask, dtype=bool)
                mask_b1[:b_size] = mask[:b_size]
                patient_df.loc[mask_b1, feature] = final_values[f'{feature}_b1']
                #################################
                mask_b2 = np.zeros_like(mask, dtype=bool)
                mask_b2[b_size: 2*b_size] = mask[b_size: 2*b_size]
                patient_df.loc[mask_b2, feature] = final_values[f'{feature}_b2']
                #################################
                mask_b3 = np.zeros_like(mask, dtype=bool)
                mask_b3[2*b_size:] = mask[2*b_size:]
                patient_df.loc[mask_b3, feature] = final_values[f'{feature}_b3']

        train_df = self.collect_patients_df(train_patients)
        return train_df


    def create_patient_group_mapping(self):
        sick_df_training = self.train_df[self.train_df.y == 1]
        sick_grouped = sick_df_training.groupby('patient').agg(
            {'patient': ['first'], 'timestamp': ['max']}).reset_index()
        sick_grouped.columns = ['_'.join(col) for col in sick_grouped.columns.values]
        sick_grouped = sick_grouped[['patient_first', 'timestamp_max']]
        sick_grouped = sick_grouped.rename(columns={'patient_first': 'patient'})
        sick_united = pd.merge(sick_df_training, sick_grouped, on='patient', how='inner')

        fast_sick = sick_united[sick_united['timestamp_max'] < 24]
        med_sick = sick_united[(sick_united['timestamp_max'] >= 24) & (sick_united['timestamp_max'] < 72)]
        slow_sick = sick_united[sick_united['timestamp_max'] >= 72]

        patient_fast = fast_sick.patient.unique()
        patient_med = med_sick.patient.unique()
        patient_slow = slow_sick.patient.unique()
        self.patient_mapping = {'fast': patient_fast, 'med': patient_med, 'slow': patient_slow}

    def assign_patient_group(self, patient):
        if patient in self.patient_mapping['fast']:
            return 0
        elif patient in self.patient_mapping['med']:
            return 1
        elif patient in self.patient_mapping['slow']:
            return 2
        else:
            return -1  # Assign a default value for patients not found in any group - healthy

    def impute_on_patient_group(self, train_df: pd.DataFrame):
        """
            Imputes missing values in demographic features of the training data.

            Parameters:
            -----------
            train_df : pandas.DataFrame
                The training data to be imputed.

            Returns:
            --------
            pandas.DataFrame
                The training data with imputed demographic features.
        """
        # Add a new column to train_df with the assigned values
        train_df['patient_group'] = train_df['patient'].apply(self.assign_patient_group)

        # Group the data by patient_group and continuous feature, and get the mean for each group
        mean_vals = train_df.groupby(['patient_group'])[self.continuous_features].mean().reset_index()
        # Merge the mean values back into the original dataframe
        train_df = pd.merge(train_df, mean_vals, on='patient_group', how='left', suffixes=('', '_mean'))
        # Use vectorized operations to impute missing values based on the mean values
        for feature in self.continuous_features:
            mask = train_df[feature].isna()
            train_df.loc[mask, feature] = train_df.loc[mask, feature + '_mean']

        # Remove the extra columns
        train_df = train_df.drop(columns=[feature + '_mean' for feature in self.continuous_features])
        return train_df


    def impute_demog_features(self, train_df):
        """
        Imputes missing values in demographic features of the training data.

        Parameters:
        -----------
        train_df : pandas.DataFrame
            The training data to be imputed.

        Returns:
        --------
        pandas.DataFrame
            The training data with imputed demographic features.
        """
        # Add a new column to train_df with the assigned values
        train_df['patient_group'] = train_df['patient'].apply(self.assign_patient_group)

        def impute_on_last_timestamp(feature):
            # Find the last non-NaN value for this feature within the patient's data
            last_value = train_df.groupby('patient')[feature].last().dropna()
            # Map each patient to their last value
            return train_df['patient'].map(last_value)

        def impute_most_common(feature):
            # Group the DataFrame by the patient_group column and get the mode of the feature for each group
            mode_values = train_df.groupby('patient_group')[feature].apply(lambda x: x.mode().iloc[0])
            # Create a dictionary with the patient_group value and its mode value
            mode_values_dict = mode_values.to_dict()
            # Replace null values of the feature with the corresponding mode value
            imputed_values = train_df[feature].fillna(train_df['patient_group'].map(mode_values_dict))
            return imputed_values

        for feature in self.demogs:
            # Apply the imputation function to the feature
            train_df[feature] = train_df[feature].fillna(impute_on_last_timestamp(feature))
            train_df[feature] = train_df[feature].fillna(impute_most_common(feature))


        return train_df

class Normalization:
    def __init__(self, train_df, cont_features):
        self.train_df = train_df
        self.cont_features = cont_features

    def normalize_by(self, normalization_type):
        if normalization_type == 'Mean':
            return self.normalize_mean()
        elif normalization_type == 'WindowsMeanBucket':
            return self.normalize_bucket(by='Mean')
        elif normalization_type == 'WindowsMedianBucket':
            return self.normalize_bucket(by='Median')

    def normalize_mean(self):
        # Group rows by patient and compute the mean
        train_df =  self.train_df[self.cont_features]
        groups = train_df.groupby('patient').mean()
        # Create a new DataFrame where each row represents the mean for a patient
        train_df_mean = pd.DataFrame(groups.values, columns=groups.columns, index=groups.index).reset_index()
        return train_df_mean

    def normalize_bucket(self, by = 'Mean'):
        train_patients = dict(list(self.train_df.groupby('patient')))
        func = np.max
        if by == 'Mean':
            func = np.mean
        elif by == 'Median':
            func = np.median
        for patient, patient_df in train_patients.items():
            for feature in self.cont_features:
                values = patient_df[feature].values
                if len(values) > 5:
                    bucket_size = int(len(values) / 3)
                    values1, values2, values3 = [values[:bucket_size], values[bucket_size:2 * bucket_size], values[
                                                                                                            2 * bucket_size:]]
                    val1, val2, val3 = func(values1), func(values2), func(values3)
                elif len(values) > 2:
                    val1, val2, val3 = values[0], values[1], func(values[2:])
                else:
                    val1, val2, val3 = values[0], values[0], values[0]
                patient_df.loc[:, f'{feature}_b1'] = val1
                patient_df.loc[:, f'{feature}_b2'] = val2
                patient_df.loc[:, f'{feature}_b3'] = val3
                patient_df.drop(columns=[feature], inplace=True)

            ## keep one row per patient
            train_patients[patient] = patient_df.head(1)

        train_df = pd.concat(train_patients.values(), ignore_index=True).reset_index()
        return train_df


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
must features:
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

class PreProcess:
    def __init__(self, train_path, sample=True):
        self.train_df = pd.read_csv(train_path)
        if sample:
            self.train_df = self.train_df.head(10000)

        self.train_df = self.train_df.sort_values(['patient', 'timestamp'])
        self.train_df.drop(columns=['index', 'Unnamed: 0.1'], inplace=True)
        self.patients_ids = self.train_df.patient.unique()
        self.demogs_features = ['patient', 'y', 'Age', 'Gender', 'HospAdmTime_final', 'ICULOS_final']
        # List of continuous features
        self.cont_features = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                         'Alkalinephos',
                         'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
                         'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct',
                         'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'HR', 'O2Sat', 'Temp', 'SBP',
                         'MAP', 'DBP', 'Resp', 'EtCO2']
        self.special_features = ['SIRS']
        self.default_features = self.demogs_features + self.cont_features + self.special_features

    @staticmethod
    def compute_SIRS(row):
        counter_sirs = 0
        if row['HR'] > 90:
            counter_sirs += 1
        if row['Temp'] < 36 or row['Temp'] > 38:
            counter_sirs += 1
        if row['PaCO2'] < 32 or row['Resp'] > 20:
            counter_sirs += 1
        if row['WBC'] > 12000 or row['WBC'] < 4000:
            counter_sirs += 1
        return counter_sirs

    @staticmethod
    def get_last_feature(train_df):
        # Create a new column 'ICULOS_final' with the last non-null value of ICULOS for each patient
        last_ICULOS = train_df.groupby('patient')['ICULOS'].last()
        last_ICULOS_dict = last_ICULOS.to_dict()
        train_df['ICULOS_final'] = train_df['patient'].map(last_ICULOS_dict)
        # Create a new column 'HospAdmTime_final' with the last non-null value of HospAdmTime for each patient
        last_ICULOS = train_df.groupby('patient')['HospAdmTime'].last()
        last_ICULOS_dict = last_ICULOS.to_dict()
        train_df['HospAdmTime_final'] = train_df['patient'].map(last_ICULOS_dict)
        return train_df

    def run_pipeline(self, pipeline_dict=None):
        """
        pipeline_dict = {"impute_type": 'Mean' (one of 'Mean', "WindowsMeanBucket" or "PatientBucket") ,
                        "normalization_type": 'mean' (one of 'Mean', 'WindowsMeanBucket', 'WindowsMedianBucket')}
        :param pipeline_dict:
        :return:
        """
        if pipeline_dict is None:
            pipeline_dict = { "impute_type": 'Mean', 'normalization_type': 'Mean',
                              "feature_set": self.default_features}

        train_df_organized = self.get_last_feature(self.train_df)
        impute_obj = Imputations(train_df_organized, self.patients_ids)
        train_df_imputed = impute_obj.impute_by(pipeline_dict.get("impute_type"))
        train_df_imputed['SIRS'] = train_df_imputed.apply(self.compute_SIRS, axis=1)
        features_final = pipeline_dict.get("feature_set") + self.demogs_features + self.special_features
        train_df_filtered = train_df_imputed[features_final]
        new_cont_features = list(set(self.cont_features) & set(train_df_filtered.columns)) + self.special_features
        norm_obj = Normalization(train_df_filtered, new_cont_features)
        train_df_normalized = norm_obj.normalize_by(pipeline_dict.get("normalization_type"))
        return train_df_normalized