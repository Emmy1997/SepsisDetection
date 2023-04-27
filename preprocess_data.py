import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Imputations:
    def __init__(self, train_path):
        self.train_df = pd.read_csv(train_path)
        self.train_df = self.train_df.sort_values(['patient', 'timestamp'])
        self.patients_ids = self.train_df.patient.unique()
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
            patients_df_features = {}
            ### calculate mean of each bucket
            for patient_df in train_patients.values():
                values = patient_df[feature].dropna()
                if len(values) > 0:
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
                patients_df_features[patient] = {}
                if len(values) > 0:
                    bucket_size = int(len(values) / 3)
                    values1, values2, values3 = values[:bucket_size], values[bucket_size:2 * bucket_size], values[
                                                                                                           2 * bucket_size:]
                    med1, med2, med3 = np.median(values1), np.median(values2), np.median(values3)
                    patients_df_features[patient][f'{feature}_b1'] = med1
                    patients_df_features[patient][f'{feature}_b2'] = med2
                    patients_df_features[patient][f'{feature}_b3'] = med3
                else:
                    patients_df_features[patient][f'{feature}_b1'] = mean_val_b1
                    patients_df_features[patient][f'{feature}_b2'] = mean_val_b2
                    patients_df_features[patient][f'{feature}_b3'] = mean_val_b3

            for patient, features_dict in patients_df_features.items():
                for feature, value in features_dict.items():
                    mask = train_patients[patient][feature].isna()
                    train_patients[patient].loc[mask, feature] = value

        train_df = self.collect_patients_df(train_patients)
        return train_df

    # def impute_windows(self, train_df):
    #     self.create_patients_dict()
    #     train_patients = copy.deepcopy(self.patients_dfs)
    #     for feature in self.continuous_features:
    #         # concatenate all patient dataframes into a single one
    #         feature_df = train_df[["patient", feature]]
    #
    #         # compute the mean of each bucket over all patients
    #         feature_df["bucket"] = pd.cut(feature_df[feature], bins=3, labels=["b1", "b2", "b3"])
    #         mean_vals = feature_df.groupby("bucket")[feature].mean().to_dict()
    #
    #         # use vectorized operations to update each patient's dataframe
    #         for patient, patient_df in train_patients.items():
    #             bucket_mask = pd.cut(patient_df[feature], bins=3, labels=["b1", "b2", "b3"]).notna()
    #             bucket_vals = bucket_mask.map(mean_vals).fillna(0)
    #             for bucket in ["b1", "b2", "b3"]:
    #                 bucket_col = f"{feature}_{bucket}"
    #                 patient_df.loc[bucket_mask & (patient_df["bucket"] == bucket), bucket_col] = bucket_vals[bucket]
    #                 patient_df.loc[~bucket_mask, bucket_col] = mean_vals[bucket]
    #
    #     train_df = self.collect_patients_df(train_patients)
    #     return train_df

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

        # Create a new column 'ICULOS_final' with the last non-null value of ICULOS for each patient
        last_ICULOS = train_df.groupby('patient')['ICULOS'].last()
        last_ICULOS_dict = last_ICULOS.to_dict()
        train_df['ICULOS_final'] = train_df['patient'].map(last_ICULOS_dict)

        return train_df