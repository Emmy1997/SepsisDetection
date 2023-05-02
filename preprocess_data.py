import copy
import pandas as pd
import numpy as np


class Imputations:
    def __init__(self, data_df: pd.DataFrame, patients_ids: list, pipeline_dict: dict):
        self.data_df = data_df
        self.patients_ids = patients_ids
        self.create_patient_group_mapping(self.data_df)
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
        self.pipeline_dict = pipeline_dict

    def create_patients_dict(self):
        self.patients_dfs = dict(list(self.data_df.groupby('patient')))

    def collect_patients_df(self, df_dict):
        dfs = []
        for patient, df in df_dict.items():
            dfs.append(df)

        train_df = pd.concat(dfs, axis=0).reset_index()
        return train_df


    def impute_by(self, data_df,  impute_name: str):
        """
        impute main function
        :param impute_name: one of 'mean', "WindowsMeanBucket" or "PatientBucket"
        :return:
        """
        if impute_name == 'Mean':
            return self.impute_mean(data_df)
        elif impute_name == 'WindowsMeanBucket':
            return self.impute_windows(data_df)
        elif impute_name == 'PatientBucket':
            return self.impute_on_patient_group(data_df)

    def impute_mean(self, data_df):
        # Perform mean imputation for continuous features
        data_df[self.continuous_features] = data_df[self.continuous_features].fillna(
            data_df[self.continuous_features].mean())
        return data_df


    def impute_windows(self, data_df):
        func = np.median
        data_patients = dict(list(data_df.groupby('patient')))
        self.pipeline_dict['features_train'] = {}
        for feature in self.continuous_features:
            mean_val_b1_list, mean_val_b2_list, mean_val_b3_list = [], [], []
            ### calculate mean of each bucket

            for patient_df in data_patients.values():
                values = patient_df[feature].values
                bucket_size = int(len(values) / 3)
                values1, values2, values3 = values[:bucket_size], values[bucket_size:2 * bucket_size], values[2 * bucket_size:]
                values1, values2, values3 = values1[~np.isnan(values1)], values2[~np.isnan(values2)], values3[~np.isnan(values3)]
                if len(values1) > 0:
                    mean_val_b1_list.extend(values1)
                if len(values2) > 0:
                    mean_val_b2_list.extend(values2)
                if len(values3) > 0:
                    mean_val_b3_list.extend(values3)

            mean_val_b1, mean_val_b2, mean_val_b3 = np.mean(mean_val_b1_list), np.mean(mean_val_b2_list), np.mean(
                mean_val_b3_list)
            self.pipeline_dict['features_train'][f'{feature}_b1'] = mean_val_b1
            self.pipeline_dict['features_train'][f'{feature}_b2'] = mean_val_b2
            self.pipeline_dict['features_train'][f'{feature}_b3'] = mean_val_b3

            for patient, patient_df in data_patients.items():
                values = patient_df[feature].values
                val_b1, val_b2, val_b3 = mean_val_b1, mean_val_b2, mean_val_b3

                bucket_size = int(len(values) / 3)
                values1, values2, values3 = values[:bucket_size], values[bucket_size:2 * bucket_size], values[
                                                                                                       2 * bucket_size:]
                values1, values2, values3 = values1[~np.isnan(values1)], values2[~np.isnan(values2)], values3[
                    ~np.isnan(values3)]

                if len(values) > 5:
                    if len(values1) > 0:
                        val_b1 = func(values1)
                    if len(values2) > 0:
                        val_b2 = func(values2)
                    if len(values3) > 0:
                        val_b3 = func(values3)

                ## take null rows per bucket - True if the value is null and matches current bucket indexing
                mask = patient_df[feature].isna()
                b_size = len(mask) // 3
                mask_b1 = np.zeros_like(mask, dtype=bool)
                mask_b1[:b_size] = mask[:b_size]
                patient_df.loc[mask_b1, feature] = val_b1
                #################################
                mask_b2 = np.zeros_like(mask, dtype=bool)
                mask_b2[b_size: 2 * b_size] = mask[b_size: 2 * b_size]
                patient_df.loc[mask_b2, feature] = val_b2
                #################################
                mask_b3 = np.zeros_like(mask, dtype=bool)
                mask_b3[2 * b_size:] = mask[2 * b_size:]
                patient_df.loc[mask_b3, feature] = val_b3

        data_df = self.collect_patients_df(data_patients)
        return data_df


    def create_patient_group_mapping(self, df_curr):
        # Create a new column 'timestamp_max' with the last non-null value of timestamp_max for each patient
        last_ts = df_curr.groupby('patient')['timestamp'].last()
        last_ts_dict = last_ts.to_dict()
        df_curr['timestamp_max'] = df_curr['patient'].map(last_ts_dict)

        fast_sick = df_curr[df_curr['timestamp_max'] < 24]
        med_sick = df_curr[(df_curr['timestamp_max'] >= 24) & (df_curr['timestamp_max'] < 72)]
        slow_sick = df_curr[df_curr['timestamp_max'] >= 72]

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

    def impute_on_patient_group(self, curr_df: pd.DataFrame):
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
        curr_df['patient_group'] = curr_df['patient'].apply(self.assign_patient_group)

        # Group the data by patient_group and continuous feature, and get the mean for each group
        mean_vals = curr_df.groupby(['patient_group'])[self.continuous_features].mean().reset_index()
        # Merge the mean values back into the original dataframe
        curr_df = pd.merge(curr_df, mean_vals, on='patient_group', how='left', suffixes=('', '_mean'))
        # Use vectorized operations to impute missing values based on the mean values
        for feature in self.continuous_features:
            mask = curr_df[feature].isna()
            curr_df.loc[mask, feature] = curr_df.loc[mask, feature + '_mean']

        # Remove the extra columns
        curr_df = curr_df.drop(columns=[feature + '_mean' for feature in self.continuous_features])
        return curr_df

class ImputationsTest(Imputations):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, pipeline_dict: dict):
        train_patient_ids = list(train_df.patient.unique())
        super().__init__(train_df, train_patient_ids, pipeline_dict)
        self.test_df = test_df
        self.create_patient_group_mapping(test_df)


    def impute_mean(self, test_df):
        # Perform mean imputation for continuous features based on means from train
        test_df[self.continuous_features] = test_df[self.continuous_features].fillna(
            self.data_df[self.continuous_features].mean())
        return test_df

    def impute_windows(self, test_df):
        test_patients = dict(list(test_df.groupby('patient')))
        func = np.median
        for feature in self.continuous_features:
            mean_val_b1 = self.pipeline_dict['features_train'][f'{feature}_b1']
            mean_val_b2 = self.pipeline_dict['features_train'][f'{feature}_b2']
            mean_val_b3 = self.pipeline_dict['features_train'][f'{feature}_b3']

            for patient, patient_df in test_patients.items():
                values = patient_df[feature].values
                val_b1, val_b2, val_b3 = mean_val_b1, mean_val_b2, mean_val_b3

                bucket_size = int(len(values) / 3)
                values1, values2, values3 = values[:bucket_size], values[bucket_size:2 * bucket_size], values[
                                                                                                       2 * bucket_size:]
                values1, values2, values3 = values1[~np.isnan(values1)], values2[~np.isnan(values2)], values3[
                    ~np.isnan(values3)]

                if len(values) > 5:
                    if len(values1) > 0:
                        val_b1 = func(values1)
                    if len(values2) > 0:
                        val_b2 = func(values2)
                    if len(values3) > 0:
                        val_b3 = func(values3)

                ## take null rows per bucket - True if the value is null and matches current bucket indexing
                mask = patient_df[feature].isna()
                b_size = len(mask) // 3
                mask_b1 = np.zeros_like(mask, dtype=bool)
                mask_b1[:b_size] = mask[:b_size]
                patient_df.loc[mask_b1, feature] = val_b1
                #################################
                mask_b2 = np.zeros_like(mask, dtype=bool)
                mask_b2[b_size: 2 * b_size] = mask[b_size: 2 * b_size]
                patient_df.loc[mask_b2, feature] = val_b2
                #################################
                mask_b3 = np.zeros_like(mask, dtype=bool)
                mask_b3[2 * b_size:] = mask[2 * b_size:]
                patient_df.loc[mask_b3, feature] = val_b3

        test_df = self.collect_patients_df(test_patients)
        return test_df


class Normalization:
    def __init__(self, data_df, cont_features):
        self.data_df = data_df
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
        # curr_features = self.cont_features + ['patient']
        data_df = self.data_df
        groups = data_df.groupby('patient').mean()
        # Create a new DataFrame where each row represents the mean for a patient
        data_df_mean = pd.DataFrame(groups.values, columns=groups.columns, index=groups.index).reset_index()
        return data_df_mean

    def normalize_bucket(self, by = 'Mean'):
        """
        Normalize continuous features by dividing them into three buckets using the specified method of
        aggregation (mean, median or max) and computing the corresponding bucket values for each patient.

        :param by: The method of aggregation to use (default: 'Max').
            One of 'Mean', 'Median' or 'Max'.
        :type by: str

        :return: A DataFrame containing one row per patient and the computed bucket values for each continuous feature.
        :rtype: pandas.DataFrame
        """
        data_patients = dict(list(self.data_df.groupby('patient')))
        agg_func = np.max
        if by == 'Mean':
            agg_func = np.mean
        elif by == 'Median':
            agg_func = np.median
        for patient, patient_df in data_patients.items():
            for feature in self.cont_features:
                values = patient_df[feature].values
                if len(values) > 5:
                    bucket_size = int(len(values) / 3)
                    values1, values2, values3 = [values[:bucket_size], values[bucket_size:2 * bucket_size], values[
                                                                                                            2 * bucket_size:]]
                    val1, val2, val3 = agg_func(values1), agg_func(values2), agg_func(values3)
                elif len(values) > 2:
                    val1, val2, val3 = values[0], values[1], agg_func(values[2:])
                else:
                    val1, val2, val3 = values[0], values[0], values[0]
                patient_df.loc[:, f'{feature}_b1'] = val1
                patient_df.loc[:, f'{feature}_b2'] = val2
                patient_df.loc[:, f'{feature}_b3'] = val3
                patient_df.drop(columns=[feature], inplace=True)

            ## keep one row per patient
            data_patients[patient] = patient_df.head(1)

        data_df = pd.concat(data_patients.values(), ignore_index=True).reset_index()
        return data_df


class PreProcess:
    def __init__(self, df: pd.DataFrame, sample=True):
        self.df = df
        if sample:
            self.df = self.df.head(10000)

        self.df = self.df.sort_values(['patient', 'timestamp'])
        self.df.drop(columns=['index', 'Unnamed: 0.1'], inplace=True)
        self.patients_ids = self.df.patient.unique()
        self.demogs_features = ['patient', 'y', 'Age', 'Gender', 'HospAdmTime_final', 'ICULOS_final']
        # List of continuous features
        self.cont_features = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                         'Alkalinephos',
                         'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
                         'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct',
                         'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'HR', 'O2Sat', 'Temp', 'SBP',
                         'MAP', 'DBP', 'Resp', 'EtCO2', 'HospAdmTime_final', 'ICULOS_final']
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
    def get_last_feature(df_curr):
        """
        create 2 new columns based on last value
        :param df_curr:
        :return:
        """
        # Create a new column 'ICULOS_final' with the last non-null value of ICULOS for each patient
        last_ICULOS = df_curr.groupby('patient')['ICULOS'].last()
        last_ICULOS_dict = last_ICULOS.to_dict()
        df_curr['ICULOS_final'] = df_curr['patient'].map(last_ICULOS_dict)
        # Create a new column 'HospAdmTime_final' with the last non-null value of HospAdmTime for each patient
        last_ICULOS = df_curr.groupby('patient')['HospAdmTime'].last()
        last_ICULOS_dict = last_ICULOS.to_dict()
        df_curr['HospAdmTime_final'] = df_curr['patient'].map(last_ICULOS_dict)
        return df_curr

    def run_pipeline(self, train = True, train_df=None, pipeline_dict=None):
        """
            Runs a pipeline of data preprocessing steps on the training dataframe.

            Args:
                pipeline_dict (dict): A dictionary containing the following keys (default is None):
                    - 'impute_type' (str): Type of imputation method to use. One of 'Mean', 'WindowsMeanBucket' or 'PatientBucket'.
                    - 'normalization_type' (str): Type of normalization method to use. One of 'Mean', 'WindowsMeanBucket' or 'WindowsMedianBucket'.
                    - 'feature_set' (list): A list of features to use for training. Default is self.default_features.

            Returns:
                pd.DataFrame: The preprocessed training dataframe with imputations and normalizations applied.
            """
        # If pipeline_dict is None, use default parameters
        if pipeline_dict is None:
            self.pipeline_dict = {"impute_type": 'Mean', 'normalization_type': 'mean', "feature_set": self.default_features}
        if pipeline_dict.get('feature_set') is None:
            self.pipeline_dict['feature_set'] = self.default_features

        # Organize train data by patient and keep only the last feature observation
        df_curr = copy.deepcopy(self.df)
        df_organized = self.get_last_feature(df_curr)

        # Perform imputation on the train data
        if train:
            impute_obj = Imputations(df_organized, self.patients_ids, self.pipeline_dict)

        else:
            ## test
            if train_df is not None:
                impute_obj = ImputationsTest(train_df=train_df, test_df=df_organized, pipeline_dict=self.pipeline_dict)
            else:
                raise ValueError("if test then must specify train_df Dataframe as input!")

        df_imputed = impute_obj.impute_by(df_organized, pipeline_dict.get("impute_type"))

        # Compute SIRS score and add to the train data
        df_imputed['SIRS'] = df_imputed.apply(self.compute_SIRS, axis=1)

        # Filter train data to only include specified features
        features_final = list(set(pipeline_dict.get("feature_set") + self.demogs_features + self.special_features))
        df_filtered = df_imputed[features_final]

        # Normalize timeseries feature to have only one per patient based on normalization_type
        new_cont_features = list(set(list(set(self.cont_features) & set(df_filtered.columns)) + self.special_features))
        norm_obj = Normalization(df_filtered, new_cont_features)
        df_normalized = norm_obj.normalize_by(pipeline_dict.get("normalization_type")).reset_index()

        return df_normalized



def train_val_split(path, p = 0.7):
    df = pd.read_csv(path)
    unique_ids = df.patient.unique()
    n_patients = len(unique_ids)
    n_train = int(n_patients*p)
    idx_train = np.random.choice(np.arange(n_patients), n_train)
    train_patients = unique_ids[idx_train]
    idx_val = [x for x in range(n_patients) if x not in idx_train]
    val_patients = unique_ids[idx_val]
    train_df = df[df.patient.isin(train_patients)]
    val_df = df[df.patient.isin(val_patients)]
    return train_df, val_df
