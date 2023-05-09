import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm
import pickle



train_df = pd.read_csv(f'transformed_files/experiment_9/train_transformed.csv')
val_df = pd.read_csv(f'transformed_files/experiment_9/val_transformed.csv')
# train_df_with_labeles = pd.merge(train_df, train_df_filtered, on='patient',how='inner')
# val_df_with_labeles = pd.merge(val_df, train_df_filtered, on='patient',how='inner')
to_remove = ['level_0', 'index', 'patient', 'y', 'Unnamed: 0']
features = list(set(train_df.columns))
features = [x for x in features if x not in to_remove]
print(features)
    # ["Age", "Gender", "HospAdmTime_final", "ICULOS_final", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "SIRS"]))
val_df_p2y = val_df[['index', 'patient', 'y']]
# val_df_p2y.to_csv('val_p2y')
train_Y= train_df[['y']]
train_X = train_df.drop(columns=to_remove)
val_Y = val_df[['y']]
val_X = val_df.drop(columns=to_remove)
# evaluate_all_models(train_X, train_Y.values.ravel(), val_X, val_Y.values.ravel(), features, i)
full_X =  pd.concat([train_X, val_X], axis=0)
full_Y = pd.concat([train_Y, val_Y], axis=0)

# best params
# num estimators = 400 max_depth = 80 sub_sample=0.8

xgboost = XGBClassifier(n_estimators=400, scale_pos_weight=13,
                             max_depth=80, verbosity=1, eval_metric='error', max_delta_step=0.15,
                             subsample=0.8)
xgboost.fit(full_X[features], full_Y)
with open('xgb_model.pickle', 'wb') as f:
    pickle.dump(xgboost, f)