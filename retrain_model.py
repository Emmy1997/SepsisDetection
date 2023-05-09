import pandas as pd
from xgboost import XGBClassifier
import pickle


####
train_df = pd.read_csv(f'transformed_files/experiment_9/train_transformed.csv')
val_df = pd.read_csv(f'transformed_files/experiment_9/val_transformed.csv')
to_remove = ['level_0', 'index', 'patient', 'y', 'Unnamed: 0']
features = list(set(train_df.columns))
features = [x for x in features if x not in to_remove]
train_Y= train_df[['y']]
train_X = train_df.drop(columns=to_remove)
val_Y = val_df[['y']]
val_X = val_df.drop(columns=to_remove)
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