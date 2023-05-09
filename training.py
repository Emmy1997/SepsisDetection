from models import KNN, RandomForest, XGBOOST, CATBOOST
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd
import pickle
import json


def min_max_scaler(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df


def training_models(train_X, train_Y, val_X, val_Y, features_set):
    knn_f1_scores, rf_f1_scores, xgb_f1_scores, cat_f1_scores = {}, {}, {}, {}
    # train knn models
    train_scaled, val_scaled = min_max_scaler(train_X), min_max_scaler(val_X)
    print('running KNN')
    for k in tqdm(range(1, 25), desc='KNN'):
        print(f"running KNN with k: {k}")
        knn = KNN(k, features_set)
        score = knn.train_knn(train_scaled, train_Y, val_scaled, val_Y)
        knn_f1_scores[str(k)] = score
        print(f"f1 score: {score}")

    best_k  = max(knn_f1_scores, key=lambda k: knn_f1_scores[k])
    best_score_knn = knn_f1_scores[best_k]
    print(f"best score: {best_score_knn}, with k: {best_k}")

    ## train rf models
    max_depths = [200]
    n_estimators = [25, 50, 100, 150, 200, 300, 400]
    print('running RandomForest')
    for n in tqdm(n_estimators, desc='RandomForest'):
        for max_dep in max_depths:
            print("running RandomForest with params:")
            print(f" n: {n}, maxdepth: {max_dep}")
            rf = RandomForest(n, max_dep, features_set)
            score = rf.train_rf(train_X, train_Y, val_X, val_Y)
            rf_f1_scores[str(n)] = score
            print(f"f1 score: {score}")
    best_n_max_depth_rf  = max(rf_f1_scores, key=lambda k: rf_f1_scores[k])
    best_score_rf = rf_f1_scores[best_n_max_depth_rf]
    print(f"best rf score: {best_score_rf}, with params: {best_n_max_depth_rf}")

    # # train xgb models
    max_depths = [80 ,100, 150, 200, 300]
    n_estimators = [25, 50, 100, 150, 200, 300, 400, 500]
    sub_sample = [0.5, 0.6, 0.7, 0.8, 0.9, None]
    print('running XGBOOST')
    for s in tqdm(sub_sample, desc='XGBOOST'):
        for n in n_estimators:
            for max_dep in max_depths:
                print("running XGBOOST with params:")
                print(f" subsmpl: {s}, n: {n}, maxdepth: {max_dep}")
                xgb = XGBOOST(n, max_dep, s, features_set)
                score = xgb.train_xgb(train_X, train_Y, val_X, val_Y)
                xgb_f1_scores[str(n) +' '+ str(max_dep) +' '+ str(s)] = score
                print(f"f1 score: ){score}")
    best_params_xgb  = max(xgb_f1_scores, key=lambda k: xgb_f1_scores[k])
    best_score_xgb = xgb_f1_scores[best_params_xgb]
    print(f"best xgb score: {best_score_xgb}, with params: {best_params_xgb}")

    ## train CatBoost model
    max_depths = [8, 10, 12, 13, 15, 16]
    n_estimators = [1000]
    # sub_sample = [0.5, 0.6, 0.7, 0.8, 0.9, None]
    reg = [0.01, 0.1, 1.0, 10.0]
    lrs= [0.01, 0.05, 0.1, 0.2, 0.3]
    s=1
    print("running Catboost")
    for lr in tqdm(lrs, desc='CASTBOOST'):
        for r in reg:
        # for s in sub_sample:
            for n in n_estimators:
                for max_dep in max_depths:
                    print("running CATBOOST with params:")
                    print(f" lr: {lr}, regu :{r}, subsmpl: {s}, n: {n}, maxdepth: {max_dep}")
                    cat = CATBOOST(n, max_dep, s, r, lr, features_set)
                    score, best_iteration = cat.train_cat(train_X, train_Y, val_X, val_Y)
                    cat_f1_scores[(n, max_dep, r, lr)] = (score, best_iteration)
                    print(f"f1 score: {score}")
    best_params_cat  = max(cat_f1_scores, key=lambda k: cat_f1_scores[k][0])
    best_score_cat = cat_f1_scores[best_params_cat][0]
    print(f"best cat score: {best_score_cat}, with params: {best_params_cat}")


    return best_k, best_score_knn, knn_f1_scores,\
        best_params_xgb, best_score_xgb, xgb_f1_scores,\
        best_n_max_depth_rf, best_score_rf, rf_f1_scores, \
        best_params_cat,  best_score_cat, cat_f1_scores


def evaluate_all_models(train_X, train_Y, val_X, val_Y, features_set, i):
    ##evaluate knn
    ## add loop over all the different imputations and featues sets
    best_k, best_score_knn, knn_f1_scores, \
        best_n_max_depth_rf, best_score_rf, rf_f1_scores, \
        best_params_xgb, best_score_xgb, xgb_f1_scores, \
        best_params_cat, best_score_cat, cat_f1_scores = training_models(train_X, train_Y, val_X, val_Y, features_set)
    results_dict = {'KNN': [best_k, best_score_knn, knn_f1_scores],
                    'RF': [best_n_max_depth_rf, best_score_rf, rf_f1_scores],
                    'XGB': [best_params_xgb, best_score_xgb, xgb_f1_scores],
                    'CAT': [best_params_cat, best_score_cat, cat_f1_scores]}
    with open(f'results_dict_experiment_{i}.json', 'w') as f:
        json.dump(results_dict, f)



## optimization loop
for i in range(4, 13):
    print(f"Running Hyperparam tune on experiment {i}")
    train_df = pd.read_csv(f'transformed_files/experiment_{i}/train_transformed.csv')
    val_df = pd.read_csv(f'transformed_files/experiment_{i}/val_transformed.csv')
    to_remove =['level_0', 'index', 'patient', 'y', 'Unnamed: 0']
    cols = list(set(train_df.columns))
    features = [x for x in cols if x not in to_remove]
    train_X = train_df[features]
    val_X = val_df[features]
    train_Y= train_df[['y']]
    val_Y = val_df[['y']]
    evaluate_all_models(train_X, train_Y.values.ravel(), val_X, val_Y.values.ravel(), features, i)

## rows for running full training on best model

# train_df = pd.read_csv(f'transformed_files/experiment_{9}/train_transformed.csv')
# val_df = pd.read_csv(f'transformed_files/experiment_{9}/val_transformed.csv')
# to_remove =['level_0', 'index', 'patient', 'y', 'Unnamed: 0']
# cols = list(set(train_df.columns))
# features = [x for x in cols if x not in to_remove]
# train_X = train_df[features]
# val_X = val_df[features]
# train_Y= train_df[['y']]
# val_Y = val_df[['y']]
# full_X =  pd.concat([train_X, val_X], axis=0)
# full_Y = pd.concat([train_Y, val_Y], axis=0)

# best params
## num estimators = 400 max_depth = 80 sub_sample=0.8

# xgboost = XGBClassifier(n_estimators=400, scale_pos_weight=13,
#                              max_depth=80, verbosity=1, eval_metric='error', max_delta_step=0.15,
#                              subsample=0.8)
# xgboost.fit(full_X[features], full_Y)
# with open('xgb_model.pickle', 'wb') as f:
#     pickle.dump(xgboost, f)

