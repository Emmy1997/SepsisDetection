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



class KNN:
    def __init__(self, k, features_set):
        self.k = k
        self.features_set = features_set
        self.knn_model = KNeighborsClassifier(n_neighbors=self.k)


    def train_knn(self, train_X, train_Y, val_X, val_Y):
        self.knn_model.fit(train_X[self.features_set], train_Y)
        y_pred = self.knn_model.predict(val_X[self.features_set])
        score = f1_score(val_Y, y_pred)
        return score

class RandomForest:
    def __init__(self, n, max_depth, features_set):
        self.n = n
        self.max_depth= max_depth
        self.features_set = features_set
        self.rf_model = RandomForestClassifier(n_estimators=self.n, max_depth=self.max_depth, verbose=1, random_state=6)


    def train_rf(self, train_X, train_Y, val_X, val_Y):
        self.rf_model.fit(train_X[self.features_set], train_Y)
        y_pred = self.rf_model.predict(val_X[self.features_set])
        score = f1_score(val_Y, y_pred)
        return score

class XGBOOST:
    def __init__(self, n, max_depth, subsample, features_set):
        self.n = n
        self.max_depth= max_depth
        self.subsample= subsample
        self.features_set = features_set
        self.xgboost = XGBClassifier(n_estimators=self.n, scale_pos_weight=12,
                            max_depth=self.max_depth, verbosity=1, eval_metric='error', max_delta_step=0.15,
                            subsample=self.subsample)

    def train_xgb(self, train_X, train_Y, val_X, val_Y):
        self.xgboost.fit(train_X[self.features_set], train_Y)
        y_pred = self.xgboost.predict(val_X[self.features_set])
        score = f1_score(val_Y, y_pred)
        return score


class CATBOOST:
    def __init__(self, n, max_depth, subsample, reg, lr, features_set):
        self.n = n
        self.max_depth= max_depth
        self.features_set = features_set
        self.subsample = subsample
        self.l2_reg= reg
        self.lr = lr
        self.catboost = CatBoostClassifier(n_estimators=self.n, l2_leaf_reg =self.l2_reg , scale_pos_weight=12,
                            depth=self.max_depth, subsample= self.subsample, learning_rate= self.lr)

    def train_cat(self, train_X, train_Y, val_X, val_Y):
        self.catboost.fit(train_X[self.features_set], train_Y)
        y_pred = self.catboost.predict(val_X[self.features_set])
        score = f1_score(val_Y, y_pred)
        return score


def min_max_scaler(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df


def training_models(train_X, train_Y, val_X, val_Y, features_set):
    knn_f1_scores = []
    rf_f1_scores, xgb_f1_scores, cat_f1_scores = {}, {}, {}
    # ## train knn models
    # train_scaled, val_scaled = min_max_scaler(train_X), min_max_scaler(val_X)
    # print('running KNN')
    # for k in tqdm(range(1, 25), desc='KNN'):
    #     print(f"running KNN with k: {k}")
    #     knn = KNN(k, features_set)
    #     score = knn.train_knn(train_scaled, train_Y, val_scaled, val_Y)
    #     knn_f1_scores.append(score)
    #     print(f"f1 score: {score}")
    # best_score_knn = np.max(knn_f1_scores)
    # best_k = np.argmax(knn_f1_scores) + 1
    # print(f"best score: {best_score_knn}, with k: {best_k}")
    #
    # ## train rf models
    # max_depths = [5, 10, 15, 20, 25, 30, 35, 40]
    # n_estimators = [25, 50, 100, 150, 200, 300]
    # print('running RandomForest')
    # for n in tqdm(n_estimators, desc='RandomForest'):
    #     for max_dep in max_depths:
    #         print("running RandomForest with params:")
    #         print(f" n: {n}, maxdepth: {max_dep}")
    #         rf = RandomForest(n, max_dep, features_set)
    #         score = rf.train_rf(train_X, train_Y, val_X, val_Y)
    #         rf_f1_scores[(n, max_dep)] = score
    #         print(f"f1 score: {score}")
    # best_n_max_depth_rf  = max(rf_f1_scores, key=lambda k: rf_f1_scores[k])
    # best_score_rf = rf_f1_scores[best_n_max_depth_rf]
    # print(f"best rf score: {best_score_rf}, with params: {best_n_max_depth_rf}")

    ## train xgb models
    max_depths = [5, 10, 15, 20, 25, 30, 35, 40]
    n_estimators = [25, 50, 100, 150, 200, 300]
    sub_sample = [0.5, 0.6, 0.7, 0.8, 0.9, None]
    print('running XGBOOST')
    for s in tqdm(sub_sample, desc='XGBOOST'):
        for n in n_estimators:
            for max_dep in max_depths:
                print("running XGBOOST with params:")
                print(f" subsmpl: {s}, n: {n}, maxdepth: {max_dep}")
                xgb = XGBOOST(n, max_dep, s, features_set)
                score = xgb.train_xgb(train_X, train_Y, val_X, val_Y)
                xgb_f1_scores[(n, max_dep, s)] = score
                print(f"f1 score: {score}")
    best_params_xgb  = max(xgb_f1_scores, key=lambda k: xgb_f1_scores[k])
    best_score_xgb = xgb_f1_scores[best_params_xgb]
    print(f"best xgb score: {best_score_xgb}, with params: {best_params_xgb}")

    ## train CatBoost model
    max_depths = [5, 8, 10, 12, 13, 15, 16]
    n_estimators = [25, 50, 100, 150, 200, 300]
    sub_sample = [0.5, 0.6, 0.7, 0.8, 0.9, None]
    reg = [0.01, 0.1, 1.0, 10.0, 100.0]
    lrs= [0.01, 0.05, 0.1, 0.2, 0.3]
    print("running Catboost")
    for lr in tqdm(lrs, desc='XGBOOST'):
        for r in reg:
            for s in sub_sample:
                for n in n_estimators:
                    for max_dep in max_depths:
                        print("running CATBOOST with params:")
                        print(f" lr: {lr}, regu :{r}, subsmpl: {s}, n: {n}, maxdepth: {max_dep}")
                        cat = CATBOOST(n, max_dep, s, r, lr, features_set)
                        score = cat.train_cat(train_X, train_Y, val_X, val_Y)
                        cat_f1_scores[(n, max_dep, s, r, lr)] = score
    best_params_cat  = max(cat_f1_scores, key=lambda k: cat_f1_scores[k])
    best_score_cat = cat_f1_scores[best_params_cat]
    print(f"best cat score: {best_score_cat}, with params: {best_params_cat}")


    return best_k, best_score_knn, knn_f1_scores,\
        best_n_max_depth_rf, best_score_rf, rf_f1_scores, \
        best_params_xgb, best_score_xgb, xgb_f1_scores,\
        best_params_cat,  best_score_cat, cat_f1_scores

#
#
# For ig_key in ['A', 'B', 'C']:
#     for n in range(50, 200, 10):
#         rf = RandomForestClassifier(n_estimators=n, verbose=1, random_state=6)
#         if ig_key == 'A':
#             rf.fit(all_data_means, Y_train)
#             y_pred = rf.predict(all_data_means_test)
#         else:
#             rf.fit(all_data_means.drop(columns=ignores[ig_key]), Y_train)
#             y_pred = rf.predict(all_data_means_test.drop(columns=ignores[ig_key]))
#         new_score = f1_score(Y_test, y_pred)
#         f1_scores[ig_key].append(new_score)
#         if new_score > score:
#             score = new_score
#             n_estimators = n
#             ignore_type = ig_key
#
# print('f1 test score:', score)
# print(f"n_estimators={n_estimators}, ignore_type={ignore_type}")






def evaluate_all_models(train_X, train_Y, val_X, val_Y, features_set, i):
    ##evaluate knn
    ## add loop over all the different imputations and featues sets
    best_k, best_score_knn, knn_f1_scores, \
        best_n_max_depth_rf, best_score_rf, rf_f1_scores, \
        best_params_xgb, best_score_xgb, xgb_f1_scores, \
        best_params_cat, best_score_cat, cat_f1_scores = training_models(train_X, train_Y, val_X, val_Y, features_set)
    # print(f"best knn model on validation set is with: {best_k} nbrs")
    # knn = KNN(best_k)
    # test_y_pred= knn.knn_model.predict(test_X)
    # test_f1_score = f1_score(test_Y, test_y_pred)
    # test_acuuracy = accuracy_score(test_Y, test_y_pred)
    # test_recall = recall_score(test_Y, test_y_pred)
    # print(f"F1 score: {test_f1_score}, test accuracy: {test_acuuracy}, recall: {test_recall}")
    results_dict = {'KNN': [best_k, best_score_knn, knn_f1_scores],
                    'RF': [best_n_max_depth_rf, best_score_rf, rf_f1_scores],
                    'XGB': [best_params_xgb, best_score_xgb, xgb_f1_scores],
                    'CAT': [best_params_cat, best_score_cat, cat_f1_scores]}
    with open(f'results_dict_experiment_{i}.json', 'w') as f:
        json.dump(results_dict, f)


# train_df_filtered = pd.read_csv('train_df_filtered.csv')
# train_df_filtered = train_df_filtered[['patient', 'y']]
# train_df_filtered = train_df_filtered.groupby(['patient']).max(['y']).reset_index()
for i in [4, 5]:
    print(f"Running Hyperparam tune on experiment {i}")
    train_df = pd.read_csv(f'transformed_files/experiment_{i}/train_transformed.csv')
    val_df = pd.read_csv(f'transformed_files/experiment_{i}/val_transformed.csv')
    # train_df_with_labeles = pd.merge(train_df, train_df_filtered, on='patient',how='inner')
    # val_df_with_labeles = pd.merge(val_df, train_df_filtered, on='patient',how='inner')

    features = list(set(["Age", "Gender", "HospAdmTime_final", "ICULOS_final", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "SIRS"]))
    val_df_p2y = val_df[['index', 'patient', 'y']]
    # val_df_p2y.to_csv('val_p2y')
    train_Y= train_df[['y']]
    train_X = train_df.drop(columns=['patient', 'y'])
    val_Y = val_df[['y']]
    val_X = val_df.drop(columns=['patient', 'y'])
    evaluate_all_models(train_X, train_Y.values.ravel(), val_X, val_Y.values.ravel(), features, i)



