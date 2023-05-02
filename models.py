import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


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
    def __init__(self, n, max_depth, features_set):
        self.n = n
        self.max_depth= max_depth
        self.features_set = features_set
        self.xgboost = XGBClassifier(n_estimators=500, use_label_encoder=False, scale_pos_weight=12,
                            max_depth=8,verbosity=1, eval_metric='error', max_delta_step=0.15,
                            subsample=None, alpha=0)

    def train_xgb(self, train_X, train_Y, val_X, val_Y):
        self.xgboost.fit(train_X[self.features_set], train_Y)
        y_pred = self.xgboost.predict(val_X[self.features_set])
        score = f1_score(val_Y, y_pred)
        return score


def training_models(train_X, train_Y, val_X, val_Y, features_set):
    knn_f1_scores = []
    rf_f1_scores = {}
    xgb_f1_scores = {}
    ## train knn models
    for k in range(1, 25):
        knn = KNN(k, features_set)
        score = knn.train_knn(train_X, train_Y, val_X, val_Y)
        knn_f1_scores.append(score)
    best_score_knn = np.max(knn_f1_scores)
    best_k = np.argmax(knn_f1_scores) + 1

    ## train rf models
    max_depths = [40, 50, 80, 100, 150, 200]
    n_estimators = [50, 100, 150, 200, 300]
    for n in n_estimators:
        for max_dep in max_depths:
            rf = RandomForest(n, max_dep, features_set)
            score = rf.train_rf(train_X, train_Y, val_X, val_Y)
            rf_f1_scores[(n, max_dep)] = score
    best_n_max_depth_rf  = max(rf_f1_scores, key=lambda k: rf_f1_scores[k])
    best_score_rf = rf_f1_scores[best_n_max_depth_rf]

    ## train xgb models
    max_depths = [40, 50, 80, 100, 150, 200]
    n_estimators = [50, 100, 150, 200, 300]
    for n in n_estimators:
        for max_dep in max_depths:
            xgb = XGBOOST(n, max_dep, features_set)
            score = xgb.train_xgb(train_X, train_Y, val_X, val_Y)
            xgb_f1_scores[(n, max_dep)] = score
    best_n_max_depth_xgb  = max(rf_f1_scores, key=lambda k: xgb_f1_scores[k])
    best_score_xgb = xgb_f1_scores[best_n_max_depth_xgb]





    return best_score_knn, best_k, best_score_knn, best_n_max_depth_rf, best_score_rf, \
        best_n_max_depth_xgb,best_score_xgb

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






def evaluate_all_models(train_X, train_Y, val_X, val_Y, features_set, test_X, test_Y):
    ##evaluate knn
    ## add loop over all the different imputations and featues sets
    f1_scores, best_k, best_score = training_models(train_X, train_Y, val_X, val_Y, features_set)
    print(f"best knn model on validation set is with: {best_k} nbrs")
    knn = KNN(best_k)
    test_y_pred= knn.knn_model.predict(test_X)
    test_f1_score = f1_score(test_Y, test_y_pred)
    test_acuuracy = accuracy_score(test_Y, test_y_pred)
    test_recall = recall_score(test_Y, test_y_pred)
    print(f"F1 score: {test_f1_score}, test accuracy: {test_acuuracy}, recall: {test_recall}")


