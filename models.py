from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score
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
        self.rf_model = RandomForestClassifier(n_estimators=self.n, max_depth=self.max_depth, verbose=1, random_state=6,
                                               class_weight='balanced')


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
        self.xgboost = XGBClassifier(n_estimators=self.n, scale_pos_weight=13,
                            max_depth=self.max_depth, verbosity=0, eval_metric='error', max_delta_step=0.15,



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
        self.catboost = CatBoostClassifier(n_estimators=self.n, loss_function='Logloss',
                                           od_type ='Iter', od_wait= 50, random_state= 72, eval_metric='F1',
                                           verbose= False, l2_leaf_reg =self.l2_reg ,
                                        depth=self.max_depth, learning_rate= self.lr,
                                           class_weights={0: 1, 1: 13})


    def train_cat(self, train_X, train_Y, val_X, val_Y):
        self.catboost.fit(train_X[self.features_set], train_Y, eval_set = (val_X[self.features_set], val_Y))
        print(self.catboost.best_iteration_)
        y_pred = self.catboost.predict(val_X[self.features_set])
        score = f1_score(val_Y, y_pred)
        return score, self.catboost.best_iteration_
