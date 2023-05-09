import json
import pandas as pd
import pickle
from sklearn.metrics import f1_score


def main():
    test_df_transformed = pd.read_csv("../test_df_transformed.csv")
    model = pickle.load(open("../xgb_model.pickle", "rb"))
    cols = model.get_booster().feature_names

    ages_groups = {'child': list(range(0, 10)),
                   'youth': list(range(10, 18)),
                   'young adult': list(range(18, 28)),
                   'adult': list(range(28, 40)),
                   'mature adult': list(range(40, 57)),
                   'old adult': list(range(57, 80)),
                   'very old': list(range(80, 120)),
                   }
    results_dict = {}
    for age_name, age_range in ages_groups.items():
        rel_test = test_df_transformed[test_df_transformed.Age.isin(age_range)]
        score = 0
        if len(rel_test) > 0:
            y_true = rel_test['y'].values
            print(y_true[:5])
            cols_to_drop = ['patient', 'y', 'index', "level_0", 'SepsisLabel']
            cols_to_drop = list(set(rel_test.columns) & set(cols_to_drop))
            test_df_transformed_clean = rel_test.drop(columns=cols_to_drop)
            y_pred = model.predict(test_df_transformed_clean[cols])
            print(y_pred[:5])

            score = f1_score(y_true, y_pred)
        results_dict[age_name] = round(score, 4)

    with open('results_dict_agegroups.json', 'w') as fp:
        json.dump(results_dict, fp)


if __name__ == '__main__':
    main()