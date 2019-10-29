import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

def mean_encoding(df, cols, target, alpha):
    global_mean = df[target].mean()
    for col in cols:
        df[col + '_mean'] = df[col].map(df.groupby(col)[target].count()) * df[col].map(df.groupby(col)[target].mean())
        df[col + '_mean'] += global_mean * alpha
        df[col + '_mean'] /= (df[col].map(df.groupby(col)[target].count()) + alpha)
    return df

def mean_encoding_single(train, test, cols, target, alpha):
    train = mean_encoding(train, cols, target, alpha)
    for col in cols:
        test[col + '_mean'] = test[col].map(train.groupby(col)[target].mean())
    return train, test

def mean_encoding_kfold(train, test, cols, target, alpha):
    y_tr = train[target].values
    prior = np.median(y_tr)
    for col in cols:
        train[col + '_mean'] = prior

    train_new = train.copy()
    test_new = test.copy()

    # train mapping
    kf = StratifiedKFold(n_splits=5, random_state=2019)
    for i, (tr_ind, val_ind) in enumerate(kf.split(y_tr, y_tr)):
        X_tr, X_val = train.iloc[tr_ind], train.iloc[val_ind]
        X_tr = mean_encoding(X_tr, cols, target, alpha)
        for col in cols:
            means = X_val[col].map(X_tr.groupby(col)[col + '_mean'].mean())
            X_val[col + '_mean'] = means
        train_new.iloc[val_ind] = X_val

    # test mapping
    for col in cols:
        test_new[col + '_mean'] = test_new[col].map(train_new.groupby(col)[target].mean())

    return train_new, test_new

class LightGBMModel:

    def __init__(self):
        file_train = os.path.join("input", "cleaned_training.csv")
        file_test = os.path.join("input", "cleaned_test.csv")

        df_train = pd.read_csv(file_train)
        df_test = pd.read_csv(file_test)

        main_drop_columns = [ \
            "alert_ids", "categoryname_overallseverity", \
            "ip_zone_12", "ip_zone_123", \
            "ipcategory_name", "srcipscope_dominate_dstipscope_dominate", \
            'alerttype_21', 'alerttype_41', 'srcipcategory_0', \
            'lookup_protocol_0', 'lookup_protocol_3', 'lookup_protocol_5', \
            'grandparent_category', 'alerttype_27', "timestamp_hour", \
            'ip_zone_34', 'ip_zone_234', "lookup_protocol_dominate", \
            "lookup_srcport_dominate", "lookup_dstport_dominate", \
            "username_1", "domain_1", "signature_1", \
            "matched_ipcategory", "matched_portcategory", \
            "alerttime_count" \
        ]

        # convert feature to log
        for feature in ["alerttime_sum"]:
            df_train["{}_log".format(feature)] = np.log1p(df_train[feature])
            df_test["{}_log".format(feature)] = np.log1p(df_test[feature])
            main_drop_columns.append(feature)

        df_train["count_avg"] = df_train["timestamp_dist"] / (df_train["count_sum"] + 1)
        df_test["count_avg"] = df_test["timestamp_dist"] / (df_test["count_sum"] + 1)

        tuples = [('client_code', 'ipcategory_scope', ['mean', 'std'], True), \
                  ('client_code', 'alerttime_sum', ['mean', 'std'], True), \
                  ('client_code', 'protocol_cnt', ['mean', 'std', 'count'], True), \
                  ('client_code', 'start_hour', ['mean', 'std', 'min', 'max'], True), \
                  ('client_code', 'username_cd', ['mean', 'std'], True), \
                  ('srcip_cd', 'overallseverity', ['mean', 'std'], True)]

        for (col, ncol, agg_types, div_flag) in tuples:
            for agg_type in agg_types:
                new_col_name = col + '_' + ncol + '_' + agg_type
                temp = pd.concat([df_train[[col, ncol]], df_test[[col, ncol]]])
                temp = temp.groupby([col])[ncol].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})

                temp.index = list(temp[col])
                temp = temp[new_col_name].to_dict()

                df_train[new_col_name] = df_train[col].map(temp)
                df_test[new_col_name] = df_test[col].map(temp)

                if div_flag:
                    df_train[new_col_name + '_div'] = df_train[ncol] / df_train.groupby([col])[ncol].transform(agg_type)
                    df_test[new_col_name + '_div'] = df_test[ncol] / df_test.groupby([col])[ncol].transform(agg_type)
           
        alert_drop_columns_cnt = [
            "devicetype_cnt", \
            "reportingdevice_code_cnt", "devicevendor_code_cnt", \
            "direction_cnt", "severity_cnt", "severity_dominate", \
            "domain_cnt", "username_cnt", "signature_cnt", \
            "srcipscope_cnt", "dstipscope_cnt" \
        ]

        alert_drop_columns_numna = [ \
            "alerttype_numna",
            "srcipcategory_numna", "dstipcategory_numna", \
            "srcportcategory_numna", "dstportcategory_numna", \
            "direction_numna", "severity_numna", \
            "domain_numna", \
            "username_numna", "signature_numna" \
        ] + ["lookup_devicetype_{}".format(i) for i in range(6)]

        drop_columns = main_drop_columns + alert_drop_columns_numna + alert_drop_columns_cnt
        df_train.drop(drop_columns, axis=1, inplace=True)
        df_test.drop(drop_columns, axis=1, inplace=True)

        self.df_train = df_train
        self.df_test = df_test

    def train_and_predict(self, cat_features=[]):

        early_stop = 200
        verbose_eval = 100
        num_rounds = 3000
        n_splits = 6
        X_train = self.df_train
        X_test = self.df_test

        params = {
            'objective': 'binary',
            'boosting': 'gbdt',
            'metric': 'auc',
            'num_leaves': 60,
            'max_depth': 9,
            'learning_rate': 0.005,
            'bagging_fraction': 0.85,
            'feature_fraction': 0.85,
            'verbosity': -1,
            'data_random_seed': 0
        }

        kfold = StratifiedKFold(n_splits=n_splits, random_state=2019)
        oof_train = np.zeros((X_train.shape[0]))
        oof_test = np.zeros((X_test.shape[0], n_splits))

        fold = 1
        scores = []
        feature_importance_df = pd.DataFrame()

        for train_index, valid_index in kfold.split(X_train, X_train['notified'].values):
            params["data_random_seed"] = fold

            X_tr = X_train.iloc[train_index, :]
            X_val = X_train.iloc[valid_index, :]

            y_tr = X_tr['notified'].values
            X_tr = X_tr.drop(['notified'], axis=1)

            y_val = X_val['notified'].values
            X_val = X_val.drop(['notified'], axis=1)

            print("Fold {}, shape of x1 {}, y1 {}, x2 {}, y2 {}".format(fold, X_tr.shape, len(y_tr), X_val.shape, len(y_val)))

            d_train = lgb.Dataset(X_tr, label=y_tr)
            d_valid = lgb.Dataset(X_val, label=y_val)
            watchlist = [d_train, d_valid]

            print('training LGB:')
            if len(cat_features) == 0:
                model = lgb.train(params,
                                  train_set=d_train,
                                  num_boost_round=num_rounds,
                                  valid_sets=watchlist,
                                  verbose_eval=verbose_eval,
                                  early_stopping_rounds=early_stop)
            else:
                model = lgb.train(params,
                                  train_set=d_train,
                                  num_boost_round=num_rounds,
                                  valid_sets=watchlist,
                                  verbose_eval=verbose_eval,
                                  categorical_feature=list(cat_features),
                                  early_stopping_rounds=early_stop)

            fold_importance_df = pd.DataFrame()
            fold_importance_df['feature'] = X_tr.columns.values
            fold_importance_df['importance'] = model.feature_importance()
            fold_importance_df['fold'] = fold
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            test_pred = model.predict(X_test, num_iteration=model.best_iteration)

            fpr, tpr, thresholds = metrics.roc_curve(y_val, val_pred, pos_label=1)
            score = metrics.auc(fpr, tpr)
            scores.append(score)

            oof_train[valid_index] = val_pred
            oof_test[:, fold - 1] = test_pred
            fold += 1

        preds = oof_test.mean(axis=1)
        output_file = os.path.join("output", "submission_lgb.txt")
        fout = open(output_file, "w")
        fout.writelines(["{}\n".format(pred) for pred in preds])

        score_mean = sum(scores) / len(scores)
        print("Mean of validation scores: {}, {}".format(score_mean, scores))

        feature_file = os.path.join("output", "feature_importance.csv")
        feature_importance_df.to_csv(feature_file, index=False)

if __name__ == "__main__":
    lgb_model = LightGBMModel()
    trained_model = lgb_model.train_and_predict()