import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb

from scipy.sparse import csc_matrix
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

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

class XGBoostModel:

    def __init__(self):
        file_train = os.path.join("input", "cleaned_training.csv")
        file_test = os.path.join("input", "cleaned_test.csv")

        df_train = pd.read_csv(file_train)
        df_test = pd.read_csv(file_test)

        main_drop_columns = ["alert_ids", "lookup_protocol_dominate"] + \
            ["lookup_srcport_{}".format(i) for i in range(4)] + \
            ["lookup_dstport_{}".format(i) for i in range(4)]

        df_train["count_avg"] = df_train["timestamp_dist"] / (df_train["count_sum"] + 1)
        df_test["count_avg"] = df_test["timestamp_dist"] / (df_test["count_sum"] + 1)

        tripples = [('client_code', 'ipcategory_scope', ['mean', 'std']), \
                    ('client_code', 'alerttime_sum', ['mean', 'std']), \
                    ('client_code', 'protocol_cnt', ['mean', 'std', 'count']), \
                    ('client_code', 'start_hour', ['mean', 'std', 'min', 'max']), \
                    ('client_code', 'username_cd', ['mean', 'std']), \
                    ('srcip_cd', 'overallseverity', ['mean', 'std'])]

        for (col, ncol, agg_types) in tripples:
            for agg_type in agg_types:
                new_col_name = col + '_' + ncol + '_' + agg_type
                temp = pd.concat([df_train[[col, ncol]], df_test[[col, ncol]]])
                temp = temp.groupby([col])[ncol].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})

                temp.index = list(temp[col])
                temp = temp[new_col_name].to_dict()

                df_train[new_col_name] = df_train[col].map(temp)
                df_test[new_col_name] = df_test[col].map(temp)

                df_train[new_col_name + '_div'] = df_train[ncol] / df_train.groupby([col])[ncol].transform(agg_type)
                df_test[new_col_name + '_div'] = df_test[ncol] / df_test.groupby([col])[ncol].transform(agg_type)

        alert_drop_columns_cnt = [
            "devicetype_cnt", "alerttype_27", \
            "reportingdevice_code_cnt", "devicevendor_code_cnt", \
            "direction_cnt", "severity_cnt", \
            "domain_cnt", "username_cnt", "signature_cnt", \
            "srcipscope_cnt", "dstipscope_cnt", "lookup_protocol_cnt", \
            "devicetype_dominate", "devicevendor_code_dominate", \
            "protocol_dominate", "reportingdevice_code_dominate", \
            "severity_dominate", "alerttype_dominate" \
        ]

        alert_drop_columns_numna = [ \
            "alerttype_numna", "devicetype_numna", \
            "reportingdevice_code_numna", "devicevendor_code_numna", \
            "srcip_numna", "dstip_numna", \
            "srcipcategory_numna", "dstipcategory_numna", \
            "srcport_numna", "dstport_numna", \
            "srcportcategory_numna", "dstportcategory_numna", \
            "direction_numna", "severity_numna", \
            "domain_numna", "protocol_numna", \
            "username_numna", "signature_numna" \
        ] + ["lookup_devicetype_{}".format(i) for i in range(4)] \
        + ["lookup_reportingdevicecode_{}".format(i) for i in range(8)]

        drop_columns = main_drop_columns + alert_drop_columns_cnt + alert_drop_columns_numna
        df_train.drop(drop_columns, axis=1, inplace=True)
        df_test.drop(drop_columns, axis=1, inplace=True)

        self.df_train = df_train
        self.df_test = df_test

    '''
    train xgboost model
    '''
    def train_and_predict(self, is_plot = False):

        # get data for training
        x = self.df_train
        features = list(x.columns.values)
        features.remove("notified")
        print("Number of features: {}".format(len(features)))

        denom = 0
        fold = 5
        scores = []

        kfold = StratifiedKFold(n_splits=fold, random_state=2019)
        for train_index, valid_index in kfold.split(x, x['notified'].values):

            # params
            params = {
                'eta': 0.01,
                'max_depth': 8,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'seed': denom + 1,
                'silent': True,
                'subsample': 0.85,
                'colsample_bytree': 0.75,
                'colsample_bylevel': 0.75
            }

            x1 = x.iloc[train_index, :]
            x2 = x.iloc[valid_index, :]

            y1 = x1['notified'].values
            x1 = x1.drop(['notified'], axis=1)

            y2 = x2['notified'].values
            x2 = x2.drop(['notified'], axis=1)

            print("Fold {}, shape of x1 {}, y1 {}, x2 {}, y2 {}".format(denom + 1, x1.shape, len(y1), x2.shape, len(y2)))

            x1 = csc_matrix(x1)
            x2 = csc_matrix(x2)

            watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
            model = xgb.train(params, xgb.DMatrix(x1, y1), 2000, watchlist, verbose_eval=50, early_stopping_rounds=100)
            pred_valid = model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit + 10)
            fpr, tpr, thresholds = metrics.roc_curve(y2, pred_valid, pos_label=1)
            score = metrics.auc(fpr, tpr)
            scores.append(score)

            # show important features
            if is_plot:
                import operator
                import matplotlib.pyplot as plt
                outfile = open('xgb.fmap', 'w')
                i = 0
                for feat in features:
                    outfile.write('{0}\t{1}\tq\n'.format(i, feat))
                    i = i + 1
                outfile.close()
                importance = model.get_fscore(fmap='xgb.fmap')
                importance = sorted(importance.items(), key=operator.itemgetter(1))
                df = pd.DataFrame(importance, columns=['feature', 'fscore'])
                df.to_csv("feature_scores_{}.csv".format(denom + 1), index=False)
                df['fscore'] = df['fscore'] / df['fscore'].sum()

                # plot it up
                plt.figure()
                df.plot()
                df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
                plt.title('XGBoost Feature Importance')
                plt.xlabel('relative importance')
                plt.gcf().savefig(os.path.join("output", 'feature_importance_xgb.png'))
                plt.show()

            # get test data for prediction
            test = csc_matrix(self.df_test)
            print("Shape of test {}".format(test.shape))
            if denom != 0:
                pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit + 10)
                preds += pred
            else:
                pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit + 10)
                preds = pred.copy()
            denom += 1

            del test
            gc.collect()

        preds = preds / denom

        output_file = os.path.join("output", "submission_xgb.txt")
        fout = open(output_file, "w")
        fout.writelines(["{}\n".format(pred) for pred in preds])

        score_mean = sum(scores) / len(scores)
        print("Mean of validation scores: {}, {}".format(score_mean, scores))

    '''
    analyze features
    '''
    def analyze_features(self):

        # get data for training
        x = self.df_train
        features = list(x.columns.values)
        features.remove("notified")
        print("Number of features: {}".format(len(features)))

        # params
        params = {
            'eta': 0.01,
            'max_depth': 9,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'seed': 2019,
            'silent': True,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'colsample_bylevel': 0.85
        }

        y = x['notified'].values
        x = x.drop(['notified'], axis=1)
        x = csc_matrix(x)

        watchlist = [(xgb.DMatrix(x, y), 'train')]
        model = xgb.train(params, xgb.DMatrix(x, y), 1800, watchlist, verbose_eval=50, early_stopping_rounds=100)

        import operator
        import matplotlib.pyplot as plt
        outfile = open('xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()
        importance = model.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df.to_csv("feature_scores.csv", index=False)
        df['fscore'] = df['fscore'] / df['fscore'].sum()

        # plot it up
        plt.figure()
        df.plot()
        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative importance')
        plt.gcf().savefig(os.path.join("output", 'feature_importance_xgb.png'))
        plt.show()

if __name__ == "__main__":
    xgb_model = XGBoostModel()
    trained_model = xgb_model.train_and_predict(is_plot=False)