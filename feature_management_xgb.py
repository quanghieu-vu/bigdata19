import os
import utils
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from model_xgb import XGBoostModel

class FeatureManagement():

    '''
    init method
    '''
    def __init__(self):
        file_train = os.path.join("input", "cybersecurity_training.csv")
        file_test = os.path.join("input", "cybersecurity_test.csv")
        file_alerts = os.path.join("dict", "alerts.csv")

        df_alerts = pd.read_csv(file_alerts)
        df_train = pd.read_csv(file_train, sep = "|")
        df_test = pd.read_csv(file_test, sep = "|")
        df_test["notified"] = -1

        self.df_all = pd.concat([df_train, df_test], axis=0, sort=False)
        self.df_all = self.df_all.merge(df_alerts, how="inner", on="alert_ids")

        self.le_columns = [
            "client_code",
            "ip",
            "ip_zone_1",
            "ip_zone_2",
            "ip_zone_3",
            "ip_zone_4"
        ]

        self.oe_columns = [
            "categoryname",
            "ipcategory_name",
            "ipcategory_scope",
            "parent_category",
            "grandparent_category",
            "weekday",
            "dstipcategory_dominate",
            "srcipcategory_dominate",
            "dstipscope_dominate",
            "srcipscope_dominate",
            "dstportcategory_dominate",
            "srcportcategory_dominate",
            "devicetype_dominate",
            "devicevendor_code_dominate",
            "protocol_dominate",
            "alerttype_dominate",
            "reportingdevice_code_dominate",
            "lookup_protocol_dominate"
        ]

        self.nanum_columns = [
            "n1",
            "n2",
            "n3",
            "n4",
            "n5",
            "n6",
            "n7",
            "n8",
            "n9",
            "n10",
            "score"
        ]

        self.nastr_columns = [
            "devicetype_dominate",
            "devicevendor_code_dominate",
            "protocol_dominate",
            "alerttype_dominate",
            "reportingdevice_code_dominate",
            "lookup_protocol_dominate"
        ]

        # reset to try,
        self.le_columns = self.le_columns + self.oe_columns
        self.oe_columns = []

    '''
    fill na and encode features
    '''
    def process(self):
        # fill na
        for column in self.nanum_columns:
            print("Fill NA {}".format(column))
            self.df_all[column].fillna(-1, inplace=True)

        for column in self.nastr_columns:
            print("Fill NA {}".format(column))
            self.df_all[column].fillna("", inplace=True)

        # new features
        self.df_all["dstipscope_dominate"] = self.df_all.apply(lambda row: utils.get_ip_scope(row["dstipcategory_dominate"]), axis=1)
        self.df_all["srcipscope_dominate"] = self.df_all.apply(lambda row: utils.get_ip_scope(row["srcipcategory_dominate"]), axis=1)

        # ip zone features
        self.df_all["ip_zone_1"] = self.df_all.apply(lambda row: utils.get_ip_zone(row["ip"], 1), axis=1)
        self.df_all["ip_zone_2"] = self.df_all.apply(lambda row: utils.get_ip_zone(row["ip"], 2), axis=1)
        self.df_all["ip_zone_3"] = self.df_all.apply(lambda row: utils.get_ip_zone(row["ip"], 3), axis=1)
        self.df_all["ip_zone_4"] = self.df_all.apply(lambda row: utils.get_ip_zone(row["ip"], 4), axis=1)

        # concatenation features
        self.df_all["ip_zone_12"] = self.df_all.apply(lambda row: utils.concatenate_values([row["ip_zone_1"], row["ip_zone_2"]]), axis = 1)
        self.df_all["ip_zone_123"] = self.df_all.apply(lambda row: utils.concatenate_values([row["ip_zone_1"], row["ip_zone_2"], row["ip_zone_3"]]), axis = 1)
        self.df_all["categoryname_ipscope"] = self.df_all.apply(lambda row: utils.concatenate_values([row["categoryname"], row["ipcategory_scope"]]), axis = 1)
        self.df_all["categoryname_severity"] = self.df_all.apply(lambda row: utils.concatenate_values([row["categoryname"], row["overallseverity"]]), axis = 1)
        self.df_all["srcdstipscope_dominate"] = self.df_all.apply(lambda row: utils.concatenate_values([row["srcipscope_dominate"], row["dstipscope_dominate"]]), axis=1)

        self.le_columns.append("ip_zone_12")
        self.le_columns.append("ip_zone_123")
        self.le_columns.append("categoryname_ipscope")
        self.le_columns.append("categoryname_severity")
        self.le_columns.append("srcdstipscope_dominate")

        # timestamp_dist in hour and minute
        self.df_all["timestamp_hour"] = self.df_all.apply(lambda row: utils.get_duration(row["timestamp_dist"]), axis = 1)

        # ending time features
        self.df_all["end_hour"] = self.df_all.apply(lambda row: utils.get_end_time(row["start_hour"], row["start_minute"], row["start_second"], row["timestamp_dist"], "hour"), axis=1)
        self.df_all["end_minute"] = self.df_all.apply(lambda row: utils.get_end_time(row["start_hour"], row["start_minute"], row["start_second"], row["timestamp_dist"], "minute"), axis=1)
        self.df_all["end_second"] = self.df_all.apply(lambda row: utils.get_end_time(row["start_hour"], row["start_minute"], row["start_second"], row["timestamp_dist"], "second"), axis=1)

        # sum score features
        self.df_all["sum_score"] = self.df_all.apply(lambda row: utils.get_sum([row["{}score".format(score)] for score in ["untrust", "flow", "trust", "enforcement"]]), axis = 1)
        self.df_all["sum_n"] = self.df_all.apply(lambda row: utils.get_sum([row["n{}".format(i)] for i in range(1, 11)]), axis = 1)
        self.df_all["sum_p5"] = self.df_all.apply(lambda row: utils.get_sum([row["p5{}".format(p5)] for p5 in ["m", "w", "d"]]), axis = 1)
        self.df_all["sum_p8"] = self.df_all.apply(lambda row: utils.get_sum([row["p8{}".format(p8)] for p8 in ["m", "w", "d"]]), axis = 1)

        # get ratio features
        self.df_all["thrcnt_month_day"] = self.df_all.apply(lambda row: utils.get_ratio(row["thrcnt_month"], row["thrcnt_day"]), axis = 1)
        self.df_all["thrcnt_week_day"] = self.df_all.apply(lambda row: utils.get_ratio(row["thrcnt_week"], row["thrcnt_day"]), axis=1)

        # encode features with label encoder
        label_encoder = LabelEncoder()
        for column in self.le_columns:
            print("Label encoding {}".format(column))
            label_encoder.fit(self.df_all[column])
            self.df_all[column] = label_encoder.transform(self.df_all[column])

        # encode features with one-hot encoder
        for column in self.oe_columns:
            print("One-hot encoding {}".format(column))
            pd_encoded = pd.get_dummies(self.df_all[column])
            pd_encoded.columns = ["{}_{}".format(column, "_".join(str(col).lower().split())) for col in pd_encoded.columns]
            self.df_all.drop(column, axis = 1, inplace = True)
            self.df_all = pd.concat([self.df_all, pd_encoded], axis=1)

    '''
    split and save training and test files
    '''
    def split_and_save(self):
        df_all = self.df_all
        df_train = df_all[(df_all["notified"] != -1)]
        df_test = df_all[df_all["notified"] == -1]
        df_test = df_test.drop(["notified"], axis = 1)

        print("Total number of samples in train {}, test {}".format(len(df_train), len(df_test)))

        file_train = os.path.join("input", "cleaned_training.csv")
        file_test = os.path.join("input", "cleaned_test.csv")
        df_train.to_csv(file_train, index=False)
        df_test.to_csv(file_test, index=False)

if __name__ == "__main__":
    featureMan = FeatureManagement()
    featureMan.process()
    featureMan.split_and_save()

    xgb_model = XGBoostModel()
    trained_model = xgb_model.train_and_predict(is_plot=False)