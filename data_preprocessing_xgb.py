import os, re
import gc
import pandas as pd
import utils

from model_xgb import XGBoostModel
from feature_management_xgb import FeatureManagement

class DataPreprocessing():

    '''
    init
    '''
    def __init__(self):
        self.file_alerts = os.path.join("input", "localized_alerts_data.csv")
        self.file_dict = os.path.join("dict", "alerts.csv")

        self.lookup_protocols = sorted(["tcp", "udp", "ssh", "https", "ftp", "dns", "imap", "pop", "smtp", "ntp"])
        self.lookup_devicetypes = sorted(["fw", "dns", "nids"])
        self.lookup_reportingdevicecodes = sorted(["yxz", "gnl", "kfo", "adi", "jwl", "dim", "lyf"])
        self.lookup_srcports = ["l100", "l200", "l500"]
        self.lookup_dstports = ["l100", "l200", "l500"]

    def eda(self):
        for column in ["alerttype", "devicetype", "reportingdevice_code", "devicevendor_code", \
            "srcip", "dstip", "srcipcategory", "dstipcategory", \
            "srcport", "dstport", "srcportcategory", "dstportcategory", \
            "direction", "severity", "domain", "protocol", "username", "signature", \
            "alerttime", "count"]:
            df_alerts = pd.read_csv(self.file_alerts, sep="|", usecols = [column])
            values = df_alerts[column].value_counts()
            print("Total number values of {}: {}".format(column, len(values)))
            print(values)

            del df_alerts
            gc.collect()

    '''
    get mapping dict of a column
    return list of strings
    '''
    def get_values(self, column):
        if column == "lookup_protocol":
            '''
            limit = 50
            k = Counter(self.lookup_protocols)
            print(k.most_common(129))
            selected_protocols = k.most_common(limit)
            print(selected_protocols)
            return sorted([protocol[0] for protocol in selected_protocols])
            '''
            return self.lookup_protocols + ["others"]
        elif column == "lookup_devicetype":
            return self.lookup_devicetypes + ["others"]
        elif column == "lookup_reportingdevicecode":
            return self.lookup_reportingdevicecodes + ["others"]
        elif column == "lookup_srcport":
            return self.lookup_dstports + ["others"]
        elif column == "lookup_dstport":
            return self.lookup_dstports + ["others"]
        else:
            df_alerts = pd.read_csv(self.file_alerts, sep="|", usecols=[column])
            return sorted([str(item).lower() for item in df_alerts[column].dropna().unique()])

    '''
    build alert dictionary    
    '''
    def build_alert_dict(self):
        # open files and write header
        fin = open(self.file_alerts)
        fin.readline()

        dict_mappings = {
            "alerttype": 1,
            "devicetype": 2,
            "reportingdevice_code": 3,
            "devicevendor_code": 4,
            "srcip": 5,
            "dstip": 6,
            "srcipcategory": 7,
            "dstipcategory": 8,
            "srcport": 9,
            "dstport": 10,
            "srcportcategory": 11,
            "dstportcategory": 12,
            "direction": 13,
            "severity": 15,
            "domain": 17,
            "protocol": 18,
            "username": 19,
            "signature": 20
        }
        dict_sorted_keys = sorted(dict_mappings.keys())

        detailed_columns = ["alerttype", "srcipcategory", "dstipcategory", \
                            "srcportcategory", "dstportcategory", \
                            "lookup_protocol", "lookup_devicetype", \
                            "lookup_reportingdevicecode", \
                            "lookup_srcport", "lookup_dstport"]
        dominate_columns = ["alerttype", "severity", "devicetype", "devicevendor_code", \
                            "reportingdevice_code", "protocol", "lookup_protocol"]

        alert_dict = dict()
        total_line = 1
        while True:
            line = fin.readline().strip()
            if line == '':
                break
            total_line += 1
            params = line.split("|")
            alert_id = params[0]
            alerttime = int(params[14])
            count = int(params[16])

            tmp_protocol = re.sub(r'[^a-zA-Z]+', ' ', params[dict_mappings["protocol"]]).strip()
            lookup_protocol = "others"
            if tmp_protocol != '':
                for protocol in tmp_protocol.split():
                    if protocol in self.lookup_protocols:
                        lookup_protocol = protocol
                        break

            lookup_devicetype = "others"
            for devicetype in self.lookup_devicetypes:
                if devicetype == params[dict_mappings["devicetype"]].lower():
                    lookup_devicetype = devicetype
                    break

            lookup_reportingdevicecode = "others"
            for reportingdevicecode in self.lookup_reportingdevicecodes:
                if reportingdevicecode == params[dict_mappings["reportingdevice_code"]].lower():
                    lookup_reportingdevicecode = reportingdevicecode
                    break

            lookup_dstport = "others"
            if params[dict_mappings["dstport"]] != '':
                dstport_value = int(params[dict_mappings["dstport"]])
                if dstport_value < 100:
                    lookup_dstport = "l100"
                elif dstport_value < 200:
                    lookup_dstport = "l200"
                elif dstport_value < 500:
                    lookup_dstport = "l500"

            lookup_srcport = "others"
            if params[dict_mappings["srcport"]] != '':
                srcport_value = int(params[dict_mappings["srcport"]])
                if srcport_value < 100:
                    lookup_srcport = "l100"
                elif srcport_value < 200:
                    lookup_srcport = "l200"
                elif srcport_value < 500:
                    lookup_srcport = "l500"

            severity_level = int(params[dict_mappings["severity"]])
            srcipscope = utils.get_ip_scope(params[dict_mappings["srcipcategory"]]).lower()
            dstipscope = utils.get_ip_scope(params[dict_mappings["dstipcategory"]]).lower()

            if alert_id not in alert_dict:
                alert_dict[alert_id] = dict()
                alert_dict[alert_id]["alerttime"] = 0
                alert_dict[alert_id]["count"] = 0
                alert_dict[alert_id]["severity_level"] = 0
                for key in dict_sorted_keys:
                    alert_dict[alert_id][key] = dict()
                alert_dict[alert_id]["srcipscope"] = dict()
                alert_dict[alert_id]["dstipscope"] = dict()
                alert_dict[alert_id]["lookup_protocol"] = dict()
                alert_dict[alert_id]["lookup_devicetype"] = dict()
                alert_dict[alert_id]["lookup_reportingdevicecode"] = dict()
                alert_dict[alert_id]["lookup_srcport"] = dict()
                alert_dict[alert_id]["lookup_dstport"] = dict()

            alert_dict[alert_id]["alerttime"] += alerttime
            alert_dict[alert_id]["count"] += count
            alert_dict[alert_id]["severity_level"] += severity_level
            for key in dict_sorted_keys:
                value = params[dict_mappings[key]].lower()
                if value not in alert_dict[alert_id][key]:
                    alert_dict[alert_id][key][value] = 0
                alert_dict[alert_id][key][value] += 1

            # adding source ip scope
            if srcipscope not in alert_dict[alert_id]["srcipscope"]:
                alert_dict[alert_id]["srcipscope"][srcipscope] = 0
            alert_dict[alert_id]["srcipscope"][srcipscope] += 1

            # adding destination ip scope
            if dstipscope not in alert_dict[alert_id]["dstipscope"]:
                alert_dict[alert_id]["dstipscope"][dstipscope] = 0
            alert_dict[alert_id]["dstipscope"][dstipscope] += 1

            # adding protocol
            if lookup_protocol not in alert_dict[alert_id]["lookup_protocol"]:
                alert_dict[alert_id]["lookup_protocol"][lookup_protocol] = 0
            alert_dict[alert_id]["lookup_protocol"][lookup_protocol] += 1

            # adding devicetype
            if lookup_devicetype not in alert_dict[alert_id]["lookup_devicetype"]:
                alert_dict[alert_id]["lookup_devicetype"][lookup_devicetype] = 0
            alert_dict[alert_id]["lookup_devicetype"][lookup_devicetype] += 1

            # adding reportingdevicecode
            if lookup_reportingdevicecode not in alert_dict[alert_id]["lookup_reportingdevicecode"]:
                alert_dict[alert_id]["lookup_reportingdevicecode"][lookup_reportingdevicecode] = 0
            alert_dict[alert_id]["lookup_reportingdevicecode"][lookup_reportingdevicecode] += 1

            if lookup_dstport not in alert_dict[alert_id]["lookup_dstport"]:
                alert_dict[alert_id]["lookup_dstport"][lookup_dstport] = 0
            alert_dict[alert_id]["lookup_dstport"][lookup_dstport] += 1

            if lookup_srcport not in alert_dict[alert_id]["lookup_srcport"]:
                alert_dict[alert_id]["lookup_srcport"][lookup_srcport] = 0
            alert_dict[alert_id]["lookup_srcport"][lookup_srcport] += 1

            if total_line % 100000 == 0:
                print("Finished processing of {} lines".format(total_line))

        fin.close()

        selected_columns_values = []
        for i in range(len(detailed_columns)):
            column = detailed_columns[i]
            distinct_values = self.get_values(column)
            selected_columns_values.append(distinct_values)

            print("{} has {} values".format(column, len(distinct_values)))
            print(distinct_values)

        # output dictionary to csv file
        fout = open(self.file_dict, "w")
        fout.write("alert_ids,alerttime_sum,count_sum")

        for key in dict_sorted_keys:
            fout.write(",{}_cnt".format(key))

        fout.write(",srcipscope_cnt")
        fout.write(",dstipscope_cnt")
        fout.write(",lookup_protocol_cnt")

        # numna features
        for key in dict_sorted_keys:
            fout.write(",{}_numna".format(key))

        for i in range(len(detailed_columns)):
            column = detailed_columns[i]
            distinct_values = selected_columns_values[i]
            for j in range(len(distinct_values)):
                fout.write(",{}_{}".format(column, j))

        for i in range(len(dominate_columns)):
            column = dominate_columns[i]
            fout.write(",{}_dominate".format(column))

        # negative domain, username and signature count
        fout.write(",domain_0,username_0,signature_0")

        # finish headers
        fout.write("\n")

        test_count = 10
        for alert_id in sorted(alert_dict.keys()):

            if test_count > 0:
                print(alert_id)
                print(alert_dict[alert_id])
                test_count -= 1

            alerttime_sum = alert_dict[alert_id]["alerttime"]
            count_sum = alert_dict[alert_id]["count"]
            fout.write("{},{},{}".format(alert_id, alerttime_sum, count_sum))

            for key in dict_sorted_keys:
                values = list(alert_dict[alert_id][key].values())
                fout.write(",{}".format(len(values)))

            fout.write(",{}".format(len(alert_dict[alert_id]["srcipscope"])))
            fout.write(",{}".format(len(alert_dict[alert_id]["dstipscope"])))
            fout.write(",{}".format(len(alert_dict[alert_id]["lookup_protocol"])))

            # numna features
            for key in dict_sorted_keys:
                if "" in alert_dict[alert_id][key]:
                    fout.write(",{}".format(alert_dict[alert_id][key][""]))
                else:
                    fout.write(",0")

            # detailed features
            for i in range(len(detailed_columns)):
                column = detailed_columns[i]
                distinct_values = selected_columns_values[i]
                for j in range(len(distinct_values)):
                    value = distinct_values[j]
                    if value in alert_dict[alert_id][column]:
                        fout.write(",{}".format(alert_dict[alert_id][column][value]))
                    else:
                        fout.write(",0")

            # dominate values
            for i in range(len(dominate_columns)):
                column = dominate_columns[i]
                value = max(alert_dict[alert_id][column], key=alert_dict[alert_id][column].get)
                fout.write(",{}".format(value))

            # count negative values of domain, username, and signature
            for column in ["domain", "username", "signature"]:
                if "0" in alert_dict[alert_id][column]:
                    fout.write(",{}".format(alert_dict[alert_id][column]["0"]))
                else:
                    fout.write(",0")

            fout.write("\n")
        fout.close()

        print("Total number of protocols: {}".format(len(self.lookup_protocols)))
        print(self.get_values("lookup_protocol"))


if __name__ == "__main__":

    dataPreProcessing = DataPreprocessing()
    dataPreProcessing.build_alert_dict()

    # also update features
    featureMan = FeatureManagement()
    featureMan.process()
    featureMan.split_and_save()

    #train and valid model
    xgb_model = XGBoostModel()
    trained_model = xgb_model.train_and_predict(is_plot=False)
