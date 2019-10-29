import os

input_xgb = os.path.join("output", "submission_xgb.txt")
input_lgb = os.path.join("output", "submission_lgb.txt")
output_file = os.path.join("output", "submission_ensemble.txt")

fin_xgb = open(input_xgb, "r")
fin_lgb = open(input_lgb, "r")
fout = open(output_file, "w")

while True:
    line_xgb = fin_xgb.readline().strip()
    line_lgb = fin_lgb.readline().strip()

    if line_xgb == '' or line_lgb == '':
        break

    output_value = (float(line_xgb) + float(line_lgb)) / 2.0
    fout.write("{}\n".format(output_value))

fin_xgb.close()
fin_lgb.close()
fout.close()