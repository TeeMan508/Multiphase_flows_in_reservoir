import os
from hyperopt import hp, fmin, tpe, Trials
import numpy as np
from hyperopt import STATUS_OK, STATUS_FAIL
import pandas as pd


def working_well_str():
    return "      'INJ'  'GAS'  'OPEN'  'RATE'  4700  1*  4000\n"


def sleeping_well_str():
    return "      'INJ'  'GAS'  'OPEN'  'RATE'  0  1*  4000\n"


def form_datafile(filename, regime):
    well1 = regime[0]
    well2 = regime[1]

    assert (well1[1] > well1[0] or well2[1] > well2[0])

    well_1_on = False
    well_2_on = False

    iter = 0
    with open(filename, 'r') as f, open("./" + filename.split(".")[1] + "_RES." + filename.split(".")[2], "w") as f2:
        for i, line in enumerate(f, 1):
            if "WCONINJE" in line:
                f2.write("WCONINJE\n"
                         "-- Item #: 1	  2	 3	 4      5     6   7\n")
                if iter == well1[0]:
                    f2.write(working_well_str())
                    well_1_on = True
                elif iter == well1[1]:
                    f2.write(sleeping_well_str())
                    well_1_on = False
                else:
                    if well_1_on:
                        f2.write(working_well_str())
                    else:
                        f2.write(sleeping_well_str())

                if iter == well2[0]:
                    f2.write(working_well_str().replace("\n", " /\n"))
                    well_2_on = True
                elif iter == well2[1]:
                    f2.write(sleeping_well_str().replace("\n", " /\n"))
                    well_2_on = False
                else:
                    if well_2_on:
                        f2.write(working_well_str().replace("\n", " /\n"))
                    else:
                        f2.write(sleeping_well_str().replace("\n", " /\n"))

                iter += 1

            else:
                if ("-- Item #: 1	  2	 3	 4      5     6   7" not in line) and "'INJ'\t 'GAS'  'OPEN'  'RATE'" not in line:
                    f2.write(line)


def calc_loss(data):
    return 10**6/data


def calculate(start):
    w1 = [start[0], start[1]]
    w2 = [start[2], start[3]]
    try:
        form_datafile("../Task1/src/SPE3CASE1.DATA", [w1, w2])
    except AssertionError:
        return {'loss': 10000, 'params': [w1, w2], 'status': STATUS_FAIL}
    os.system("flow ./src/SPE3CASE1_RES.DATA")

    data = func("./src/SPE3CASE1_RES.PRT")
    loss = calc_loss(data)

    return {'loss': loss, 'params': [w1, w2], 'status': STATUS_FAIL}


if __name__ == "__main__":
    form_datafile("../Task1/src/SPE3CASE1.DATA", [[6, 8], [0, 13]])
