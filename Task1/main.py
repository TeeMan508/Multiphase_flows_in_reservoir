import os
from hyperopt import hp, fmin, tpe, Trials
import numpy as np
from hyperopt import STATUS_OK, STATUS_FAIL
import pandas as pd

REQUIRED_DATA = [7300000, 15220000, 22465200, 28266890, 32960926, 36983876, 40550352, 43696008, 46_461_860]


def write_nth_line(filename, line_no, por):
    with open(filename, "r") as f, open("./" + filename.split(".")[1] + "_res." + filename.split(".")[2], "w") as f2:
        for i, line in enumerate(f, 1):
            if i == line_no:
                f2.write(f"    300*{por} /\n")
            else:
                f2.write(line)
        else:
            return 1


def calc_loss(data):
    return np.max(np.abs(np.array(data) - np.array(REQUIRED_DATA)) / np.array(REQUIRED_DATA))


def calculate_prod(por):
    write_nth_line("./SPE1CASE1.DATA", 93, por)
    try:
        os.system('flow SPE1CASE1_res.DATA')
        data = func("SPE1CASE1_res.PRT") #### здесь функция парсящая дату
        loss = calc_loss(data)
        return {'loss': loss, 'params': [por], 'status': STATUS_OK}
    except:
        return {'loss': 10000, 'params': [por], 'status': STATUS_FAIL}





if __name__ == "__main__":
    a = []
    for i in range(0, 16):
        for j in range(0, 16):
            for k in range(0, 16):
                for l in range(0, 16):
                    if j > i and l > k and (j - i) + (l - k) == 15:
                        a.append([[i, j], [k, l]])
    print(len(a))
    # n_iter = 20
    # options_dict = {
    #     0: list(np.arange(0, 1.0, 0.1)),
    # }
    # search_space = {
    #     0: hp.choice(label='por', options=options_dict[0]),
    # }
    # trials = Trials()
    # best = fmin(
    #     # функция для оптимизации
    #     fn=calculate_prod,
    #     # пространство поиска гиперпараметров
    #     space=search_space,
    #     # алгоритм поиска
    #     algo=tpe.suggest,
    #     # число итераций
    #     max_evals=n_iter,
    #     # куда сохранять историю поиска
    #     trials=trials,
    #     # random state
    #     rstate=np.random.default_rng(21),
    #     # progressbar
    #     show_progressbar=True)
    #
    # print(trials.results)
    #
    # data = pd.DataFrame(trials.results)
    # print(data)
    #
    # minimum = min(data.loss)
    # min_idx = data[abs(data['loss'] - minimum) < 1e-10].index
    # buf = data.loc[min_idx, 'params']
    # print(buf)