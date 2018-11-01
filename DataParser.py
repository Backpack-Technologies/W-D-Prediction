import ProductInfo
import numpy as np
import json
from sklearn.model_selection import train_test_split


DATA_FILE = "data/resD.vec"
DIMENSION_FILE = "data/djdataF.txt"
PARSED_DATA = 'data/dataF'

dimensions = dict()


def get_data(cat1, cat2, cat3, cat4):
    with open(DIMENSION_FILE) as infile:
        for line in infile:
            now = json.loads(line)
            dimension = now['dimensions']
            if cat1 not in dimension or cat2 not in dimension or cat3 not in dimension or cat4 not in dimension:
                continue
            dimensions[now['asin']] = now['dimensions']

    with open(DATA_FILE, "r") as infile:
        lineNo = 0
        datas = []
        for line in infile:
            lineNo += 1
            if lineNo > 2:
                row = line.split()

                asin = row[0]
                del row[0]

                if asin not in dimensions:
                    continue

                row.append(dimensions[asin][cat1])
                row.append(dimensions[asin][cat2])
                row.append(dimensions[asin][cat3])
                row.append(dimensions[asin][cat4])

                datas.append(row)

            if lineNo % 1000 == 0:
                print(lineNo)
        return datas


def get_splitted_data_for_model(load_data):
    if load_data:
        datas = np.asarray(np.float_(parse_data()))
        np.save(PARSED_DATA, datas)

    datas = np.load(PARSED_DATA + ".npy")
    X = datas[:, 0:100]
    y = datas[:, 100:104]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def get_data_for_model(load_data):
    if load_data:
        datas = np.asarray(np.float_(parse_data()))
        np.save(PARSED_DATA, datas)

    datas = np.load(PARSED_DATA + ".npy")
    X = datas[:, 0:100]
    y = datas[:, 100:104]
    return X, y


def parse_data():
    datas = get_data("length", "width", "height", "weight")
    return datas


if __name__ == "__main__":
    # parse_data()
    tmp = get_data("length", "width", "height", "weight")
    print(len(tmp))
