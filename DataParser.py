import ProductInfo
import numpy as np
import json
from sklearn.model_selection import train_test_split


DATA_FILE = "data/p2v-embeddings1000000"
DIMENSION_FILE = "data/asin+dimensions.txt"
PARSED_DATA = 'data/dataset1000000'

dimensions = dict()
NUM_OF_COLS = 300


def get_data(cat1, cat2, cat3, cat4):
    loop = 0
    with open(DIMENSION_FILE) as infile:
        for line in infile:
            now = json.loads(line)
            dimension = now['dimensions']
            if cat1 not in dimension or cat2 not in dimension or cat3 not in dimension or cat4 not in dimension:
                continue
            dimensions[now['asin']] = now['dimensions']

            loop += 1
            if loop % 10000 == 0:
                print("created mapping", loop)

    with open(DATA_FILE, "r") as infile:
        lineNo = 0
        datas = []
        for line in infile:
            lineNo += 1

            if lineNo % 1000 == 0:
                print(lineNo)

            if lineNo > 1:
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

        return datas


def get_splitted_data_for_model(load_data):
    if load_data:
        datas = np.asarray(np.float_(parse_data()))
        np.save(PARSED_DATA, datas)

    datas = np.load(PARSED_DATA + ".npy")
    X = datas[:, 0:NUM_OF_COLS]
    y = datas[:, NUM_OF_COLS:NUM_OF_COLS+4]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
