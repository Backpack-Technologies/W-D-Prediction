import DataFromDB
import Product2Vec as p2v
import DimensionPredictKeras

TOP = 1000000

P2V_DATA_AVAILABLE = False
SEARCH_QUERY_FILE_LOAD = False

P2V_EMBEDDINGS_AVAILABLE = False
P2V_EMBEDDING_SIZE = 300
P2V_EPOCH = 16

NN_EPOCH = 100


def main():
    if not P2V_DATA_AVAILABLE:
        DataFromDB.TOP = TOP
        DataFromDB.get_data(SEARCH_QUERY_FILE_LOAD)
    print("Phase 1 completed")

    if not P2V_EMBEDDINGS_AVAILABLE:
        p2v.TOP = TOP
        p2v.INTERACTIVE_SHELL = False
        p2v.EPOCH = P2V_EPOCH
        p2v.EMBEDDING_SIZE = P2V_EMBEDDING_SIZE
        p2v.main()
    print("Phase 2 completed")

    DimensionPredictKeras.TOP = TOP
    DimensionPredictKeras.EPCHOES = NN_EPOCH
    DimensionPredictKeras.NUM_OF_COLS = P2V_EMBEDDING_SIZE
    DimensionPredictKeras.INTERACTIVE_SHELL = True
    DimensionPredictKeras.main(TOP)


if __name__ == "__main__":
    main()

