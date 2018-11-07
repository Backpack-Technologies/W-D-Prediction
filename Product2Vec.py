import gzip
import gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TOP = 1000000
INTERACTIVE_SHELL = False
NUMBER_OF_DATAS = str(TOP)
DATA_FILE = "data/p2v-data" + NUMBER_OF_DATAS
QUESTION_FILE = "data/p2v-question" + NUMBER_OF_DATAS
EMBEDDINGS_FILE = "data/p2v-embeddings" + NUMBER_OF_DATAS

EMBEDDING_SIZE = 300
EPOCH = 16


def init_file_name():
    global NUMBER_OF_DATAS
    global DATA_FILE
    global QUESTION_FILE
    global EMBEDDINGS_FILE

    NUMBER_OF_DATAS = str(TOP)
    DATA_FILE = "data/p2v-data" + NUMBER_OF_DATAS
    QUESTION_FILE = "data/p2v-question" + NUMBER_OF_DATAS
    EMBEDDINGS_FILE = "data/p2v-embeddings" + NUMBER_OF_DATAS


def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))

    with open(input_file, 'r') as f:
        for i, line in enumerate(f):

            if i % 10000 == 0:
                logging.info("read {0} Products".format(i))
            # do some pre-processing and return a list of words for each review text
            yield line.split()


def _start_shell(local_ns=None):
    # An interactive shell is useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


def main():
    init_file_name()

    documents = list(read_input(DATA_FILE))
    logging.info("Done reading data file")

    model = gensim.models.Word2Vec(documents, size=EMBEDDING_SIZE, window=20, min_count=0, workers=20, sg=1, hs=1,
                                   alpha=0.016, min_alpha=0.00001)
    model.train(documents, total_examples=len(documents), epochs=EPOCH)

    model.wv.save_word2vec_format(EMBEDDINGS_FILE, binary=False)
    model.accuracy(QUESTION_FILE)

    if INTERACTIVE_SHELL:
        _start_shell(locals())


if __name__ == "__main__":
    INTERACTIVE_SHELL = True
    main()
