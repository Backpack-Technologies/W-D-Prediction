import tensorflow as tf
from tensorflow import keras
import DataParser
import ProductInfo
import numpy as np

INTERACTIVE_SHELL = False
TOP = 1000000
EPCHOES = 4
NUM_OF_COLS = 300
MODEL_FILE_NAME = './model/Weights-best' + str(TOP) + ".h5"
EMBEDDINGS_FILE = 'data/p2v-embeddings' + str(TOP)
LOAD_DATA_FROM_FILE = True


def init_file_name():
    global EMBEDDINGS_FILE
    global MODEL_FILE_NAME

    MODEL_FILE_NAME = './model/Weights-best' + str(TOP) + ".h5"
    EMBEDDINGS_FILE = 'data/p2v-embeddings' + str(TOP)


def create_neural_model():
    model = keras.Sequential()

    activation_function = keras.layers.PReLU();

    # Input Layer
    model.add(keras.layers.Dense(405, kernel_initializer='normal', input_dim=NUM_OF_COLS,
                                 kernel_regularizer=keras.regularizers.l2(0.01),
                                 bias_regularizer=keras.regularizers.l2(0.01), name="input"))
    model.add(activation_function)

    # Hidden Layer
    model.add(keras.layers.Dense(405, kernel_initializer='normal', name="hidden1"))
    model.add(activation_function)
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(405, kernel_initializer='normal', name="hidden2"))
    model.add(activation_function)
    model.add(keras.layers.Dropout(0.2))

    # Output Layer
    model.add(keras.layers.Dense(4, kernel_initializer='normal', name="output"))

    model.compile(loss='mean_absolute_error', optimizer="Adam", metrics=['mean_absolute_error'])
    return model


def train_neural_model(model):
    DataParser.TOP = TOP
    x_train, x_test, y_train, y_test = DataParser.get_splitted_data_for_model(LOAD_DATA_FROM_FILE)

    print(x_train.shape, y_train.shape)

    # checkpoint_name = './modelDP/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    # checkpoint_name = './modelDP/Weights-best1.hdf5'

    checkpoint = keras.callbacks.ModelCheckpoint(MODEL_FILE_NAME, monitor='val_loss', verbose=1, save_best_only=True,
                                                 mode='auto')

    earlyStopping = keras.callbacks.EarlyStopping(patience=16, mode="auto")
    tensorBoard = keras.callbacks.TensorBoard(log_dir="./logs")

    callbacks_list = [checkpoint, earlyStopping]

    # model = create_neural_model()
    model.fit(x_train, y_train, epochs=EPCHOES, batch_size=444, validation_split=0.2, callbacks=callbacks_list)
    print("Current one: ", model.evaluate(x_test, y_test, batch_size=44))

    model.load_weights(MODEL_FILE_NAME)
    print("Best one: ", model.evaluate(x_test, y_test, batch_size=44))


def _start_shell(local_ns=None):
    # An interactive shell is useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


def predict(ind):
    with open(EMBEDDINGS_FILE, "r") as infile:
        line = ""
        while ind:
            line = infile.readline()
            ind -= 1
        row = line.split()
        print(ProductInfo.get("dimensions", row[0]))
        del row[0]
        npRow = np.asarray(np.float_(row))
        return npRow.reshape(1, NUM_OF_COLS)


def main(_):
    init_file_name()
    model = create_neural_model()
    train_neural_model(model)

    if INTERACTIVE_SHELL:
        _start_shell(locals())


if __name__ == "__main__":
    INTERACTIVE_SHELL = True
    tf.app.run()
