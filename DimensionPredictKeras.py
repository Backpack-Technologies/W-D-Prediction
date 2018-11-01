import tensorflow as tf
from tensorflow import keras
import DataParser
import ProductInfo
import numpy as np

EPCHOES = 16
NUM_OF_COLS = 100
MODEL_FILE_NAME = '/Users/dipta007/my-world/backpack/work/DimensionPredict/model/test4/'
LOAD_DATA_FROM_FILE = False


def neural_model():
    model = keras.Sequential()

    # Input Layer
    # model.add(keras.layers.Dense(128, kernel_initializer='orthogonal', input_dim=NUM_OF_COLS, activation="relu",
    #                              kernel_regularizer=keras.regularizers.l1(0.01),
    #                              bias_regularizer=keras.regularizers.l1(0.01)))
    model.add(keras.layers.Dropout(0.2, input_shape=(NUM_OF_COLS, )))

    # Hidden Layer
    model.add(keras.layers.Dense(2560, kernel_initializer='orthogonal', activation='relu',
                                 kernel_regularizer=keras.regularizers.l1(0.01),
                                 bias_regularizer=keras.regularizers.l1(0.01)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(2560, kernel_initializer='orthogonal', activation='relu',
                                 kernel_regularizer=keras.regularizers.l1(0.01),
                                 bias_regularizer=keras.regularizers.l1(0.01)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(2560, kernel_initializer='orthogonal', activation='relu',
                                 kernel_regularizer=keras.regularizers.l1(0.01),
                                 bias_regularizer=keras.regularizers.l1(0.01)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(2560, kernel_initializer='orthogonal', activation='relu',
                                 kernel_regularizer=keras.regularizers.l1(0.01),
                                 bias_regularizer=keras.regularizers.l1(0.01)))
    model.add(keras.layers.Dropout(0.5))

    # Output Layer
    model.add(keras.layers.Dense(4, kernel_initializer='orthogonal', activation='linear',
                                 kernel_regularizer=keras.regularizers.l1(0.01),
                                 bias_regularizer=keras.regularizers.l1(0.01)))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    return model


def _start_shell(local_ns=None):
    # An interactive shell is useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


def predict(ind):
    with open('data/resD.vec', "r") as infile:
        line = ""
        while ind:
            line = infile.readline()
            ind -= 1
        row = line.split()
        print(ProductInfo.get("dimensions", row[0]))
        del row[0]
        npRow = np.asarray(np.float_(row))
        return npRow.reshape(1, 100)


def main(_):
    x_train, y_train = DataParser.get_data_for_model(LOAD_DATA_FROM_FILE)

    print(x_train.shape, y_train.shape)

    checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'

    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True,
                                                 mode='auto')
    callbacks_list = [checkpoint]

    model = neural_model()
    model.fit(x_train, y_train, epochs=EPCHOES, batch_size=32, validation_split=0.1)

    _start_shell(locals())


if __name__ == "__main__":
    tf.app.run()