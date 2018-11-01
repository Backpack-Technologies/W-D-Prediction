import tensorflow as tf
import DataParser
import numpy as np
import ProductInfo


EPCHOES = 4
NUM_OF_COLS = 100
MODEL_FILE_NAME = '/Users/dipta007/my-world/backpack/work/DimensionPredict/model/test3/'
LOAD_DATA_FROM_FILE = False


def neural_net_model(x_data, input_dim):
    w1 = tf.Variable(tf.random_uniform([input_dim, 256]))
    b1 = tf.Variable(tf.zeros([256]))
    layer1 = tf.add(tf.matmul(x_data, w1), b1)
    layer1 = tf.nn.relu(layer1)

    w2 = tf.Variable(tf.random_uniform([256, 256]))
    b2 = tf.Variable(tf.zeros([256]))
    layer2 = tf.add(tf.matmul(layer1, w2), b2)
    layer2 = tf.nn.relu(layer2)

    w3 = tf.Variable(tf.random_uniform([256, 1]))
    b3 = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer2, w3), b3)

    return output


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
    x_train, x_test, y_train, y_test = DataParser.get_data_for_model(LOAD_DATA_FROM_FILE)
    c_t = []
    c_test = []

    xs = tf.placeholder('float')
    ys = tf.placeholder('float')

    output = neural_net_model(xs, NUM_OF_COLS)

    cost = tf.reduce_mean(tf.square(output - ys))

    train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # saver.restore(sess, MODEL_FILE_NAME)

            for epoch in range(EPCHOES):
                for j in range(x_train.shape[0]):
                    sess.run([cost, train], feed_dict={xs: x_train[j, :].reshape(1, NUM_OF_COLS), ys: y_train[j]})

                c_t.append(sess.run(cost, feed_dict={xs: x_train, ys:y_train}))
                c_test.append(sess.run(cost, feed_dict={xs: x_test, ys: y_test}))

                print("Epoch ", epoch, " cost: ", c_test[epoch])

        # pred = sess.run(output, feed_dict={xs:x_test})
        saver.save(sess, MODEL_FILE_NAME + "test.ckpt")
        print("model saved")

        _start_shell(locals())


if __name__ == "__main__":
    tf.app.run()
    # get_data(True)
