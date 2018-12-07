#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
from scipy.misc import imsave
from tensorflow.python.keras.models import load_model


def sgd(x, x0, target, pred, alpha, lmb):
    # compute loss
    loss1 = K.categorical_crossentropy(target, pred)
    loss2 = K.mean(K.square(x - x0), axis=[1, 2, 3])
    loss = loss1 + 0.5 * lmb * loss2
    deriv, = K.gradients(loss, x)
    next_x = K.clip(x - alpha*K.sign(deriv), 0, 1)
    return next_x, loss


# REF: https://github.com/faysalhossain2007/k-variant-ensemble/tree/master
def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_data = np.concatenate([x_train, x_test], axis=0)
    x_data = x_data[-1000::2, ...] / 255.0
    y_data = np.random.randint(2, size=(x_data.shape[0], 1))
    y_data = keras.utils.to_categorical(y_data, 2)

    with tf.Graph().as_default():
        sess = tf.Session()
        K.set_session(sess)
        sess.run(tf.global_variables_initializer())
        model = load_model("model.h5")
        x = tf.placeholder(tf.float32, shape=(None,)+x_data.shape[1:])
        target = tf.placeholder(tf.float32, shape=(None,)+y_data.shape[1:])
        next_x = x
        losses = []
        for j in range(20):
            next_x, loss = sgd(next_x, x, y_data, model(next_x), 0.005, 0.001)
            losses.append(loss)
        x_embed, *ret_losses = sess.run([next_x] + losses,
                                        feed_dict={x: x_data, target: y_data})
        print(" MAX:", np.max(x_embed - x_data))
        print("PSNR:", -10 * np.log10(np.mean(np.square(x_embed - x_data))))
        for i in range(x_data.shape[0]):
            imsave("input/wt_%d.png" % i, x_data[i, ...]*255)
            imsave("output/wt_%d_%d.png" % (i, np.argmax(y_data[i, :])),
                   x_embed[i, ...]*255)
        ret = model.evaluate(x_embed, y_data)
        print(ret)
        sess.close()


if __name__ == "__main__":
    main()
