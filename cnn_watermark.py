#!/usr/bin/env python3

import os
import itertools

import numpy as np
from scipy.misc import imsave

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.datasets import cifar10
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
    x_data = x_data[:2000, ...] / 255.0
    y_data = np.random.randint(2, size=(x_data.shape[0], 1))
    y_data = keras.utils.to_categorical(y_data, 2)

    if not os.path.isdir("debug"):
        os.mkdir("debug")
    for i in range(10):
        imsave("debug/in-%d.png" % i, x_data[i, ...]*255)

    with tf.Graph().as_default():
        # my simple network, really simple
        model = Sequential()
        model.add(Conv2D(20, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=x_data.shape[1:]))
        model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='adam',
                      metrics=['accuracy'])
        model.save('model.h5')

        for i in itertools.count():
            del model
            print("ROUND:", i)
            sess = tf.Session()
            K.set_session(sess)
            sess.run(tf.global_variables_initializer())
            model = load_model("model.h5")

            # Stage1: watermark embedding
            x = tf.placeholder(tf.float32, shape=(None,)+x_data.shape[1:])
            target = tf.placeholder(tf.float32, shape=(None,)+y_data.shape[1:])
            next_x = x
            losses = []
            for j in range(20):
                next_x, loss = sgd(next_x, x, y_data, model(next_x),
                                   alpha=0.005, lmb=0.001)
                losses.append(loss)
            feed_dict = {x: x_data, target: y_data}
            x_embed, *ret_losses = sess.run([next_x] + losses,
                                            feed_dict=feed_dict)
            for j, item in enumerate(ret_losses):
                print("Embed #%d, loss=%f" % (j, item[0]))
            for j in range(10):
                imsave("debug/out-%d.png" % j, x_embed[j, ...]*255)

            # Stage2: attack simulation (only Gaussian noise)
            x_embed += np.random.normal(scale=0.2, size=x_embed.shape)

            # Stage3: network update
            model.fit(x_embed, y_data,
                      batch_size=64, epochs=16,
                      validation_split=0.2)
            model.save('model.h5')

            sess.close()
            del x_embed
            del ret_losses
            del sess


if __name__ == "__main__":
    main()
