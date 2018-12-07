#!/usr/bin/env python3

import sys

import numpy as np
from tensorflow.python.keras.models import load_model
from scipy.misc import imread


def main():
    model = load_model("model.h5")
    paths = sys.argv[1:]
    n_sample = len(paths)
    x_data = np.zeros((n_sample, 32, 32, 3), dtype=np.float32)
    for idx, item in enumerate(paths):
        x_data[idx, ...] = imread(item, mode="RGB") / 255.0
    y_pred = model.predict_classes(x_data)
    for p, y in zip(paths, y_pred):
        print(p, y)


if __name__ == "__main__":
    main()
