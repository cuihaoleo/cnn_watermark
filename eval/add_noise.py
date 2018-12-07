#!/usr/bin/env python3

import sys
import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave


def main():
    outdir = "noise/"
    paths = sys.argv[1:]
    n_sample = len(paths)
    x_data = np.zeros((n_sample, 32, 32, 3), dtype=np.float32)
    for idx, item in enumerate(paths):
        x_data[idx, ...] = imread(item, mode="RGB") / 255.0
    x_data += np.random.normal(scale=0.2, size=x_data.shape)
    np.clip(x_data, 0, 1)
    for idx, item in enumerate(paths):
        outpath = os.path.join(outdir, os.path.basename(item))
        imsave(outpath, x_data[idx, ...]*255)


if __name__ == "__main__":
    main()
