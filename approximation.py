"""Portable MNIST loading and affine / PCA-quadratic logit approximation."""
from pathlib import Path
import gzip
import struct
import numpy as np

def load_data(root):
    root = Path(root)
    def images(name):
        with gzip.open(root / 'data/mnist_data' / name, 'rb') as f:
            magic, n, h, w = struct.unpack('>IIII', f.read(16))
            if magic != 2051: raise ValueError('Invalid IDX image header')
            return np.frombuffer(f.read(), np.uint8).reshape(n, h*w)
    def labels(name):
        with gzip.open(root / 'data/mnist_data' / name, 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            if magic != 2049: raise ValueError('Invalid IDX label header')
            return np.frombuffer(f.read(), np.uint8).reshape(n)
    X = images('train-images-idx3-ubyte.gz')
    Xt = images('t10k-images-idx3-ubyte.gz')
    y = labels('train-labels-idx1-ubyte.gz')
    yt = labels('t10k-labels-idx1-ubyte.gz')
    Y = np.load(root / 'data/Y_lenet_mnist.npy', allow_pickle=False)
    if Y.shape != (len(X), 10): raise ValueError('Unexpected cached logit shape')
    return X, Xt, y, yt, Y

def quadratic_features(z):
    z = np.asarray(z, dtype=np.float64)
    i, j = np.triu_indices(z.shape[1])
    return np.column_stack((z, z[:, i]*z[:, j], np.ones(len(z))))

def fit_quadratic(X, Y, k=20):
    X = np.asarray(X, dtype=np.float64) / 255.0
    if not 1 <= k <= min(X.shape[1], len(X)-1): raise ValueError('Invalid PCA rank')
    mu = X.mean(0)
    _, _, vh = np.linalg.svd(X-mu, full_matrices=False)
    V = vh[:k]
    A = quadratic_features((X-mu) @ V.T)
    W = np.linalg.lstsq(A, Y, rcond=None)[0]
    return mu, V, W

def predict_quadratic(X, model):
    mu, V, W = model
    z = (np.asarray(X, dtype=np.float64)/255.0-mu) @ V.T
    return quadratic_features(z) @ W

def fit_affine(X, Y):
    X = np.asarray(X, dtype=np.float64)/255.0
    return np.linalg.lstsq(np.column_stack((X, np.ones(len(X)))), Y, rcond=None)[0]

def predict_affine(X, W):
    X = np.asarray(X, dtype=np.float64)/255.0
    return np.column_stack((X, np.ones(len(X)))) @ W
