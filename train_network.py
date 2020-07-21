import h5py
import numpy as np
import cupy as cp
from conv_layer import ConvLayer
from pool_layer import PoolLayer
import network as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_data(hfile):
    with h5py.File(hfile, 'r') as f:
        g = f["Data"]
        Y = g["Y_data"]
        X = g["X_data"]
        X_data = np.array(X)/255
        Y_data = np.array(Y)
        f.close()
    return X_data, Y_data



X_data, Y_data = load_data("data/copper_data.h5")
X_test, Y_test = load_data("data/copper_test_data.h5")
X_test = X_test[:6]
Y_test = Y_test[:6]

mb_size = 5
in_shape = (mb_size, *X_data.shape[1:])
lamb = 0.001

l1 = ConvLayer(in_shape, 2, 10, 1, 1, activation='relu')
l2 = PoolLayer(l1.dim_out, 2, 2, mode='max')
l3 = ConvLayer(l2.dim_out, 4, 20, 1, 2, activation='relu')
l4 = PoolLayer(l3.dim_out, 4, 4, mode='max')
l5 = ConvLayer(l4.dim_out, 3, 20, 1, 0, activation='relu')
l6 = PoolLayer(l5.dim_out, 4, 4, mode='max')
l7 = ConvLayer(l6.dim_out, 1, 5, 1, 0, activation='relu')

net = nn.Network((l1, l2, l3, l4, l5, l6, l7), lamb=lamb)
net.load_network("model")

# cost = net.train_network(X_data, Y_data, (X_test, Y_test),
#                          200, 0.002, mb_size)


res = net.forward_prop(cp.array(X_test))

def test_pict(X, res, pict):
    img = X[pict-1]

    x_stride = 34
    y_stride = 34

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for i in range(0, res.shape[1]):
        for j in range(0, res.shape[2]):
            if res[pict-1, i, j][0] > 0.5:
                bx, by, w, h = res[pict-1, i, j][1:]
                x = (j + bx - w/2) * x_stride
                y = (i + by - h/2) * y_stride
                ax.add_patch(Rectangle((x, y), w*x_stride, h*y_stride, color='red', fill=False))


    plt.show()
