from PIL import ImageTk, Image
import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from conv_layer import ConvLayer
from pool_layer import PoolLayer


class Network:
    def __init__(self, layers, lamb, lcoord=1, lnoobj=0.2):
        self.layers = layers
        self.lamb = lamb
        self.lamb_coord = lcoord
        self.lamb_noobj = lnoobj
        self._set_lamb()

    def _set_lamb(self):
        """Set the L2 regularisation parameter in the layers."""
        for l in self.layers:
            l.lamb = self.lamb

    def save_network(self, path):
        """Save weights and biases of network to files in path."""
        for i, l in enumerate(self.layers):
            l.save_layers(path, i)
        print("Saved network in folder {:}".format(path))

    def load_network(self, path):
        """"""
        for i, l in enumerate(self.layers):
            l.load_layers(path, i)
        print("Loaded network from folder {:}".format(path))

    def get_minibatch(self, X, Y, size, i, permutation):
        """Get minibatch loaded in GPU-RAM.
        """
        m = X.shape[0]
        N_full_batches = int(m/size)

        if (i+1)*size >= m:
            end = -1
        else:
            end = (i+1)*size
        
        return cp.array(X[permutation[i*size:end]]), cp.array(Y[permutation[i*size:end]])


    def compute_cost(self, A, Y, lamb_coord, lamb_noobj):
        """Cost function for the YOLO algorithm."""
        m = Y.shape[0]
        L2 = 0
        for l in self.layers:
            L2 += self.lamb/m/2*cp.sum(cp.square(l.W))
        cost = 1/m * lamb_coord * Y[:, :, :, 0] * ((A[:,:,:,1]-Y[:, :, :, 1])**2
                                                   + (A[:, :, :, 2]-Y[:,:, :, 2])**2
                                                   + (cp.sqrt(A[:,:,:,3])-cp.sqrt(Y[:,:,:, 3]))**2
                                                   + (cp.sqrt(A[:,:,:,4])-cp.sqrt(Y[:,:,:, 4]))**2)
        cost += 1/m*(Y[:, :, :, 0]*(A[:,:,:,0]-Y[:, :, :, 0])**2 +
                     lamb_noobj*(1-Y[:,:,:,0])*(A[:,:,:,0]-Y[:, :, :, 0])**2)
        cost = cp.sum(cost) + L2
        return cost

    def compute_dA(self, A, Y, lamb_coord, lamb_noobj):
        """Compute dA for the softmax layer.
        Args:
            A (np.array): predicted labels
            Y (np.array): true labels
        Returns:
            np.array: dA
        """
        dA = np.zeros_like(Y)
        dA[:,:,:,0] = 2*(A[:,:,:,0]-Y[:,:,:,0])*(Y[:,:,:,0] + (1-Y[:,:,:,0])*lamb_noobj)
        dA[:,:,:,1] = lamb_coord*2*Y[:,:,:,0]*(A[:,:,:,1]-Y[:, :, :, 1])
        dA[:,:,:,2] = lamb_coord*2*Y[:,:,:,0]*(A[:,:,:,2]-Y[:, :, :, 2])
        dA[:,:,:,3] = lamb_coord*Y[:,:,:,0]*cp.nan_to_num((cp.sqrt(A[:,:,:,3])-cp.sqrt(Y[:, :, :, 3]))
                                                          /cp.sqrt(A[:,:,:,3]))
        dA[:,:,:,4] = lamb_coord*Y[:,:,:,0]*cp.nan_to_num((cp.sqrt(A[:,:,:,4])-cp.sqrt(Y[:, :, :, 4]))
                                                          /cp.sqrt(A[:,:,:,4]))
        return dA

    def forward_prop(self, X):
        """Forward propagation"""
        for l in self.layers:
            X = l.forward(X)
        A = X
        return A

    def backward_prop(self, A, Y, rate, t):
        """Backward propagation and update of parameters"""
        dA = self.compute_dA(A, Y, self.lamb_coord, self.lamb_noobj)
        for l in reversed(self.layers):
            dA = l.backward(dA)
            l.update_parameters(rate, t)

    def predict(self, X, show=True):
        """Predict single image.
        Args:
            X (np.array): (nx, ny, c)"""
        X_F = cp.array([X])
        for l in self.layers:
            X_F = l.forward(X_F)
        A = X_F
        res = cp.where(A == cp.max(A))
        print("Its a {:}".format(int(res[0][0])))
        if show:
            plt.imshow(X[:, :, 0])
            plt.title("Is a {:} with prob {:.2f}%".format(
                int(res[0][0]), float(A[res][0]*100)))
            plt.show()

    def test_accuracy(self, X, Y):
        """Test the accuracy of the network on test sample.
        TODO: propper stuff
        Args:
            X (np.array): (m, nx, ny, c) input sample
            Y (np.array): (1, 10, m) true labels
        Returns:
            None.
        """
        for l in self.layers:
            X = l.forward(X)
        A = X
        pred = (A[:,:,:,0] >= 0.5).astype(cp.float32)
        acc = (pred == Y[:,:,:,0]).astype(cp.float32)
        acc = cp.sum(acc)/(acc.shape[0]*acc.shape[1]*acc.shape[2])
        print("Accuracy: {:.2f}%".format(float(acc*100)))
        return acc

    def train_network(self, X, Y, test_set, epochs, rate, mb_size):
        """"""
        X_test, Y_test = cp.array(test_set[0]), cp.array(test_set[1])
        cost = []
        acc = []
        t = 1
        for i in range(epochs):
            cost_sum = 0
            m = X.shape[0]
            permutation = list(np.random.permutation(m))
            for j in range(0, int(m/mb_size)):
                mX, mY = self.get_minibatch(X, Y, mb_size, j, permutation)
                A = self.forward_prop(mX)
                c = self.compute_cost(A, mY, self.lamb_coord, self.lamb_noobj)
                cost_sum += c
                self.backward_prop(A, mY, rate, t)
                t += 1

            cost.append(cost_sum/int(m/mb_size))
            if i % 10 == 0:
                print("Cost of epoch {:}: {:.5f}".format(i, float(cost[-1])))
                acc.append(self.test_accuracy(X_test, Y_test))
                print()
        return cost, acc

