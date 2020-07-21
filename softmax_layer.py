import numpy as np


class SoftmaxLayer:

    def __init__(self, dim_in, dim_out):
        self.X = np.zeros(dim_in)

    def forward(self, X):
        """Forward propagation of the softmax layer.
        Args:
            X (np.array): input values
        Returns:
            np.array: softmax applied values
        """
        self.X = X
        A = np.exp(X)/np.sum(np.exp(X))
        return A

    def backward(self, dA):
        """Compute the derivative of the cost function with respect to
        the input
        Args:
            A (np.array): predicted labels
            Y (np.array): true labels
        Returns:
            np.array: derivative of L.
        """
        S = np.exp(self.X)/np.sum(np.exp(self.X))
        dX =
        return
