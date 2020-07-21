import cupy as np
# import numpy as np


class FCLayer:
    def __init__(self, dim_in, dim_out, activation='relu'):
        """Initialise layer of the neural network.
        Args:
            dim_in (int): dimension of the input nodes (n_x, m) or (m, n_h, n_w, c)
            dim_out (int): n_x dimension of the output nodes
            activation (str): activation function, 'relu' or 'sigmoid'
            lamb (float): L2 regularisation parameter 0 == no regularisation
        """
        if type(dim_in) != int:
            dim_in = dim_in[1]*dim_in[2]*dim_in[3]
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        self.lamb = 0
        self.W = self._init_weights(dim_in, dim_out)
        self.b = np.zeros((dim_out, 1))
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.vW = np.zeros(self.W.shape)
        self.vb = np.zeros(self.b.shape)
        self.sW = np.zeros(self.W.shape)
        self.sb = np.zeros(self.b.shape)

    def _init_weights(self, dim_in, dim_out):
        """Initialise parameters with He initialisation.
        Args:
            dim_in (int): dimension of intput layer
            dim_out (int): dimension of output layer
        Returns:
            np.array: initialised weights
        """
        W = np.random.randn(dim_out, dim_in) * np.sqrt(2/dim_in)
        return W

    def _relu(self, z):
        """ReLu activation function
        Args:
            z (np.array): input
        Returns:
            np.array.
        """
        return np.maximum(0, z)

    def _deriv_relu(self, z):
        """Derivative of ReLu function
        Args:
            z (np.array): input values
        Returns:
            np.array: derivative at z.
        """
        return (z > 0).astype(np.float32)

    def _sigmoid(self, z):
        """Sigmoid activation function
        Args:
            z (np.array): input
        Returns:
            np.array.
        """
        return 1/(1+np.exp(-z))

    def _deriv_sigmoid(self, z):
        """Derivative of the sigmoid function
        Args:
            z (np.array): input
        Returns:
            np.array.
        """
        return np.exp(-z)/(1+np.exp(-z))**2

    def forward(self, x):
        """Implementation of the forward propagation
        Args:
            x (np.array): array of input data
        Returns:
            np.array: array of the output layer
        """
        if len(x.shape) > 2:
            self.X = x.reshape(x.shape[0], -1).T
        else:
            self.X = x.copy()
        self.Z = np.dot(self.W, self.X) + self.b
        if self.activation == 'relu':
            A = self._relu(self.Z)
        elif self.activation == 'sigmoid':
            A = self._sigmoid(self.Z)
        return A

    def backward(self, dA):
        """Implementation of the backward propagation
        Args:
            dA (np.array): dimensions (dim_out)
        Returns:
            np.array: dX
        """
        m = dA.shape[1]
        if self.activation == 'relu':
            dZ = dA * self._deriv_relu(self.Z)
        elif self.activation == 'sigmoid':
            dZ = dA * self._deriv_sigmoid(self.Z)
        self.dW = 1/m * np.dot(dZ, self.X.T) + self.lamb / m*self.W
        # self.dW = 1/m * contract('ij,kj->ik', dZ, self.X) + self.lamb/m*self.W
        self.db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dX = np.dot(self.W.T, dZ)
        # dX = contract('ji,jk->ik', self.W, dZ)
        return dX

    def update_parameters(self, rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Update parameters"""
        self.vW = beta1*self.vW + (1-beta1)*self.dW
        self.vb = beta1*self.vb + (1-beta1)*self.db
        self.sW = beta2*self.sW + (1-beta2)*self.dW**2
        self.sb = beta2*self.sb + (1-beta2)*self.db**2
        vW = self.vW / (1-beta1**t)
        vb = self.vb / (1-beta1**t)
        sW = self.sW / (1-beta2**t)
        sb = self.sb / (1-beta2**t)
        self.W -= rate * vW/(np.sqrt(sW)+epsilon)
        self.b -= rate * vb/(np.sqrt(sb)+epsilon)
