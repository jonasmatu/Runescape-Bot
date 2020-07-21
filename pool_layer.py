import cupy as np


class PoolLayer:
    def __init__(self, dim_in, f,  stride, mode='max'):
        """Initialise the pooling layer.
        Args:
            dim_in (tuple): (m, n_x, n_y, n_c)
            f (int): filter size
            stride (int): stride
            pad (int): padding
            mode (str): max or average pooling
        """
        self.dim_in = dim_in
        self.f = f
        self.stride = stride
        self.mode = mode
        self.W = 0
        self.b = 0
        self.dim_out = (dim_in[0],
                        1 + int((dim_in[1] - f)/stride),
                        1 + int((dim_in[2] - f)/stride),
                        dim_in[-1])
        self.X = np.zeros(dim_in)
        self.dX = np.zeros(dim_in)
        self.Z = np.zeros(self.dim_out)
        self.lamb = 0

    def forward_naive(self, x):
        """Forward implementation of the pooling
        Args:
            x (np.array): input values (m, n_x, n_y, n_c)
        Returns:
            np.array: output values
        """
        m, n_h, n_w, n_c = self.dim_out
        self.X = x.copy()
        for i in range(m):
            for h in range(n_h):
                v_s = h*self.stride
                v_e = v_s + self.f
                for w in range(n_w):
                    h_s = w*self.stride
                    h_e = h_s + self.f
                    for c in range(n_c):
                        if self.mode == "max":
                            self.Z[i, h, w, c] = np.max(
                                x[i, v_s:v_e, h_s:h_e, c])
                        elif self.mode == "average":
                            self.Z[i, h, w, c] = np.mean(
                                x[i, v_s:v_e, h_s:h_e, c])
        return self.Z

    def forward(self, x):
        """Foward implementation of pooling using stride tricks
        Args:
            x (np.array): input values (m, n_x, n_y, n_c)
        Returns:
            np.array: output_values
        """
        if self.dim_in != x.shape:
            self.dim_in = x.shape
            self.dX = np.zeros(self.dim_in)
        self.X = x
        n_h = self.dim_out[1]
        n_w = self.dim_out[2]
        shape = (self.X.shape[0],  # m
                 n_h,
                 n_w,
                 self.f,
                 self.f,
                 self.X.shape[-1])  # n_c
        strides = (self.X.strides[0],
                   self.X.strides[1]*self.stride,
                   self.X.strides[2]*self.stride,
                   self.X.strides[1],
                   self.X.strides[2],
                   self.X.strides[3])
        M = np.lib.stride_tricks.as_strided(
            self.X, shape=shape, strides=strides)
        Z = np.max(M, axis=(-3, -2))
        return Z

    def backward_naive(self, dA):
        """Implementation of backward pooling.
        Args:
            dA (np.array): derivative of output values
        Returns:
            np.array: derivative of intput values
        """
        if len(dA.shape) == 2:
            dA = dA.reshape(self.dim_out)
        self.dX[:, :, :, :] = 0
        m, n_h, n_w, n_c = self.dim_out
        for i in range(m):
            for h in range(n_h):
                v_s = h*self.stride
                v_e = h*self.stride+self.f
                for w in range(n_w):
                    h_s = w*self.stride
                    h_e = w*self.stride+self.f
                    for c in range(n_c):
                        if self.mode == "max":
                            mask = np.max(
                                self.X[i, v_s:v_e, h_s:h_e, c]) == self.X[i, v_s:v_e, h_s:h_e, c]
                            self.dX[i, v_s:v_e, h_s:h_e, c] += mask * \
                                dA[i, h, w, c]
                        elif self.mode == "average":
                            da = dA[i, h, w, c]
                            self.dX[i, v_s:v_e, h_s: h_e,
                                    c] += np.ones((self.f, self.f))*da/self.f**2
        return self.dX

    def backward(self, dA):
        """Implementation of backward pooling using stride tricks.
        Args:
            dA (np.array): derivative of output values
        Returns:
            np.array: derivative of intput values
        """
        if len(dA.shape) == 2:
            dA = dA.reshape(dA.shape[1], *self.dim_out[1:])
        self.dX[:, :, :, :] = 0
        n_h = self.dim_out[1]
        n_w = self.dim_out[2]
        shape = (self.X.shape[0],  # m
                 n_h,
                 n_w,
                 self.f,
                 self.f,
                 self.X.shape[-1])  # n_c
        strides = (self.X.strides[0],
                   self.X.strides[1]*self.stride,
                   self.X.strides[2]*self.stride,
                   self.X.strides[1],
                   self.X.strides[2],
                   self.X.strides[3])
        M = np.lib.stride_tricks.as_strided(
            self.X, shape=shape, strides=strides)  # , writeable=False)
        # dangerous: writing into memory, don't mess up strides !
        M_dX = np.lib.stride_tricks.as_strided(
            self.dX, shape=shape, strides=strides)  # , writeable=True)
        mask = np.max(M, axis=(-3, -2), keepdims=True) == M
        M_dX += np.multiply(mask, dA[:, :, :, None, None])
        return self.dX

    def update_parameters(self, rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        pass

    def save_layers(self, path, i):
        """Save weights and biases to file. """
        np.save("{:}/w_layer{:}.npy".format(path, i), self.W)
        np.save("{:}/b_layer{:}.npy".format(path, i), self.b)

    def load_layers(self, path, i):
        """Load weights and biases from file."""
        self.W = np.load("{:}/w_layer{:}.npy".format(path, i))
        self.b = np.load("{:}/b_layer{:}.npy".format(path, i))
