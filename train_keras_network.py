from keras.layers.attention.multi_head_attention import activation
from keras import backend as K
from pyparsing import actions
import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py

from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D


def load_data(hfile):
    """The data has the following format:
    Y_data.shape = (n, 15, 10, 5 + k)
    With n training data and k objects"""
    with h5py.File(hfile, 'r') as f:
        g = f["Data"]
        Y = g["Y_data"]
        X = g["X_data"]
        X_data = np.array(X)/255
        Y_data = np.array(Y)
        f.close()
    return X_data, Y_data


@tf.function()
def yolo_loss(y, y_pred):
    """Cost function for the YOLO algorithm."""
    A = y_pred
    #tf.print(A.shape)
    #tf.print(A[np.where(A < 0)])
    Y = y
    lamb_coord = 1
    lamb_noobj = .2
    # Coordinate loss
    mse = tf.keras.losses.MeanSquaredError()
    
    cost_coord = (A[...,1] - Y[...,1])*(A[...,1] - Y[...,1])
    cost_coord += (A[...,2] - Y[...,2])*(A[...,2] - Y[...,2])
    cost_coord += (tf.sqrt(A[..., 3]) - tf.sqrt(Y[..., 3])) \
        * (tf.sqrt(A[..., 3]) - tf.sqrt(Y[..., 3]))
    cost_coord += (tf.sqrt(A[..., 4]) - tf.sqrt(Y[..., 4])) \
        * (tf.sqrt(A[..., 4]) - tf.sqrt(Y[..., 4]))

    cost_coord =  tf.math.reduce_sum(Y[..., 0] * cost_coord)
    
    # cost_coord = lamb_coord * mse(Y[...,0]*A[..., 1], Y[..., 0]*Y[..., 1])
    # cost_coord += lamb_coord * mse(Y[...,0]*A[..., 2], Y[..., 0]*Y[..., 2])
    # cost_coord += lamb_coord * mse(Y[...,0]*tf.sqrt(A[:,:,:, 3]),
    #                                Y[..., 0]*tf.sqrt(Y[:,:,:, 3]))
    # cost_coord += lamb_coord * mse(Y[...,0]*tf.sqrt(A[:,:,:, 4]),
    #                                Y[..., 0]*tf.sqrt(Y[:,:,:, 4]))
    
    # tf.print("\nCost coordinates:", cost_coord)
    # tf.print("Coords shape:", cost_coord.shape)
    # confidence
    # cost_conf = mse(Y[...,0]*A[...,0], Y[...,0]*Y[...,0])
    # cost_conf += lamb_noobj*mse(tf.square((1-A[:,:,:, 0]))*A[...,0],
    #                             tf.square(1-A[:,:,:, 0])* Y[...,0])

    cost_conf= tf.math.reduce_sum(Y[..., 0]*tf.square(A[..., 0]-Y[..., 0]) +
                lamb_noobj*tf.abs(1-A[..., 0])*tf.square(A[..., 0]-Y[..., 0]))
    # tf.print("Shape conf:", cost_conf.shape)
    # tf.print("Cost conf:", cost_conf)

    # class loss:
    # cost_class = K.sum(mse(Y[:,:,:,5:], A[:,:,:,5:]), axis=-1)
    cost_class = mse(Y[:,:,:,5:], A[:,:,:,5:])
    
    # tf.print("Shape class:", cost_class.shape)
    # tf.print(cost_class)
    # tf.print("Cost class:", K.sum(cost_class))

    cost = cost_conf + cost_class + cost_coord
    return cost


X_data, Y_data = load_data("testdata.h5")
# X_data, Y_data = load_data("5objectdata.h5")
# X_data = X_data[0].reshape(1, *X_data.shape[1:])
# Y_data = Y_data[0].reshape(1, *Y_data.shape[1:])

in_shape = (X_data.shape[1:])
out_len = Y_data.shape[-1]

l2 = tf.keras.regularizers.l2(0.0001)
lrelu = tf.keras.layers.LeakyReLU(alpha=0.05)

model = tf.keras.Sequential()

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(16, input_shape=in_shape, strides=1, kernel_size=(2,2),
                                 activation=lrelu))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(ZeroPadding2D(padding=(2, 2)))
model.add(Conv2D(32, strides=1, kernel_size=(4,4), activation=lrelu))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,strides=1, kernel_size=(2,2), activation=lrelu))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(32,strides=1, kernel_size=(2,2), activation=lrelu))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(out_len, strides=1, kernel_size=(2,2), activation="relu"))

# model = tf.keras.models.load_model("testmodel.h5", compile=False)

model.compile(loss = yolo_loss, 
   optimizer = keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy'])


mb_size = 4
model.fit(X_data, Y_data, batch_size = mb_size, epochs = 100, verbose = 1, shuffle=True)
