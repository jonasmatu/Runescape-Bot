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
    Y = y
    lamb_coord = 1
    lamb_noobj = .2
    # Coordinate loss
    mse = keras.losses.MeanSquaredError()
    
    cost_coord = (A[...,1] - Y[...,1])*(A[...,1] - Y[...,1])
    cost_coord += (A[...,2] - Y[...,2])*(A[...,2] - Y[...,2])
    cost_coord += (K.sqrt(A[..., 3]) - K.sqrt(Y[..., 3])) \
        * (tf.sqrt(A[..., 3]) - tf.sqrt(Y[..., 3]))
    cost_coord += (K.sqrt(A[..., 4]) - K.sqrt(Y[..., 4])) \
        * (K.sqrt(A[..., 4]) - K.sqrt(Y[..., 4]))

    cost_coord =  K.sum(Y[..., 0] * cost_coord)
    
    cost_conf= K.sum(Y[..., 0]*(A[..., 0]-Y[..., 0])*(A[..., 0]-Y[..., 0]) +
                     lamb_noobj*(1-Y[..., 0])*(A[..., 0]-Y[..., 0])*(A[..., 0]-Y[..., 0]))

    cost_class = K.sum((Y[...,5:]- A[...,5:])*(Y[...,5:]- A[...,5:]))
    

    cost = cost_conf + cost_coord  + cost_class 
    return cost

@tf.function()
def yolo_accuracy_class(y, ypred):
    
    acc = K.mean(K.equal(y[..., 5:], K.round(ypred[..., 5:])).where(y[...,0] == 1.0))
    #print("Accuracy: {:.2f}%".format(float(acc*100)))
    return acc


@tf.function()
def yolo_accuracy_object(y, ypred):
    acc = K.mean(K.equal(y[..., 0], K.round(ypred[..., 0])))
    return acc
    
X_data, Y_data = load_data("10objectdata.h5")
# X_data, Y_data = load_data("5objectdata.h5")
# X_data = X_data[0].reshape(1, *X_data.shape[1:])
# Y_data = Y_data[0].reshape(1, *Y_data.shape[1:])

in_shape = (X_data.shape[1:])
out_len = Y_data.shape[-1]

l2 = tf.keras.regularizers.l2(0.0001)
lrelu = tf.keras.layers.LeakyReLU(alpha=0.05)

model = tf.keras.Sequential()
model.add(ZeroPadding2D(padding=(4,4)))
model.add(Conv2D(8, input_shape=in_shape, strides=1, kernel_size=(5,5), activation=lrelu))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(16, strides=1, kernel_size=(2,2),
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

# model = tf.keras.models.load_model("testmodel10.h5", compile=False)

model.compile(loss = yolo_loss, 
   optimizer = keras.optimizers.Adam(learning_rate=0.001), metrics = [yolo_accuracy_object, yolo_accuracy_class])


mb_size = 4
model.fit(X_data, Y_data, batch_size = mb_size, epochs = 100, verbose = 1, shuffle=True)
