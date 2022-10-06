from keras.layers.attention.multi_head_attention import activation
from pyparsing import actions
import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py


def load_data(hfile):
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
    cost_coord = (lamb_coord * Y[:,:,:, 0] * (tf.square(A[:,:,:, 1] - Y[:,:,:, 1]) +
                                            tf.square(A[:,:,:, 2] - Y[:,:,:, 2]) +
                                            tf.square(tf.sqrt(tf.abs(A[:,:,:, 3]))-tf.sqrt(tf.abs(Y[:,:,:, 3])))
                                            + tf.square(tf.sqrt(tf.abs(A[:,:,:, 4]))-tf.sqrt(tf.abs(Y[:,:,:, 4])))))

    # confidence
    cost_conf= (Y[:,:,:, 0]*tf.square(A[:,:,:, 0]-Y[:,:,:, 0]) +
                lamb_noobj*(1-Y[:,:,:, 0])*tf.square(A[:,:,:, 0]-Y[:,:,:, 0]))

    # class loss:
    cost_class =  tf.square(Y[:,:,:, 5:] - A[:,:,:, 5:])

    # cost = tf.sum(cost) # + L2
    # return cost_coord + cost_conf + cost_class
    cost = cost_coord + cost_conf + cost_coord
    # cost = tf.clip_by_value(cost, -1e6, 1e6)
    cost = tf.math.reduce_sum(tf.math.reduce_sum(cost, axis=-1), axis=-1)
    tf.print(cost)
    cost = tf.math.reduce_sum(cost)
    return cost

X_data, Y_data = load_data("5objectdata.h5")

mb_size = 1
in_shape = (X_data.shape[1:])

eta = 0.001
beta1=0.9
beta2=0.999
lamb = 0.001

model = tf.keras.Sequential()

model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
model.add(tf.keras.layers.Conv2D(10, input_shape=in_shape, strides=1, kernel_size=(2,2),
                                 activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.ZeroPadding2D(padding=(2, 2)))
model.add(tf.keras.layers.Conv2D(20, strides=1, kernel_size=(4,4), activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(20,strides=1, kernel_size=(2,2), activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(4,4)))
model.add(tf.keras.layers.ZeroPadding2D(padding=(1,1)))
model.add(tf.keras.layers.Conv2D(20,strides=1, kernel_size=(2,2), activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(11, strides=1, kernel_size=(2,2), activation="relu"))


model.compile(loss = yolo_loss, 
   optimizer = keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy'])

model.fit(
   X_data, Y_data, 
   batch_size = mb_size, 
   epochs = 100, 
   verbose = 1
)
