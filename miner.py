from PIL import ImageTk, Image
import numpy as np
import tkinter as tk
import windowTools as wt
import _thread
import time
import h5py

import tensorflow as tf
import keras
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects

# import network as nn
# from conv_layer import ConvLayer
# from pool_layer import PoolLayer


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
    

    cost = cost_coord  + cost_class +  cost_conf
    return cost

@tf.function()
def yolo_activation(X):
    Z1 = K.maximum(0.0, X[..., 0:5])
    Z2 = K.exp(X[..., 5:])/K.sum(K.exp(X[..., 5:]), axis=-1, keepdims=True)
    return K.concatenate((Z1, Z2), axis=-1)

@tf.function()
def yolo_accuracy_class(y, ypred):
    acc = K.mean(K.equal(y[..., 5:], K.round(ypred[..., 5:]))[y[...,0] == 1.0])
    #print("Accuracy: {:.2f}%".format(float(acc*100)))
    return acc


@tf.function()
def yolo_accuracy_object(y, ypred):
    acc = K.mean(K.equal(y[..., 0], K.round(ypred[..., 0])))
    return acc

custom_objs = {'yolo_activation': yolo_activation, "yolo_accuracy_object" : yolo_accuracy_object, "yolo_loss": yolo_loss, "yolo_accuracy_class": yolo_accuracy_class}

get_custom_objects().update(custom_objs)



class Miner:
    def __init__(self, master):
        self.title = wt.get_window_title()
        self.master = master
        master.title("RuneScape Bot")
        self.gamefield = (29, 0, 510, 340)  # (x_min, y_min, x_max, y_max)
        self.mapfield = (595, 9, 740-595, 160-9)
        self.win_info = wt.get_wininfo(self.title)
        self.raw_im = Image.open("data/train1.png")

        self.colors= ["red", "blue", "green", "grey", "orange", "white", "yellow", "brown",
                      "violet", "dark violet", "forrest green", "mint cream", "misty rose",
                      "cyan", "lavender blush", "salmon", "coral", "tomato", "ivory", "azure"]
        
        self.ar_im = np.array([])
        self.time = 0

        self.x_offset = 0  # image offset in canvas
        self.y_offset = 0
        self.player_pos = (285, 180)  # absolute coordinates (x, y)
        self.nearest_ore = self.player_pos
        # image and buttons
        self.canvas = tk.Canvas(master, width=800, height=600)
        self.canvas.pack()

        self.button_update = tk.Button(
            master, text='Update', command=self.update)
        self.button_update.pack()

        self.img = ImageTk.PhotoImage(self.raw_im)
        self.canvas.create_image(
            self.x_offset, self.y_offset, anchor=tk.NW, image=self.img)

        self.ar_gamefield = np.array(self.raw_im.crop(self.gamefield))
        self.mark_iron_ores(self.ar_gamefield)
        self.player = self.canvas.create_oval(
            self.player_pos[0]-5 +
            self.x_offset, self.player_pos[1]-5+self.y_offset,
            self.player_pos[0]+5 +
            self.x_offset, self.player_pos[1]+5+self.y_offset,
            fill='blue')

        in_shape = (1, 340, 510, 3)
        lamb = 0.001

        # l1 = ConvLayer(in_shape, 2, 10, 1, 1, activation='lrelu')
        # l2 = PoolLayer(l1.dim_out, 2, 2, mode='max')
        # l3 = ConvLayer(l2.dim_out, 4, 20, 1, 2, activation='lrelu')
        # l4 = PoolLayer(l3.dim_out, 4, 4, mode='max')
        # l5 = ConvLayer(l4.dim_out, 3, 20, 1, 0, activation='lrelu')
        # l6 = PoolLayer(l5.dim_out, 4, 4, mode='max')
        # l7 = ConvLayer(l6.dim_out, 1, 5, 1, 0, activation='lrelu')
        # self.net = nn.Network((l1, l2, l3, l4, l5, l6, l7), lamb=lamb)


        # l1 = ConvLayer(in_shape, 2, 10, 1, 1, activation='lrelu')
        # l2 = PoolLayer(l1.dim_out, 2, 2, mode='max')
        # l3 = ConvLayer(l2.dim_out, 4, 20, 1, 2, activation='lrelu')
        # l4 = PoolLayer(l3.dim_out, 2, 2, mode='max')
        # l5 = ConvLayer(l4.dim_out, 2, 20, 1, 0, activation='lrelu')
        # l6 = PoolLayer(l5.dim_out, 4, 4, mode='max')
        # l7 = ConvLayer(l6.dim_out, 2, 20, 1, 0, activation='lrelu')
        # l8 = PoolLayer(l7.dim_out, 2, 2, mode='max')

        # #fully connected
        # l9 = ConvLayer(l8.dim_out, 1, 5, 1, 0, activation='lrelu')

        # self.net = nn.Network((l1, l2, l3, l4, l5, l6, l7, l8, l9), lamb=lamb)
        # self.net.load_network("model_9L")
        self.net = tf.keras.models.load_model("miner10_full.h5", compile=False, custom_objects=custom_objs)
        



    def mark_iron_ores(self, ar_game_field):
        """Find iron ores in game_field with the miner Class and mark them"""
        # iron_ores = self.miner.detect_iron(ar_game_field[:, :, :3])
        # for (x, y) in iron_ores:
        #     x += self.x_offset + self.gamefield[0] - 15
        #     y += self.y_offset + self.gamefield[1] - 15
        #     self.canvas.create_rectangle(x, y, x+30, y+30, outline='red')

    def update(self):
        """Update"""
        # print("FPS:", 1/(time.time()-self.time))
        # self.time = time.time()
        self.raw_im = wt.get_full_screen(self.title)
        self.ar_gamefield = wt.crop_to_array(self.raw_im, self.gamefield)
        # self.street, self.fence = self.locator.get_minimap(
        #     wt.crop_to_array(self.raw_im, self.mapfield))
        im = Image.frombytes("RGB", self.raw_im.size,
                             self.raw_im.bgra, "raw", "BGRX")
        self.img = ImageTk.PhotoImage(image=im)


        X_data = np.array(self.ar_gamefield[np.newaxis,:]/255)
        # ores = self.net.forward_prop(X_data)
        obj = self.net.predict(X_data)
        # draw stuff
        self.canvas.delete("all")
        self.canvas.create_image(
            self.x_offset, self.y_offset, anchor=tk.NW, image=self.img)

        objects = self.non_max_suppression(obj, 0.25)
        for i in range(len(objects)):
            self.canvas.create_rectangle(objects[i, 1], objects[i, 2], objects[i, 3], objects[i, 4],
                                         fill="", outline=self.colors[int(objects[i,5])])

        
        self.canvas.update()

        # if self.init_mining:
        #     self.miner.mine_at_ore()

    def run(self):
        while running:
            self.update()

    def test_thread(self):
        print('Start a new thread to click!')
        _thread.start_new(self.miner.click, (self.player_pos, 1, 5,))

    def non_max_suppression(self, A, threshold=0.5):
        """Remove overlapping bounding boxes. Returns filterd boxes in screen coordinates and
        as an array with shape (n_boxes, (x1, y1, x2, y2)).
        Args:
            A (np.array): predicted labels and boxes.
            threshold (float): overlap threshold to treat as new box
        Returns:
            np.array: only max boxes.
        """
        x_stride = 34
        y_stride = 34
        score = []
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        obj_class = []
        for i in range(0, A.shape[1]):
            for j in range(0, A.shape[2]):
                if A[0, i, j][0] > threshold:
                    bx, by, w, h = A[0, i, j][1:5]
                    score.append(A[0, i, j, 0])
                    x1.append((j + bx - w/2) * x_stride + 29)
                    y1.append((i + by - h/2) * y_stride)
                    x2.append((j + bx + w/2) * x_stride + 29)
                    y2.append((i + by + h/2) * y_stride)
                    obj_class.append(np.argmax(A[0, i, j, 5:]))

        score = np.array(score)
        x1 = np.array(x1)
        x2 = np.array(x2)
        y1 = np.array(y1)
        y2 = np.array(y2)
        obj_class = np.array(obj_class)

        score_indexes = score.argsort().tolist()
        boxes_keep_index = []
        while len(score_indexes) > 0:
            index = score_indexes.pop()
            boxes_keep_index.append(index)
            if not len(score_indexes):
                break
            #iou
            xs1 = np.maximum(x1[index], x1[score_indexes])
            ys1 = np.maximum(y1[index], y1[score_indexes])
            xs2 = np.minimum(x2[index], x2[score_indexes])
            ys2 = np.minimum(y2[index], y2[score_indexes])
            intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
            unions = (x2[index]-x1[index])*(y2[index]-y1[index]) \
                + (x2[score_indexes]-x1[score_indexes])*(y2[score_indexes]-y1[score_indexes]) \
                - intersections
            ious = np.array(intersections / unions)
            filtered_indexes = set((ious > threshold).nonzero()[0])
            score_indexes = [v for (i, v) in enumerate(score_indexes)
                              if i not in filtered_indexes]

        nms_res = np.zeros((len(boxes_keep_index), 6))
        for i, j in enumerate(boxes_keep_index):
            nms_res[i, :] = np.array([score[j], x1[j], y1[j], x2[j], y2[j], obj_class[j]])
        return nms_res


running = True


def close(*ignore):
    global running
    running = False
    root.destroy()


root = tk.Tk()
window = Miner(root)
root.bind('<Escape>', close)
root.protocol("WM_DELETE_WINDOW", close)
root.after(500, window.run)
root.mainloop()
