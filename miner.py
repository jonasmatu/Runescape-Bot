from PIL import ImageTk, Image
import numpy as np
import cupy as cp 
import tkinter as tk
import windowTools as wt
import _thread
import time
import h5py

import network as nn
from conv_layer import ConvLayer
from pool_layer import PoolLayer


class Miner:
    def __init__(self, master):
        self.master = master
        master.title("RuneScape Bot")
        self.gamefield = (29, 0, 510, 340)  # (x_min, y_min, x_max, y_max)
        self.mapfield = (595, 9, 740-595, 160-9)
        self.win_info = wt.get_wininfo()
        self.raw_im = Image.open("data/train1.png")
        self.ar_im = np.array([])
        self.time = 0

        self.x_offset = 0  # image offset in canvas
        self.y_offset = 0
        self.player_pos = (285, 180)  # absolute coordinates (x, y)
        self.nearest_ore = self.player_pos
        # image and buttons
        self.canvas = tk.Canvas(master, width=900, height=800)
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


        l1 = ConvLayer(in_shape, 2, 10, 1, 1, activation='lrelu')
        l2 = PoolLayer(l1.dim_out, 2, 2, mode='max')
        l3 = ConvLayer(l2.dim_out, 4, 20, 1, 2, activation='lrelu')
        l4 = PoolLayer(l3.dim_out, 2, 2, mode='max')
        l5 = ConvLayer(l4.dim_out, 2, 20, 1, 0, activation='lrelu')
        l6 = PoolLayer(l5.dim_out, 4, 4, mode='max')
        l7 = ConvLayer(l6.dim_out, 2, 20, 1, 0, activation='lrelu')
        l8 = PoolLayer(l7.dim_out, 2, 2, mode='max')

        #fully connected
        l9 = ConvLayer(l8.dim_out, 1, 5, 1, 0, activation='lrelu')

        self.net = nn.Network((l1, l2, l3, l4, l5, l6, l7, l8, l9), lamb=lamb)
        self.net.load_network("model_9L")



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
        self.raw_im = wt.get_full_screen()
        self.ar_gamefield = wt.crop_to_array(self.raw_im, self.gamefield)
        # self.street, self.fence = self.locator.get_minimap(
        #     wt.crop_to_array(self.raw_im, self.mapfield))
        im = Image.frombytes("RGB", self.raw_im.size,
                             self.raw_im.bgra, "raw", "BGRX")
        self.img = ImageTk.PhotoImage(image=im)


        X_data = cp.array(self.ar_gamefield[np.newaxis,:]/255)
        ores = self.net.forward_prop(X_data)
        
        # draw stuff
        self.canvas.delete("all")
        self.canvas.create_image(
            self.x_offset, self.y_offset, anchor=tk.NW, image=self.img)

        # x_stride = 34
        # y_stride = 34

        # for i in range(0, ores.shape[1]):
        #     for j in range(0, ores.shape[2]):
        #         if ores[0, i, j][0] > 0.5:
        #             bx, by, w, h = ores[0, i, j][1:]
        #             x1 = (j + bx - w/2) * x_stride + 29
        #             y1 = (i + by - h/2) * y_stride
        #             x2 = (j + bx + w/2) * x_stride + 29
        #             y2 = (i + by + h/2) * y_stride
        #             self.canvas.create_rectangle(x1, y1, x2, y2,
        #                                          fill="", outline="red")

        ores = self.net.non_max_suppression(ores, 0.25)
        for i in range(len(ores)):
            self.canvas.create_rectangle(ores[i, 1], ores[i, 2], ores[i, 3], ores[i, 4],
                                         fill="", outline="red")

        
        
        # self.player = self.canvas.create_oval(
        #     self.player_pos[0]-5 +
        #     self.x_offset, self.player_pos[1]-5+self.y_offset,
        #     self.player_pos[0]+5+self.x_offset, self.player_pos[1]+5+self.y_offset, fill='blue')
        # for (x, y) in self.street:
        #     self.canvas.create_rectangle(
        #         x+595 + 20, y+9 + 20, x+1+595+20, y+1+9+20, outline='green')
        # for (x, y) in self.fence:
        #     self.canvas.create_rectangle(
        #         x+595 + 20, y+9 + 20, x+1+595+20, y+1+9+20, outline='pink')
        self.canvas.update()

        # if self.init_mining:
        #     self.miner.mine_at_ore()

    def run(self):
        while running:
            self.update()

    def test_thread(self):
        print('Start a new thread to click!')
        _thread.start_new(self.miner.click, (self.player_pos, 1, 5,))


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
