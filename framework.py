from PIL import ImageTk, Image
import numpy as np
import tkinter as tk
from miner import Miner
import windowTools as wt
from locator import Locator
import _thread
import time


class Window:
    def __init__(self, master):
        self.master = master
        master.title("RuneScape Bot")
        self.gamefield = (29, 0, 512, 334)  # (x_min, y_min, x_max, y_max)
        self.mapfield = (595, 9, 740-595, 160-9)
        self.win_info = wt.get_wininfo()
        self.raw_im = Image.open("Images/train.png")
        self.ar_im = np.array([])
        self.time = 0

        self.miner = Miner(self.win_info)

        self.x_offset = 20  # image offset in canvas
        self.y_offset = 20
        self.player_pos = (285, 180)  # absolute coordinates (x, y)

        # image and buttons
        self.canvas = tk.Canvas(master, width=900, height=800)
        self.canvas.pack()

        self.button_update = tk.Button(
            master, text='Update', command=self.update)
        self.button_update.pack()

        self.button_click = tk.Button(
            master, text='Click', command=self.test_thread)
        self.button_click.pack()

        self.button_savepic = tk.Button(
            master, text='Save', command=lambda: self.raw_im.save('Images/im.png'))
        self.button_savepic.pack()

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

        self.locator = Locator()

    def mark_iron_ores(self, ar_game_field):
        """Find iron ores in game_field with the miner Class and mark them"""
        iron_ores = self.miner.detect_iron(ar_game_field[:, :, :3])
        for (x, y) in iron_ores:
            x += self.x_offset + self.gamefield[0] - 15
            y += self.y_offset + self.gamefield[1] - 15
            self.canvas.create_rectangle(x, y, x+30, y+30, outline='red')

    def update(self):
        """Update"""
        print("FPS:", 1/(time.time()-self.time))
        self.time = time.time()
        self.raw_im = wt.get_full_screen()
        self.ar_gamefield = wt.crop_to_array(self.raw_im, self.gamefield)
        self.street, self.fence = self.locator.get_minimap(
            wt.crop_to_array(self.raw_im, self.mapfield))
        im = Image.frombytes("RGB", self.raw_im.size,
                             self.raw_im.bgra, "raw", "BGRX")
        self.img = ImageTk.PhotoImage(image=im)

        # draw stuff
        self.canvas.delete("all")
        self.canvas.create_image(
            self.x_offset, self.y_offset, anchor=tk.NW, image=self.img)

        self.mark_iron_ores(self.ar_gamefield)
        self.player = self.canvas.create_oval(
            self.player_pos[0]-5 +
            self.x_offset, self.player_pos[1]-5+self.y_offset,
            self.player_pos[0]+5+self.x_offset, self.player_pos[1]+5+self.y_offset, fill='blue')
        for (x, y) in self.street:
            self.canvas.create_rectangle(
                x+595 + 20, y+9 + 20, x+1+595+20, y+1+9+20, outline='green')
        for (x, y) in self.fence:
            self.canvas.create_rectangle(
                x+595 + 20, y+9 + 20, x+1+595+20, y+1+9+20, outline='pink')
        self.canvas.update()

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
window = Window(root)
root.bind('<Escape>', close)
root.protocol("WM_DELETE_WINDOW", close)
root.after(500, window.run)
root.mainloop()
