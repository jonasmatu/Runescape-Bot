from PIL import ImageTk, Image
import tkinter as tk
import numpy as np
import h5py
import windowTools as wt
import argparse
import os.path
import time

class Classifier:
    def __init__(self, master, hfile, obj_file):
        self.master = master
        self.title = wt.get_window_title()
        master.title("Image Data Annotations")

        self.colors= ["red", "blue", "green", "grey", "orange", "white", "yellow", "brown",
                      "violet", "dark violet", "forrest green", "mint cream", "misty rose",
                      "cyan", "lavender blush", "salmon", "coral", "tomato", "ivory", "azure"]

        self.objects= []
        self.load_objects(obj_file)
        
        self.hfile = hfile
        self.run_recording = False
        self.run_classify_recording = False

        frame1, frame2 = self.create_frames(master)
        frame1.pack()
        frame2.pack()

        self.x = 0
        self.y = 0
        self.x_offset = 20  # image offset in canvas
        self.y_offset = 20
        self.grid_x = 15
        self.grid_y = 10
        self.data = np.zeros((self.grid_y, self.grid_x, 5+len(self.objects)))

        self.last_data = (0,0)

        self.next_image()
        self.rect = None

        self.start_x = None
        self.start_y = None

        self.rectangles = []
        self.index_recording = 0


    def load_objects(self,obj_file):
        with open(obj_file, "r") as f:
            for line in f.readlines():
                print(line.replace("\n", ""))
                self.objects.append(line.replace("\n", ""))

    def create_frames(self, master):
        frame1 = tk.Frame(master)
        self.canvas = tk.Canvas(frame1, width=600, height=400)
        self.canvas.pack()

        frame = tk.Frame(master)
        # frame.columnconfigure(0, weight=1)
        # frame.columnconfigure(1, weight=1)
        
        button_save = tk.Button(
            frame, text='Save', command=self.save_data).grid(column=1, row=0)
        #self.button_save.pack()

        button_next = tk.Button(frame, text="next", command=self.next_image).grid(column=0, row=0)
        #self.button_next.pack()

        button_delete = tk.Button(frame, text="delete last Image",
                                  command=self.delete_data).grid(column=0, row=1, sticky="w")
        button_delete_box = tk.Button(frame, text="delete last box",
                                      command=self.delete_box).grid(column=1, row=1, sticky="e")
        self.button_record = tk.Button(frame, text="start record",
                                       command=self.start_recording)
        self.button_record.grid(column=0, row=2, sticky="w")
        self.button_classify = tk.Button(frame, text="classify recording",
                                            command=self.classify_recording)
        self.button_classify.grid(column=1, row=2, sticky="e")

        #self.button_delete.pack()

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        self.lb = tk.Listbox(frame)
        sb = tk.Scrollbar(frame, orient=tk.VERTICAL)
        self.lb.configure(yscrollcommand=sb.set)
        sb.config(command=self.lb.yview)
        for obj in self.objects:
            self.lb.insert("end", obj)

        self.lb.grid(column=2, row=0, rowspan=3)
        self.lb.select_set(0)
        sb.grid(column=3, row=0, rowspan=3, sticky='ns')

        return frame1, frame

    def next_image(self):
        self.canvas.delete("all")
        self.data = np.zeros((self.grid_y, self.grid_x, 5+len(self.objects)))
        if self.run_classify_recording:
           self.classify_next()
        else:
            self.ar_im = wt.get_array((30, 2, 510, 340), self.title)
            print(self.ar_im.shape)
            self.img = ImageTk.PhotoImage(Image.fromarray(self.ar_im))
            self.im_w = self.ar_im.shape[1]
            self.im_h = self.ar_im.shape[0]
            
            self.canvas.create_image(
                self.x_offset, self.y_offset, anchor=tk.NW, image=self.img)


    def start_recording(self):
        if not self.run_recording:
            self.run_recording = True
            self.button_record['text'] = "Stop recording"
        else:
            self.button_record.configure(text = "Start recording")
            print("Stop recording")
            self.run_recording = False

        if self.run_recording:        
            print("Start recording")
            self.run_recording = True
            with h5py.File("data/recording.h5", 'w') as f:
                g = f.create_group("Data")
                X_data = g.create_dataset("X_data", shape=(1, 340, 510, 3), maxshape=(None, 340, 510, 3))
                self.ar_im = wt.get_array((30, 2, 510, 340),self.title)                
                X_data[0] = self.ar_im[:, :, :3]
                self.img = ImageTk.PhotoImage(Image.fromarray(self.ar_im))
                self.im_w = self.ar_im.shape[1]
                self.im_h = self.ar_im.shape[0]
                self.canvas.create_image(self.x_offset, self.y_offset, anchor=tk.NW, image=self.img)
                print("Start recording scree to file {:}".format(self.hfile))                
                self.master.after(250, self.recording)
        
        
    def recording(self):
        f = h5py.File("data/recording.h5", 'a')
        g = f['Data']
        X_data = g['X_data']
        self.canvas.delete("all")
        self.ar_im = wt.get_array((30, 2, 510, 340),self.title)
        self.img = ImageTk.PhotoImage(Image.fromarray(self.ar_im))
        self.im_w = self.ar_im.shape[1]
        self.im_h = self.ar_im.shape[0]
        self.canvas.create_image(self.x_offset, self.y_offset, anchor=tk.NW, image=self.img)
        
        n = X_data.shape[0]
        X_data.resize((n+1, *self.ar_im[:, :, :3].shape))
        X_data[n] = self.ar_im[:, :, :3]
        f.close()
        if self.run_recording:
            self.master.after(250, self.recording)


    def classify_recording(self):
        if self.run_classify_recording:
            self.run_classify_recording = False
            self.button_classify['text'] = "Classify recording"
            print("End classifying recording")
            return 0
        else:
            self.run_classify_recording = True
            self.button_classify['text'] = "End classifying recording"
            print("Start Classifying recording")
            self.classify_next()
        

    def classify_next(self):
        with h5py.File("data/recording.h5", 'r') as f:        
            g = f['Data']
            X_data = g['X_data']
            print("index_recording = ", self.index_recording)
            print("X_data.shape", X_data.shape)
            if self.index_recording >= X_data.shape[0]:
                f.close()
                print("End of recording reached!")
                self.button_classify['text'] = "Classify recording"
                self.run_classify_recording = False
                return 0
            self.canvas.delete("all")
            self.data = np.zeros((self.grid_y, self.grid_x, 5+len(self.objects)))            
            self.ar_im = X_data[self.index_recording]
            self.img = ImageTk.PhotoImage(Image.fromarray(self.ar_im.astype(np.uint8)))
            self.im_w = self.ar_im.shape[1]
            self.im_h = self.ar_im.shape[0]
            self.canvas.create_image(
                 self.x_offset, self.y_offset, anchor=tk.NW, image=self.img)

            f.close()
            self.index_recording += 1
            
            # then delete first entry:

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y

        # create rectangle if not yet exist
        #if not self.rect:
        
        outline = self.colors[self.lb.curselection()[0]]
        self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, fill="", outline=outline)

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)
        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)


    def on_button_release(self, event):
        coords = (self.start_x-self.x_offset, self.start_y-self.y_offset,
                  event.x-self.x_offset, event.y-self.y_offset)
        # add to data
        #calc midpoint
        midx = (coords[0] + coords[2])/2
        midy = (coords[1] + coords[3])/2

        cell_x = int(midx/self.im_w * self.grid_x)
        cell_y = int(midy/self.im_h * self.grid_y)
     
        bx = (midx - cell_x*34)/34
        by = (midy - cell_y*34)/34

        #calc width and height
        w = np.abs(coords[0] - coords[2])/34
        h = np.abs(coords[1] - coords[3])/34

        # append
        obj = [0 for i in self.objects]
        obj[self.lb.curselection()[0]] = 1
        print(obj)
        self.data[cell_y, cell_x] = np.array([1, bx, by, w, h, *obj])
        self.last_data= (cell_y, cell_x)

    def save_data(self):
        """Save the simulation data into an existing h5 file.
        Args:
            i (int): iteration step
            F (np.array): merit function
            phi (np.array): level set function
            vn (np.array): velocity field
            Q (np.array): optimisation factor
        Returns:
            None.
        """
        if not os.path.isfile(self.hfile):
            with h5py.File(self.hfile, 'w') as f:
                g = f.create_group("Data")
                X_data = g.create_dataset("X_data", shape=(1, 340, 510, 3), maxshape=(None, 340, 510, 3))
                Y_data = g.create_dataset("Y_data", shape=(1, 10, 15, 5+len(self.objects)),
                                          maxshape=(None, 10, 15, 5+len(self.objects)))
                X_data[0] = self.ar_im[:, :, :3]
                Y_data[0] = self.data
                f.close()
            print("Data saved in new file {:}".format(self.hfile))
        else:
            with h5py.File(self.hfile, 'a') as f:
                g = f['Data']
                X_data = g['X_data']
                Y_data = g['Y_data']
                n = X_data.shape[0]
                Y_data.resize((n+1, *self.data.shape))
                X_data.resize((n+1, *self.ar_im[:, :, :3].shape))
                Y_data[n] = self.data
                X_data[n] = self.ar_im[:, :, :3]
                f.close()

            print("Data saved in file {:} !".format(self.hfile))

    def delete_box(self):
        "delete last entry"
        print("deleted last entry")
        self.data[self.last_data[0], self.last_data[1]] = 0
        self.canvas.delete(self.rect)

    def delete_data(self):
        """Delete last entry from file"""
        with h5py.File(self.hfile, 'a') as f:
            g = f['Data']
            X_data = g['X_data']
            Y_data = g['Y_data']
            n = X_data.shape[0]
            Y_data.resize((n-1, *self.data.shape))
            X_data.resize((n-1, *self.ar_im[:, :, :3].shape))
            f.close()

        print("Last entry deleted in file {:} !".format(self.hfile))

running = False

def close(*ignore):
    global running
    running = False
    root.destroy()


parser = argparse.ArgumentParser(description="Data Augmentation")
parser.add_argument('-f', '--file')

args = parser.parse_args()
    
root = tk.Tk()
window = Classifier(root, args.file, "objects.txt")
root.bind('<Escape>', close)
root.protocol("WM_DELETE_WINDOW", close)
root.mainloop()
