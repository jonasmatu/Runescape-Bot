import os
import time
import numpy as np
from bot import Bot


class Miner(Bot):
    """Detect and click on Mines """

    def __init__(self, windowInfo):
        print("Initialize Miner!")
        Bot.__init__(self, windowInfo)

    def detect_iron(self, ar_image):
        """Detect iron ores on image.
        Args:
            ar_image (np.array): Input image.
        """
        iron_color_light = np.array([69, 35, 25])  # (r,g,b)
        iron_color_dark = np.array([34, 17, 12])
        coords_light = np.flip(np.argwhere(
            np.all((ar_image-iron_color_light) == 0, axis=2)), axis=1)
        coords_dark = np.flip(np.argwhere(
            np.all((ar_image-iron_color_dark) == 0, axis=2)), axis=1)
        spots = np.concatenate((coords_dark, coords_light))
        centers = self.calc_center(spots)
        return centers

    # def click_basic(self, loc):
    #     Bot.click_basic(loc)

    # def click(self, loc, time=-1, radius=0):
    #     Bot.click(loc, time, radius)

    def calc_center(self, spots):
        """Calculate center of iron ores by collecting all marked pixels
        within a distance of 30 pixels into the center and deleting them
        from the spot list. Iterate through marked pixels until all ores are
        located or 10 max iterations.
        Args:
            spots (list): The location of marked iron ore pixels.
        Returns:
            centers: """
        count = 0
        centers = list()
        while (len(spots) >= 10 and count <= 10):
            count += 1
            marker = list()
            spots2 = list()
            for (x, y) in spots:
                distanceX = abs(x-spots[0][0])
                distanceY = abs(y-spots[0][1])
                if(distanceX <= 30 and distanceY <= 30):
                    marker.append([x, y])
                else:
                    spots2.append([x, y])

            x_min, x_max = marker[0][0], 0
            y_min, y_max = marker[0][1], 0
            for (x, y) in marker:
                if x > x_max:
                    x_max = x
                elif x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                elif y < y_min:
                    y_min = y
            x_val = (x_min + x_max)/2
            y_val = (y_min + y_max)/2
            centers.append([int(x_val), int(y_val)])
            spots = spots2
        return centers

    def get_nearest_ore(self, centers, player):
        """Calculate the nearest ore to player position.
        Args:
            centers (list): the ore centers.
            player (tuple): x,y position of player.
        Returns:
            tuple: nearest ore location."""
        distance = 10000
        location = player
        for loc in centers:
            dist = np.sqrt((player[0]-loc[0])**2 + (player[1]-loc[1])**2)
            if dist < distance:
                distance = dist
                location = (loc[0], loc[1])
        return location

    def mine_at_ore(self, ore):
        """Continuous mining at a ore.
        Args:
            ore (tuple): x,y pos of ore.
        Return:
            None.
        """
