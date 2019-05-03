import pyautogui as pg
import random
import numpy as np


class Bot:
    """Generic class for bot modules."""

    def __init__(self, windowInfo):
        """Init Bot module.
        Args:
            windowInfo (tuple): (x,y,w,h) position of RS-Window
        Returns:
            None."""
        self.winInf = windowInfo

    def click_basic(self, loc, time=0, radius=0):
        """Click on location in game without fancy mouse movement.
        Args:
            loc (tuple): location to click in screen-coords.
        Returns:
            None."""
        if radius != 0:
            x = loc[0] + random.randint(-radius, radius)
            y = loc[1] + random.randint(-radius, radius)
            loc = (x, y)

        pg.click(self.winInf[0]+loc[0], self.winInf[1]+loc[1])

    def click(self, loc, time=-1, radius=0):
        """Click on location in game.
        Args:
            loc (tuple): location to click on in screen-coords
            time (float): time of mouse movement. 
                          If -1 then calculate by distance.
            radius (int): area around location
        Returns:
            None."""
        print('Clicking!')
        if radius != 0:
            x = loc[0] + random.randint(-radius, radius)
            y = loc[1] + random.randint(-radius, radius)
            loc = (x, y)
        if time == -1:
            (x, y) = pg.position()
            time = np.sqrt((x-loc[0])**2 + (y-loc[1])**2) / 200
        pg.moveTo(loc[0] + self.winInf[0], loc[1] +
                  self.winInf[1], time, pg.easeOutElastic)
        pg.click()
