#!/usr/bin/env python3

import numpy as np
import pyscreenshot as ImageGrab
from PIL import Image


class Locator:
    """Track location on minimap."""

    def __init__(self):
        print('Initialize Locator-Module!')
        self.map_coords = (595, 9, 740, 160)
        self.street_color1 = np.array([109, 106, 98])
        self.street_color2 = np.array([109, 104, 98])
        self.street_color3 = np.array([109, 105, 98])
        self.fence_color = np.array([245, 247, 237])
        self.fence_color1 = np.array([238, 238, 238])
        self.image = np.array([])

    def get_minimap(self, im):
        self.im = im
        street1 = np.flip(np.argwhere(
            np.all((im - self.street_color1) == 0, axis=2)), axis=1)
        street2 = np.flip(np.argwhere(
            np.all((im - self.street_color2) == 0, axis=2)), axis=1)
        street3 = np.flip(np.argwhere(
            np.all((im - self.street_color3) == 0, axis=2)), axis=1)
        fence1 = np.flip(np.argwhere(
            np.all((im - self.fence_color) == 0, axis=2)), axis=1)
        fence2 = np.flip(np.argwhere(
            np.all((im - self.fence_color1) == 0, axis=2)), axis=1)
        # for y in range(len(self.im)):
        #     for x in range(len(self.im[0])):
        #         if (self.im[y][x] == self.street_color1).all():
        #             street.append((x, y))
        #         elif (self.im[y][x] == self.street_color2).all():
        #             street.append((x, y))
        #         elif (self.im[y][x] == self.street_color3).all():
        #             street.append((x, y))
        #         elif (np.sum(np.abs(self.im[y][x] - self.fence_color)) <= 10):
        #             fence.append((x, y))
        #         elif (self.im[y][x] == self.fence_color1).all():
        #             fence.append((x, y))
        street = np.concatenate((street1, street2, street3))
        fence = np.concatenate((fence1, fence2))
        return street, fence
