#!/usr/bin/env python3

import numpy as np
import pyscreenshot as ImageGrab
from PIL import Image


class Locator:
    """Track location on minimap."""

    def __init__(self):
        print('Initialize Locator-Module!')
        self.map_coords = (595, 9, 740, 160)
        self.street_color1 = np.array([96, 92, 87])
        self.fence_color = np.array([244, 237, 229])
        self.fence_color1 = np.array([238, 238, 238])
        self.image = np.array([])

    def get_minimap(self, im):
        self.im = im
        street1 = np.flip(np.argwhere(
            np.all((im - self.street_color1) == 0, axis=2)), axis=1)
        fence1 = np.flip(np.argwhere(
            np.all((im - self.fence_color) == 0, axis=2)), axis=1)
        fence2 = np.flip(np.argwhere(
            np.all((im - self.fence_color1) == 0, axis=2)), axis=1)
        fence = np.concatenate((fence1, fence2))
        return street1, fence
