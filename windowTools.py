import pyscreenshot as ImageGrab
from mss import mss
import os
import time
import numpy as np
from PIL import Image

sct = mss()


def screen_grab():
    """Grab the whole screen and save to png"""
    box = get_windowinfo()
    im = ImageGrab.grab((box['xpos'], box['ypos'], box['xpos'] + box['width'],
                         box['ypos'] + box['height'])).convert('RGBA')
    im.save(os.getcwd() + '/full_snap_' + str(int(time.time())) + '.png')


def get_full_screen():
    winInf = get_wininfo()
    monitor = {"left": winInf[0], "top": winInf[1],
               "width": winInf[2], "height": winInf[3]}
    sct_im = sct.grab(monitor)
    return sct_im


def crop_to_array(im, box):
    """Convert image to np.array
    Args:
        im (sct image): The image.
        box (tuple): x, y, width, height
    Returns:
        np.array: in r,g,b format
    """
    x, y, w, h = box
    ar_im = np.flip(np.array(im, dtype=np.uint8)[y:y+h, x:x+w, :3], 2)
    return ar_im


def get_array(box):
    """box = (x, y, w, h)"""
    winInf = get_wininfo()
    monitor = {"left": winInf[0] + box[0], "top": winInf[1] + box[1],
               "width": box[2], "height": box[3]}
    ar_im = np.flip(np.array(sct.grab(monitor), dtype=np.uint8)[:, :, :3], 2)
    return ar_im


def get_wininfo():
    """Get position and attributes of the runescape window.

    Returns:
        tuple: (x, y, width, height)
    """
    f = os.popen(r'xwininfo -name "Old School RuneScape"')
    data = f.readlines()
    x = int(data[3].split(':')[1].replace(' ', '')[:-1])
    y = int(data[4].split(':')[1].replace(' ', '')[:-1])
    w = int(data[7].split(':')[1].replace(' ', '')[:-1])
    h = int(data[8].split(':')[1].replace(' ', '')[:-1])
    return (x, y, w, h)


# old, slow functions

# def gamefield_grab():
#     """Grab the core gaming field and return the image."""
#     game_field = (29, 0, 512+29, 334)
#     im = get_array(game_field)
#     return np.array(im)


# def get_full_screen(wininf):
#     """Grab the whole screen and return as image.
#     Args:
#         windowInfo (x, y, x+w, x+h): window info"""
#     print(wininf)
#     im = ImageGrab.grab(
#         (wininf[0], wininf[1], wininf[2], wininf[3])).convert('RGBA')
#     return im


# def snap_grab(x, y, w, h, name='/part_snap_' + str(int(time.time()))):
#     wp = get_windowinfo()
#     print(wp)
#     box = (x + wp['xpos'], y + wp['ypos'],
#            x + w + wp['xpos'], y + h + wp['ypos'])
#     print(box)
#     im = ImageGrab.grab(box).convert('RGBA')
#     im.save(os.getcwd() + '/' + name + '.png', format='PNG')
#     print('saving image')


# def get_array(box):
#     """Get the image of an box in the RS window.
#     Args:
#         box (tupel): (x, y, w, h) image position and size.
#     Returns:
#         np.array: the image in RGBA np.array format.
#     """
#     wp = get_windowinfo()
#     box = (box[0] + wp['xpos'], box[1] + wp['ypos'],
#            box[0] + box[2] + wp['xpos'], box[1] + box[3] + wp['ypos'])
#     im = ImageGrab.grab(box).convert('RGBA')
#     return np.array(im)


# def get_image(box):
#     """Get the image of an box in the RS window.
#     Args:
#         box (tupel): (x, y, w, h) image position and size.
#     Returns:
#         im (Image): the image in RGBA format.
#     """
#     wp = get_windowinfo()
#     box = (box[0] + wp['xpos'], box[1] + wp['ypos'],
#            box[0] + box[2] + wp['xpos'], box[1] + box[3] + wp['ypos'])
#     im = ImageGrab.grab(box).convert('RGBA')
#     return im


# def get_windowinfo():
#     """Get position and attributes of the runescape window.

#     Returns:
#         dict: {xpos, ypos, width, height}
#     """
#     info = {'xpos': None, 'ypos': None, 'width': None, 'height': None}
#     f = os.popen(r'xwininfo -name "Old School RuneScape"')
#     data = f.readlines()
#     info['xpos'] = int(data[3].split(':')[1].replace(' ', '')[:-1])
#     info['ypos'] = int(data[4].split(':')[1].replace(' ', '')[:-1])
#     info['width'] = int(data[7].split(':')[1].replace(' ', '')[:-1])
#     info['height'] = int(data[8].split(':')[1].replace(' ', '')[:-1])
#     return info
