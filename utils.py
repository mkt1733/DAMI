"""
@File    : utils.py
@Time    : 2021/8/25 17:19
@Author  : Makoto
@Email   : yucheng.zhang@tum.de
@Software: PyCharm
"""

from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import os
from box import Box
import torchvision.transforms as transforms
Tensor = torch.cuda.FloatTensor



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def path_adapter(path):
    return path.replace('\\', '/')


def cutimage(img):
    """
    get tensor image list
    """
    y = 0
    x = 0
    res = []
    while y < 1281:
        while x < 1793:
            cropped = img.crop((x, y, x + 256, y + 256))
            tensor_cropped = transforms.ToTensor()(cropped)
            res.append(tensor_cropped)
            x += 128
        x = 0
        y += 128
    return res


def get_cfg():
    current_path = os.path.abspath(".")
    yaml_path = os.path.join(current_path, "cfg.yaml")
    conf = Box.from_yaml(filename=yaml_path)
    return conf


def early_stop(parameter, threshold=0.05):
    if len(parameter) > 3:
        if (parameter[-1] - parameter[-2]) / parameter[-1] < threshold:
            return True
    return False
