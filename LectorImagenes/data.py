import os
import math
import scipy.io as matlabIo
import skimage.data
import random
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt


def imRead(nbr_image):
    imPath = pathImage(nbr_image)
    im     = skimage.data.load(imPath)

    return im

def pathImage(nbr_image):
    root = os.path.abspath('dataset/Images')
    imPath = root + '/avocados_'+ str(nbr_image)+'.jpg'

    return imPath


