import os
import numpy as np
import cv2 as cv
from cli import __main__ as cli

# TODO delete the directory if there is the same one (ideally ask the user)
# TODO try out other interpolation methods (bicubic, bilinear, etc..)
mode = 0o666

output_dir = os.path.join(os.path.curdir,__package__, f"{cli.method}_{cli.resolution}")
os.mkdir(output_dir, mode)

for root, _, files in os.walk(cli.input_file):
    for name in files:
        print("Name: ", os.path.join(root, name))
        img = cv.imread(os.path.join(root, name))
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv.filter2D(img, -1, kernel)
        cv.imwrite(os.path.join(output_dir, name), dst)
