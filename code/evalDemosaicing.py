# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn

import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import sys

from runDemosaicing import *
from utils import *

#Path to your data directory
data_dir = os.path.join('../..', 'data', 'demosaic')

#Path to your output directory
out_dir = os.path.join('../..', 'output', 'demosaic')
mkdir(out_dir)

#List of images
image_names = ['balloon.jpeg','cat.jpg', 'ip.jpg',
            'puppy.jpg', 'squirrel.jpg', 'pencils.jpg',
            'house.png', 'light.png', 'sails.png', 'tree.jpeg']

# image_names = ['balloon.jpeg']
#List of methods you have to implement
methods = ['baseline', 'nn', 'linear', 'adagrad','transf_linear','transf_log'] #Add other methods to this list.

# methods = ['transf_linear','transf_log']

#Global variables.
display = True
error = np.zeros((len(image_names), len(methods)))

#Loop over methods and print results
dashes = '-'*20*(len(methods))
# print(dashes + '\n'+'#\t image \t\t\t\t '+"\t\t\t".join(methods))
print(dashes + '\n'+'#\t image \t\t\t\t '+"\t\t\t".join(methods[:-1]) + "\t"+methods[-1])
print(dashes)
for i, imgname in enumerate(image_names):
    print(f"{i}\t {imgname.ljust(11, ' ')}", end='')
    imgpath = os.path.join(data_dir, imgname)
    for j, m in enumerate(methods):
        err, color_img = runDemosaicing(imgpath, m, display)
        error[i, j] = err
        #Write the output
        outfile_path = os.path.join(out_dir, '{}-{}-dmsc.png'.format(imgname[0:-4], m))
        plt.imsave(outfile_path, color_img)
    scores = "\t\t".join([f"{err:.6f}" for err in error[i]])
    print(f"\t\t {scores}")

#Compute average errors.
print(dashes)
avg = []
for j, _ in enumerate(methods):
    avg.append(error[:, j].mean())
avg_scores = "\t\t".join([f"{err:.6f}" for err in avg])
print(f" \t average \t\t\t {avg_scores}")
print(dashes)

