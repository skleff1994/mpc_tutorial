'''
This file is an adaptation of Armand Jordana's pendulum example
The original code can be found here : https://github.com/ajordana/value_function/blob/main/value_iteration/pendulum.py
'''

import numpy as np
from matplotlib import colormaps
import matplotlib.pyplot as plt
import os
import cv2
import imageio


folder = "plot/pendulum/value_images/"

file_list = os.listdir(folder)
im_list = [arr for arr in os.listdir(folder) if arr.endswith(".npy")]
nb_im = len(im_list)


value_list = []

for i in range(nb_im):
    im = np.load(folder + str(i) + ".npy")
    value_list.append(im)


sample_lb = np.array([-np.pi, - 6.])
sample_ub = np.array([np.pi, 6.])
N = value_list[0].shape[0]
x = np.linspace(sample_lb[0], sample_ub[0], N)
y = np.linspace(sample_lb[1], sample_ub[1], N)
X, Y = np.meshgrid(x, y)



cmap = colormaps['viridis']




vmin = np.min(np.array(value_list))
vmax = np.max(np.array(value_list))

color_levels = np.linspace(start=vmin, stop=vmax, num=1000)


for i in range(nb_im):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    cs1 = axs.contourf(X, Y, value_list[i],   cmap=cmap, levels=color_levels)
    axs.set_xlabel("$\\theta$", fontsize=18)
    axs.set_ylabel("$\\dot\\theta$", fontsize=18)
    fig.colorbar(cs1)
    plt.savefig(folder +  str(i) + ".png")
    plt.close()



video_name = 'plot/pendulum/video.mp4'

images = [folder + str(i) + ".png" for i in range(nb_im)]

frame = cv2.imread(images[0])
height, width, layers = frame.shape



writer = imageio.get_writer(video_name, format='ffmpeg', fps=4)
print("saving to ")
print(video_name)
print(writer)
for img_path in images:
    img = imageio.v3.imread(img_path)[:, :, :3]
    writer.append_data(img)

writer.close()
print("Closed writer")
        