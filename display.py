import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file = 'logs/dex_nerf_real_wineglass/renderonly_path_199999/000_depth.png'
img = mpimg.imread(file)
plt.imshow(img)
plt.show()