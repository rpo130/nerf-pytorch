import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

file = '/home/ai/codebase/nerf-pytorch/logs/avt_data_glass_08/testset_001000/000_depth.npy'
img = np.load(file)
# img = mpimg.imread(file)
# plt.imshow(img)
# plt.show()
print(np.mean(img))
print(np.min(img))
print(np.max(img))