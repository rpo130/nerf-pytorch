import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import pytransform3d.transformations as pt
import pytransform3d.camera as pc
import pytransform3d.plot_utils as pp


def draw_camera_shape(ax, pose):
    scale = 0.2
    pt.plot_transform(ax=ax, A2B=pose, s=scale)
    # sensor_size = np.array([0.036, 0.024])
    # intrinsic_matrix = np.array([
    #     [0.05, 0, sensor_size[0] / 2.0],
    #     [0, 0.05, sensor_size[1] / 2.0],
    #     [0, 0, 1]
    # ])
    # virtual_image_distance = 1
    # pc.plot_camera(ax, 
    #                 M=intrinsic_matrix, 
    #                 cam2world=pose,
    #                 virtual_image_distance=virtual_image_distance,
    #                 sensor_size=sensor_size)
    
def draw_pose_list(fig, pose_list):
    ax = pp.make_3d_axis(ax_s=1)
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(0,1)
    for pose in pose_list:
        draw_camera_shape(ax, pose)

    plt.show()

def show_pose_entry():
    with open('./data/avt_data_glass_20230118_1/transforms.json', 'r') as f:
        j = f.read()
    import json
    j = json.loads(j)
    print(j)
    pose_list = []
    for i in j['frames']:
        pose_list.append(np.array(i['transform_matrix']))
    fig = plt.figure()
    draw_pose_list(fig, pose_list)

    plt.show()


def compare_depth():
    #2 28 46
    ori = '/home/ai/codebase/nerf-pytorch/data/avt_data_glass_20230118_1/np/34_Depth.npy'
    ori_color = '/home/ai/codebase/nerf-pytorch/data/avt_data_glass_20230118_1/np/34_Color.npy'
    gen = '/home/ai/codebase/nerf-pytorch/logs/avt_data_test/renderonly_test_739999/004_depth.npy'
    opt = '/home/ai/codebase/nerf-pytorch/logs/avt_data_test/renderonly_test_739999/004_depth_ff.npy'

    fig = plt.figure()

    ax = fig.add_subplot(231)
    img_ori = np.load(ori)
    img = img_ori
    im = ax.imshow(img, cmap='gray')
    ax.set_title('ori depth')
    fig.colorbar(im)

    ax = fig.add_subplot(232)
    img_gen = np.load(gen)
    img = img_gen
    im = ax.imshow(img, cmap='gray')
    fig.colorbar(im)

    ax.set_title('vanilla')

    ax = fig.add_subplot(233)
    img_dex = np.load(opt)
    img = img_dex
    im = ax.imshow(img, cmap='gray')
    ax.set_title('dex')
    fig.colorbar(im)

    ax = fig.add_subplot(234)
    img_ori_color = np.load(ori_color)
    img = img_ori_color
    im = ax.imshow(img)
    ax.set_title('ori color')

    ax = fig.add_subplot(235)
    img = img_ori - img_gen
    img = np.abs(img)
    im = ax.imshow(img)
    ax.set_title('diff ori vanilla')
    fig.colorbar(im)

    ax = fig.add_subplot(236)
    img = img_ori - img_dex
    img = np.abs(img)
    im = ax.imshow(img)
    ax.set_title('diff ori dex')
    fig.colorbar(im)



    plt.show()

if __name__ == "__main__":
    show_pose_entry()